"""
pywarper.warpers
==============
Spatial warping and profiling utilities for **neuronal skeleton reconstructions**.

This module takes a neuronal tree (nodes+edges) and the previously‑computed
ON/OFF Starburst Amacrine Cell (SAC) surface mapping in order to

1. **Warp the skeleton into the flattened SAC coordinate frame** (`warp_nodes`).
   Each node is locally re‑registered with a polynomial least‑squares fit
   (`local_ls_registration`) that references both SAC layers so that depth is
   preserved relative to the curved retina.
2. **Compute depth (z) profiles** (`get_z_profile`).  Edge lengths are first binned
   directly (histogram) and then re‑estimated with a Kaiser–Bessel gridding
   kernel to obtain a smooth 1‑D density across the inner plexiform layer.
3. **Compute planar (xy) density maps** (`get_xy_profile`).  Dendritic length is
   accumulated on a user‑defined 2‑D grid and optionally Gaussian‑smoothed for
   visualisation or group statistics.

Key algorithms
--------------
* **Polynomial local registration** – For every node we fit a 2‑D polynomial
  basis (up to a configurable `max_order`) to the positions of neighbouring
  SAC‑band sample points, solving three separate least‑squares systems in one
  go with `numpy.linalg.lstsq`.  A single **KDTree** (SciPy) accelerates the
  neighbourhood queries.
* **Kaiser–Bessel gridding** – The 1‑D `gridder1d` function emulates the
  non‑uniform FFT gridding scheme used by older MATLAB code, yielding the exact
  same numerical output but in fully vectorised NumPy.
"""

import time
from copy import deepcopy
from importlib import metadata as _metadata
from pathlib import Path

import numpy as np
import skeliner as sk
import trimesh
from numpy.linalg import lstsq
from scipy.ndimage import gaussian_filter
from scipy.spatial import KDTree
from scipy.special import i0
from skeliner._core import _bfs_parents
from skeliner.dataclass import Skeleton
from skeliner.dx import _ellipsoid_aabb, _voxelize_union

from .surface import build_mapping, fit_surface
from .utils import _ensure_new_format, build_surface_correspondences, resolve_conformal_jump

_PYWARPER_VERSION = _metadata.version("pywarper")


def poly_basis_2d(x: np.ndarray, y: np.ndarray, max_order: int) -> np.ndarray:
    """
    Return the full 2-D polynomial basis up to total order *max_order*
    for coordinates (x, y).  Shape:  (len(x), n_terms)

    Order layout ≡ original code:
        [1,
         x, y,
         x²,  x·y,  y²,      # order 2
         x³,  x²y, x y², y³, …]
    """
    cols = [np.ones_like(x), x, y]  # constant + linear
    for order in range(2, max_order + 1):
        for ox in range(order + 1):
            oy = order - ox
            cols.append(x**ox * y**oy)
    return np.stack(cols, axis=1)  # (N, n_terms)


def _build_local_ls_state(
    input_pts_list: list[np.ndarray],
    output_pts_list: list[np.ndarray],
    *,
    window: float,
    max_order: int,
) -> dict[str, np.ndarray | KDTree | float | int]:
    in_all = np.vstack(input_pts_list)
    out_all = np.vstack(output_pts_list)
    all_xy = in_all[:, :2]
    query_r = window * np.sqrt(2.0)
    tree = KDTree(all_xy)
    return {
        "in_all": in_all,
        "out_all": out_all,
        "all_xy": all_xy,
        "query_r": query_r,
        "tree": tree,
        "window": float(window),
        "max_order": int(max_order),
    }


def _apply_local_ls_state(
    nodes: np.ndarray,
    state: dict[str, np.ndarray | KDTree | float | int],
    *,
    warn: bool = True,
) -> np.ndarray:
    transformed_nodes = np.zeros_like(nodes)

    in_all = state["in_all"]
    out_all = state["out_all"]
    all_xy = state["all_xy"]
    query_r = state["query_r"]
    tree = state["tree"]
    window = state["window"]
    max_order = state["max_order"]

    workers = -1 if nodes.shape[0] >= 512 else 1
    idx_lists = tree.query_ball_point(nodes[:, :2], r=query_r, workers=workers)

    for k, (x, y, z) in enumerate(nodes):
        idx = np.array(idx_lists[k], dtype=int)

        lx, ux = x - window, x + window
        ly, uy = y - window, y + window
        mask_rect = (
            (all_xy[idx, 0] >= lx)
            & (all_xy[idx, 0] <= ux)
            & (all_xy[idx, 1] >= ly)
            & (all_xy[idx, 1] <= uy)
        )

        idx = idx[mask_rect]
        if idx.size == 0:
            if warn:
                print(
                    f"[pywarper] Warning: no neighbours for node {k} at ({x:.2f}, {y:.2f}, {z:.2f})"
                )
            transformed_nodes[k] = nodes[k]
            continue

        this_in = in_all[idx]
        this_out = out_all[idx]

        if this_in.shape[0] < 12:
            if warn:
                print(
                    f"[pywarper] Warning: not enough neighbours for node {k} at ({x:.2f}, {y:.2f}, {z:.2f})"
                )
            transformed_nodes[k] = nodes[k]
            continue

        shift_xy = this_in[:, :2].mean(axis=0)
        xin, yin, zin = (
            this_in[:, 0] - shift_xy[0],
            this_in[:, 1] - shift_xy[1],
            this_in[:, 2],
        )

        xout, yout, zout = (
            this_out[:, 0] - shift_xy[0],
            this_out[:, 1] - shift_xy[1],
            this_out[:, 2],
        )

        base_terms = poly_basis_2d(xin, yin, max_order)
        X = np.hstack((base_terms, base_terms * zin[:, None]))

        T, _, _, _ = lstsq(X, np.column_stack((xout, yout, zout)), rcond=None)

        nx, ny = nodes[k, 0] - shift_xy[0], nodes[k, 1] - shift_xy[1]
        basis_eval = poly_basis_2d(np.array([nx]), np.array([ny]), max_order).ravel()
        vec = np.concatenate((basis_eval, z * basis_eval))
        new_pos = vec @ T

        new_pos[:2] += shift_xy
        transformed_nodes[k] = new_pos

    return transformed_nodes


def local_ls_registration(
    nodes: np.ndarray,
    input_pts_list: list[np.ndarray],
    output_pts_list: list[np.ndarray],
    window: float = 5.0,
    max_order: int = 2,
    warn: bool = True,
) -> np.ndarray:
    """
    Local polynomial least-squares registration using a single KDTree
    built from all surface control points.
    """
    state = _build_local_ls_state(
        input_pts_list,
        output_pts_list,
        window=window,
        max_order=max_order,
    )
    return _apply_local_ls_state(nodes, state, warn=warn)


def _select_flattening_surfaces(
    nodes: np.ndarray,
    median_depths: dict[str, float],
    surface_order: list[str],
) -> tuple[str, str]:
    """
    Pick the two surfaces that bracket the cell's median depth.

    If the cell is shallower than all surfaces, returns the two shallowest.
    If deeper than all, returns the two deepest.
    """
    cell_median_z = float(np.median(nodes[:, 2]))

    # surface_order is already sorted by depth
    depths = [median_depths[tag] for tag in surface_order]

    if len(surface_order) < 2:
        raise ValueError("Need at least two surfaces for flattening.")

    # Find the pair that brackets cell_median_z
    for i in range(len(depths) - 1):
        if cell_median_z <= depths[i + 1]:
            return surface_order[i], surface_order[i + 1]

    # Cell is deeper than all surfaces — use the two deepest
    return surface_order[-2], surface_order[-1]


def warp_nodes(
    nodes: np.ndarray,
    surface_mapping: dict,
    conformal_jump: int | None = None,
    backward_compatible: bool = False,
    flattening_surfaces: tuple[str, str] | None = None,
    verbose: bool = False,
) -> tuple[np.ndarray, dict[str, float]]:
    """
    Warp *nodes* using a surface mapping.

    Parameters
    ----------
    flattening_surfaces : tuple[str, str] or None
        Tags of the two surfaces to use for local LS registration.
        When None, the two surfaces bracketing the cell's median depth
        are selected automatically.
    verbose : bool
        If True, print which surfaces were selected for flattening.

    Returns
    -------
    warped : (N, 3) array
    median_depths : dict mapping surface tag -> median z
    """
    resolved_jump = resolve_conformal_jump(surface_mapping, conformal_jump)
    input_pts_list, output_pts_list, median_depths = build_surface_correspondences(
        surface_mapping,
        conformal_jump=resolved_jump,
        backward_compatible=backward_compatible,
    )

    mapping = _ensure_new_format(surface_mapping)
    surface_order = mapping["surface_order"]

    # Select the two surfaces for flattening
    if flattening_surfaces is None:
        flattening_surfaces = _select_flattening_surfaces(
            nodes, median_depths, surface_order,
        )
    if verbose:
        print(
            f"[pywarper] Flattening with surfaces: "
            f"\"{flattening_surfaces[0]}\" (z={median_depths[flattening_surfaces[0]]:.2f}) "
            f"and \"{flattening_surfaces[1]}\" (z={median_depths[flattening_surfaces[1]]:.2f})"
        )

    # Filter to only the two selected surfaces
    selected_input = []
    selected_output = []
    for i, tag in enumerate(surface_order):
        if tag in flattening_surfaces:
            selected_input.append(input_pts_list[i])
            selected_output.append(output_pts_list[i])

    warped = local_ls_registration(nodes, selected_input, selected_output)

    return warped, median_depths


def normalize_nodes(
    nodes: np.ndarray,
    median_depths: dict[str, float] | None = None,
    anchors: tuple[str, str] = ("on_sac", "off_sac"),
    anchor_pos: tuple[float, float] = (0.0, 12.0),
    *,
    med_z_on: float | None = None,
    med_z_off: float | None = None,
    on_sac_pos: float | None = None,
    off_sac_pos: float | None = None,
) -> np.ndarray:
    """
    Normalize the z-coordinates based on two anchor surfaces.

    Parameters
    ----------
    nodes : np.ndarray
        (N, 3) coordinates.
    median_depths : dict[str, float] or None
        Mapping of surface tag -> median z depth.
        When None, *med_z_on* and *med_z_off* must be supplied instead.
    anchors : tuple[str, str]
        Tags of the two anchor surfaces.
    anchor_pos : tuple[float, float]
        Desired normalized positions for the two anchors.
    med_z_on, med_z_off : float | None
        Legacy keywords for the two-surface case. Used when *median_depths*
        is None.
    on_sac_pos, off_sac_pos : float | None
        Legacy keywords that override *anchor_pos*.

    Returns
    -------
    np.ndarray
        (N, 3) with normalized z.
    """
    # ---- resolve legacy call convention ------------------------------------
    if median_depths is None:
        if med_z_on is None or med_z_off is None:
            raise ValueError(
                "Either median_depths dict or both med_z_on and med_z_off must be provided."
            )
        median_depths_dict: dict[str, float] = {
            "on_sac": float(med_z_on),
            "off_sac": float(med_z_off),
        }
    elif isinstance(median_depths, (int, float)):
        # Positional scalar: treat as med_z_on for backward compat
        if med_z_off is None:
            raise ValueError("med_z_off must be provided when median_depths is a scalar (legacy API).")
        median_depths_dict = {"on_sac": float(median_depths), "off_sac": float(med_z_off)}
    else:
        median_depths_dict = median_depths

    if on_sac_pos is not None:
        anchor_pos = (on_sac_pos, anchor_pos[1] if off_sac_pos is None else off_sac_pos)
    if off_sac_pos is not None and on_sac_pos is None:
        anchor_pos = (anchor_pos[0], off_sac_pos)

    z_a = median_depths_dict[anchors[0]]
    z_b = median_depths_dict[anchors[1]]

    normalized_nodes = nodes.copy().astype(float)
    rel_depth = (nodes[:, 2] - z_a) / (z_b - z_a)
    normalized_nodes[:, 2] = anchor_pos[0] + rel_depth * (anchor_pos[1] - anchor_pos[0])

    return normalized_nodes


def warp_skeleton(
    skel: Skeleton,
    surface_mapping: dict,
    voxel_resolution: float | list[float | int] = [1.0, 1.0, 1.0],
    on_sac_pos: float = 0.0,
    off_sac_pos: float = 12.0,
    z_profile_extent: list[float | int] | None = None,  # [z_min, z_max]
    z_profile_bin_size: float | int = 1.0,
    z_profile_hdr_mass: float | int = 0.95,
    z_profile_include_soma: bool = False,
    z_profile_voxel_size: float | None = None,
    xy_profile_extents: list[float | int] | None = None,  # [x_min, x_max, y_min, y_max]
    xy_profile_bin_size: float | int = 20.0,
    xy_profile_smooth: float = 1.0,
    xy_profile_include_soma: bool = False,
    xy_profile_voxel_size: float | None = None,
    radius_metric: str | None = None,
    skeleton_nodes_scale: float = 1.0,
    conformal_jump: int | None = None,
    backward_compatible: bool = False,
    verbose: bool = False,
) -> Skeleton:
    """
    Applies a local surface flattening (warp) to a neuronal skeleton using the results
    of previously computed surface mappings.

    Parameters
    ----------
    nodes : np.ndarray
        (N, 3) array of [x, y, z] coordinates for the skeleton to be warped.
    edges : np.ndarray
        (E, 2) array of indices defining connectivity between nodes.
    radii : np.ndarray
        (N,) array of radii corresponding to each node.
    surface_mapping : dict
        Dictionary containing keys:
          - "mapped_min_positions" : np.ndarray
              (X*Y, 2) mapped coordinates for one surface band (e.g., "min" band).
          - "mapped_max_positions" : np.ndarray
              (X*Y, 2) mapped coordinates for the other surface band (e.g., "max" band).
          - "thisVZminmesh" : np.ndarray
              (X, Y) mesh representing the first surface (“min” band) in 3D space.
          - "thisVZmaxmesh" : np.ndarray
              (X, Y) mesh representing the second surface (“max” band) in 3D space.
          - "thisx" : np.ndarray
              1D array of x-indices (possibly downsampled) used during mapping.
          - "thisy" : np.ndarray
              1D array of y-indices (possibly downsampled) used during mapping.
    conformal_jump : int, default=1
        Step size used in the conformal mapping (downsampling factor).
    verbose : bool, default=False
        If True, prints timing and progress information.

    Returns
    -------
    dict
        Dictionary containing:
          - "nodes": np.ndarray
              (N, 3) warped [x, y, z] coordinates after applying local registration.
          - "edges": np.ndarray
              (E, 2) connectivity array (passed through unchanged).
          - "radii": np.ndarray
              (N,) radii array (passed through unchanged).
          - "medVZmin": float
              Median z-value of the “min” surface mesh within the region of interest.
          - "medVZmax": float
              Median z-value of the “max” surface mesh within the region of interest.

    Notes
    -----
    1. The function extracts a subregion of the surfaces according to thisx/thisy and
       conformal_jump, matching the flattening step used in the mapping.
    2. Each node in `nodes` is then warped via local least-squares registration
       (`local_ls_registration`), referencing top (min) and bottom (max) surfaces.
    3. The median z-values (medVZmin, medVZmax) are recorded, which often serve as
       reference planes in further analyses.
    """

    nodes = (
        skel.nodes.astype(float) * skeleton_nodes_scale
    )  # scale to the surface unit, which is often μm

    if verbose:
        print("[pywarper] Warping skeleton...")
        start_time = time.time()
    warped_nodes, median_depths = warp_nodes(
        nodes,
        surface_mapping,
        conformal_jump=conformal_jump,
        backward_compatible=backward_compatible,
        verbose=verbose,
    )

    # Record which surfaces were used for flattening (same logic as warp_nodes)
    mapping_fmt = _ensure_new_format(surface_mapping)
    flattening_surfaces = _select_flattening_surfaces(
        nodes, median_depths, mapping_fmt["surface_order"],
    )

    normalized_nodes = normalize_nodes(
        warped_nodes,
        median_depths=median_depths,
        anchor_pos=(on_sac_pos, off_sac_pos),
    )

    normalized_nodes /= skeleton_nodes_scale

    if verbose:
        print(f"    done in {time.time() - start_time:.2f} seconds.")

    normalized_soma = deepcopy(skel.soma)
    normalized_soma.center = (
        normalized_nodes[0] * voxel_resolution
    )  # soma is at the first node

    skel_norm = Skeleton(
        soma=normalized_soma,
        nodes=normalized_nodes * voxel_resolution,
        edges=skel.edges,
        radii=skel.radii,
        ntype=skel.ntype,
        node2verts=skel.node2verts,
        vert2node=skel.vert2node,
        meta=skel.meta.copy(),
    )

    z_profiles = {
        measure: get_z_profile(
            skel_norm,
            extent=z_profile_extent,
            bin_size=z_profile_bin_size,
            hdr_mass=z_profile_hdr_mass,
            measure=measure,
            include_soma=z_profile_include_soma,
            voxel_size=z_profile_voxel_size,
            radius_metric=radius_metric,
        )
        for measure in ["length", "volume"]
    }
    xy_profiles = {
        measure: get_xy_profile(
            skel_norm,
            extents=xy_profile_extents,
            bin_size=xy_profile_bin_size,
            smooth=xy_profile_smooth,
            measure="length",
            include_soma=xy_profile_include_soma,
            voxel_size=xy_profile_voxel_size,
            radius_metric=radius_metric,
        )
        for measure in ["length", "volume"]
    }

    skel_norm.extra = {
        "prenormed_nodes": warped_nodes
        * voxel_resolution,  # keep the pre-normed warped nodes for future use
        "median_depths": median_depths,
        "med_z_on": float(median_depths.get("on_sac", 0.0)),
        "med_z_off": float(median_depths.get("off_sac", 0.0)),
        "flattening_surfaces": flattening_surfaces,
        "z_profiles": z_profiles,
        "xy_profiles": xy_profiles,
    }
    skel_norm.meta.update(
        {
            "pywarper_version": _PYWARPER_VERSION,
            "warped_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
    )

    return skel_norm


def warp_mesh(
    mesh: trimesh.Trimesh,  # mostly nm
    surface_mapping: dict,  # mostly μm
    conformal_jump: int | None = None,
    on_sac_pos: float = 0.0,  # μm
    off_sac_pos: float = 12.0,  # μm
    mesh_vertices_scale: float = 1.0,  # scale factor for mesh vertices, e.g., 1e-3 for nm to μm
    backward_compatible: bool = False,
    verbose: bool = False,
) -> trimesh.Trimesh:
    """
    Applies a local surface flattening (warp) to a 3D mesh using the results
    of previously computed surface mappings.
    """

    vertices = (
        mesh.vertices.astype(float) * mesh_vertices_scale
    )  # scale to the surface unit, which is often μm

    if verbose:
        print("[pywarper] Warping mesh...")
        start_time = time.time()
    warped_vertices, median_depths = warp_nodes(
        vertices,
        surface_mapping,
        conformal_jump=conformal_jump,
        backward_compatible=backward_compatible,
        verbose=verbose,
    )

    normalized_vertices = normalize_nodes(
        warped_vertices,
        median_depths=median_depths,
        anchor_pos=(on_sac_pos, off_sac_pos),
    )

    if verbose:
        print(f"    done in {time.time() - start_time:.2f} seconds.")

    # Create a new mesh with the warped vertices
    warped_mesh = trimesh.Trimesh(
        vertices=normalized_vertices
        / mesh_vertices_scale,  # rescale back to original units
        faces=mesh.faces,
        process=False,  # no processing
    )
    warped_mesh.metadata = mesh.metadata.copy()  # copy metadata
    warped_mesh.metadata["median_depths"] = median_depths
    warped_mesh.metadata["med_z_on"] = float(median_depths.get("on_sac", 0.0))
    warped_mesh.metadata["med_z_off"] = float(median_depths.get("off_sac", 0.0))
    warped_mesh.metadata["conformal_jump"] = conformal_jump
    warped_mesh.metadata["surface_mapping"] = surface_mapping
    warped_mesh.metadata["on_sac_pos"] = on_sac_pos
    warped_mesh.metadata["off_sac_pos"] = off_sac_pos
    warped_mesh.metadata["pywarper_meta"] = {
        "version": _PYWARPER_VERSION,
        "warped_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    return warped_mesh


# =====================================================================
# helpers for get_z_profile() and get_xy_profile()
# =====================================================================


def segment_lengths(skel: Skeleton) -> tuple[np.ndarray, np.ndarray]:
    """
    Per-edge cable length at every non-root node & its mid-point.

    Returns
    -------
    lengths : (N,)  nonzero only at child nodes (each entry is edge length)
    mid     : (N,3) midpoints (same convention as before)
    """
    parent = np.asarray(
        _bfs_parents(skel.edges, len(skel.nodes), root=0), dtype=np.int64
    )
    child = np.where(parent != -1)[0]  # nodes with a parent

    a = skel.nodes[child]  # child coords
    b = skel.nodes[parent[child]]  # parent coords
    vec = b - a
    L = np.linalg.norm(vec, axis=1)  # edge length

    # midpoints in a full (N,3) array
    mid = skel.nodes.copy()
    mid[child] += 0.5 * vec

    lengths = np.zeros(len(skel.nodes), dtype=float)
    lengths[child] = L
    return lengths, mid


def z_slince_volumes(
    skel: Skeleton,
    *,
    voxel_size: float | None = None,
    include_soma: bool = False,
    radius_metric: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Union-correct **per-z slice** volumes as weights + sample positions.

    Returns
    -------
    volumes : (K,) float
        Union volume per *non‑empty* z‑slice (in unit³).
    mid     : (K, 3) float
        Sample positions for each slice: (x̄, ȳ, z_center), where x̄,ȳ are
        slice area‑weighted centroids and z_center is the slice center.
    """
    # choose radii column for voxelizer / bbox
    if radius_metric is None:
        radius_metric = skel.recommend_radius()[0]
    radii = np.asarray(skel.radii[radius_metric], dtype=np.float64).reshape(-1)

    # tight auto‑bbox (like dx.volume/area)
    lo_nodes = (skel.nodes - radii[:, None]).min(axis=0)
    hi_nodes = (skel.nodes + radii[:, None]).max(axis=0)
    if include_soma and getattr(skel, "soma", None) is not None:
        slo, shi = _ellipsoid_aabb(skel.soma)
        lo = np.minimum(lo_nodes, slo)
        hi = np.maximum(hi_nodes, shi)
    else:
        lo, hi = lo_nodes, hi_nodes

    # one voxelization of the union
    occ, h, (nx, ny, nz), lo, hi = _voxelize_union(
        skel, radii, lo, hi, voxel_size=voxel_size, include_soma=include_soma
    )
    if nz == 0:
        return np.zeros(0, dtype=float), np.zeros((0, 3), dtype=float)

    # volume per slice (occupied count × voxel volume)
    vol_all = occ.sum(axis=(0, 1)).astype(np.float64) * (h**3)  # (nz,)

    # keep only non‑empty slices
    mask = vol_all > 0.0
    if not mask.any():
        return np.zeros(0, dtype=float), np.zeros((0, 3), dtype=float)
    vol = vol_all[mask]
    k = np.where(mask)[0]  # slice indices kept

    # z center for each kept slice
    zc = lo[2] + (k + 0.5) * h

    # area‑weighted centroids per kept slice (optional but nice to have)
    xs = lo[0] + (np.arange(nx) + 0.5) * h  # (nx,)
    ys = lo[1] + (np.arange(ny) + 0.5) * h  # (ny,)
    occ_sel = occ[:, :, mask]  # (nx, ny, K)

    counts = occ_sel.sum(axis=(0, 1)).astype(np.float64)  # (K,)
    sum_x = (occ_sel * xs[:, None, None]).sum(axis=(0, 1))  # (K,)
    sum_y = (occ_sel * ys[None, :, None]).sum(axis=(0, 1))  # (K,)

    with np.errstate(invalid="ignore", divide="ignore"):
        xbar = np.where(counts > 0, sum_x / counts, (lo[0] + hi[0]) / 2.0)
        ybar = np.where(counts > 0, sum_y / counts, (lo[1] + hi[1]) / 2.0)

    mid = np.column_stack((xbar, ybar, zc)).astype(np.float64)
    return vol, mid


def xy_column_volume(
    skel: Skeleton,
    *,
    voxel_size: float | None = None,
    include_soma: bool = False,
    radius_metric: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Union-correct **per-(x,y) column** volumes + sample positions.

    Returns
    -------
    vol : (K,) float
        Volume in each non-empty (x,y) column (integrated over z), unit³.
    mid : (K,3) float
        (x_center, y_center, z̄) for that column, where z̄ is the z-centroid
        of occupied voxels in the column (handy but not used by XY maps).
    """
    if radius_metric is None:
        radius_metric = skel.recommend_radius()[0]
    radii = np.asarray(skel.radii[radius_metric], dtype=np.float64).reshape(-1)

    lo_nodes = (skel.nodes - radii[:, None]).min(axis=0)
    hi_nodes = (skel.nodes + radii[:, None]).max(axis=0)
    if include_soma and getattr(skel, "soma", None) is not None:
        slo, shi = _ellipsoid_aabb(skel.soma)
        lo = np.minimum(lo_nodes, slo)
        hi = np.maximum(hi_nodes, shi)
    else:
        lo, hi = lo_nodes, hi_nodes

    occ, h, (nx, ny, nz), lo, hi = _voxelize_union(
        skel, radii, lo, hi, voxel_size=voxel_size, include_soma=include_soma
    )
    if nx == 0 or ny == 0:
        return np.zeros(0), np.zeros((0, 3))

    # volume per (x,y) column
    vol_xy = occ.sum(axis=2).astype(np.float64) * (h**3)  # (nx, ny)
    mask = vol_xy > 0.0
    if not mask.any():
        return np.zeros(0), np.zeros((0, 3))

    # centers
    xs = lo[0] + (np.arange(nx) + 0.5) * h
    ys = lo[1] + (np.arange(ny) + 0.5) * h
    Xc, Yc = np.meshgrid(xs, ys, indexing="ij")

    # z centroid per column (nice to have)
    zc = lo[2] + (np.arange(nz) + 0.5) * h  # (nz,)
    occ_flat = occ.reshape(nx * ny, nz).astype(np.float64)
    counts = occ_flat.sum(axis=1)  # (#columns,)
    sum_z = occ_flat @ zc  # (#columns,)
    with np.errstate(invalid="ignore", divide="ignore"):
        zbar = np.where(counts > 0, sum_z / counts, (lo[2] + hi[2]) / 2.0)
    zbar = zbar.reshape(nx, ny)

    vol = vol_xy[mask]
    mid = np.column_stack((Xc[mask], Yc[mask], zbar[mask])).astype(np.float64)
    return vol, mid


def gridder1d(
    z_samples: np.ndarray,
    density: np.ndarray,
    n: int,
) -> np.ndarray:
    """
    Kaiser–Bessel gridding kernel in 1-D   (α=2, W=5)

    Vectorised patch-accumulation: identical output, ~2× faster.
    """
    if z_samples.shape != density.shape:
        raise ValueError("z_samples and density must have the same shape")

    # ------------------------------------------------------------------
    # Constants and lookup table (unchanged)
    # ------------------------------------------------------------------
    alpha, W, err = 2, 5, 1e-3
    S = int(np.ceil(0.91 / err / alpha))
    beta = np.pi * np.sqrt((W / alpha * (alpha - 0.5)) ** 2 - 0.8)

    s = np.linspace(-1, 1, 2 * S * W + 1)
    F_kbZ = i0(beta * np.sqrt(1 - s**2))
    F_kbZ /= F_kbZ.max()

    # ------------------------------------------------------------------
    # Fourier transform of the 1-D kernel (unchanged)
    # ------------------------------------------------------------------
    Gz = alpha * n
    z = np.arange(-Gz // 2, Gz // 2)
    arg = (np.pi * W * z / Gz) ** 2 - beta**2

    kbZ = np.empty_like(arg, dtype=float)
    pos, neg = arg > 1e-12, arg < -1e-12
    kbZ[pos] = np.sin(np.sqrt(arg[pos])) / np.sqrt(arg[pos])
    kbZ[neg] = np.sinh(np.sqrt(-arg[neg])) / np.sqrt(-arg[neg])
    kbZ[~(pos | neg)] = 1.0
    kbZ *= np.sqrt(Gz)

    # ------------------------------------------------------------------
    # Oversampled grid and *vectorised* accumulation
    # ------------------------------------------------------------------
    n_os = Gz
    out = np.zeros(n_os, dtype=float)

    centre = n_os / 2 + 1  # 1-based like MATLAB
    nz = centre + n_os * z_samples  # (N,)

    half_w = (W - 1) // 2
    lz_offsets = np.arange(-half_w, half_w + 1)  # (W,)

    # shape manipulations so that the first index is lz (to keep
    # addition order identical to the original loop)
    nz_mat = nz[None, :] + lz_offsets[:, None]  # (W, N)
    nzt = np.round(nz_mat).astype(int)  # (W, N)
    zpos_mat = S * ((nz[None, :] - nzt) + W / 2)  # (W, N)
    kw_mat = F_kbZ[np.round(zpos_mat).astype(int)]  # (W, N)

    nzt_clipped = np.clip(nzt, 0, n_os - 1)  # (W, N)
    np.add.at(
        out,
        nzt_clipped.ravel(order="C"),  # lz-major order
        (density[None, :] * kw_mat).ravel(order="C"),
    )

    out[0] = out[-1] = 0.0  # edge artefacts

    # ------------------------------------------------------------------
    # myifft  →  de-apodise  →  abs(myfft3)  (unchanged)
    # ------------------------------------------------------------------
    u = n
    f = np.fft.ifftshift(np.fft.ifft(np.fft.ifftshift(out))) * np.sqrt(u)
    f = f[int(np.ceil((f.size - u) / 2)) : int(np.ceil((f.size + u) / 2))]
    f /= kbZ[u // 2 : 3 * u // 2]

    F = np.fft.fftshift(np.fft.fftn(np.fft.fftshift(f))) / np.sqrt(f.size)
    return np.abs(F)


# =====================================================================
# helpers for get_z_profile() END
# =====================================================================


def get_z_profile(
    skel: Skeleton,
    extent: list[float | int] | None = None,
    bin_size: float = 1,  # µm
    hdr_mass: float = 0.95,
    *,
    measure: str = "length",  # ["length", "volume"]
    radius_metric: str | None = None,
    voxel_size: float | None = None,  # only used for volume (union)
    include_soma: bool = False,
) -> dict:
    """
    Compute a 1‑D depth profile.

    measure:
        "length" – cable length per bin (histogram + KB‑smoothed).
        "volume" – **union‑correct** morphology volume per bin (voxel union).

    Notes
    -----
    * volume uses a single voxelization (dx._voxelize_union) and
      **do not double count** at branch junctions or soma contacts.
    * 'include_soma' defaults to False to match the 'length' convention
      (edges only). Set True if you want soma membrane/volume included.
    * For stable plots, keep bin_size ≥ voxel_size (if you set voxel_size).
    """

    if measure == "length":
        density, nodes = segment_lengths(skel)
    elif measure == "volume":
        density, nodes = z_slince_volumes(
            skel,
            voxel_size=voxel_size,
            include_soma=include_soma,
            radius_metric=radius_metric,
        )
    else:
        raise ValueError("measure must be one of {'length','volume'}")

    z_vals = nodes[:, 2]

    # window
    if extent is None:
        z_min, z_max = np.floor(z_vals.min()), np.ceil(z_vals.max())
    else:
        z_min, z_max = extent

    # histogram bins
    n_bins = max(1, int(np.ceil((z_max - z_min) / bin_size)))
    edges = z_min + np.arange(n_bins + 1) * bin_size
    edges[-1] = z_max

    # histogram (mass‑preserving)
    z_hist, _ = np.histogram(z_vals, bins=edges, weights=density)
    tot = density.sum()
    if z_hist.sum() > 0:
        z_hist *= tot / z_hist.sum()

    # Kaiser–Bessel smoothing (same as length)
    centre = (z_min + z_max) / 2.0
    halfspan = (z_max - z_min) / 2.0
    z_samples = (z_vals - centre) / max(halfspan, np.finfo(float).eps)
    z_dist = gridder1d(z_samples / 2.0, density, n_bins)
    if z_dist.sum() > 0:
        z_dist *= tot / z_dist.sum()

    x_um = 0.5 * (edges[1:] + edges[:-1])
    intervals = hdr(x_um, z_dist, mass=hdr_mass)
    unit = skel.meta.get("unit", "µm")

    return {
        "x": x_um,
        "distribution": z_dist,
        "histogram": z_hist,
        "extent": [z_min, z_max],
        "n_bins": n_bins,
        "bin_size": bin_size,
        "hdr": intervals,
        "hdr_mass": hdr_mass,
        "measure": measure,
        "y_units": unit if measure == "length" else f"{unit}³",
        "include_soma": include_soma,
        "voxel_size": voxel_size,
        "radius_metric": radius_metric,
    }


def _edges_from_bin_size(lo: float, hi: float, bin_size: float) -> np.ndarray:
    """Generate edges ≥ bin_size wide, last bin clipped to *hi*."""
    n = int(np.ceil((hi - lo) / bin_size))
    edges = lo + np.arange(n + 1) * bin_size
    edges[-1] = hi  # ensure inclusion
    eps = np.finfo(float).eps
    edges[-1] += eps * max(1.0, abs(edges[-1]))  # numeric cushion
    return edges


def get_xy_profile(
    skel: Skeleton,
    extents: list[float | int] | None = None,
    bin_size: float | int = 2.0,
    smooth: float | int = 1.0,
    *,
    measure: str = "length",  # {"length","volume"}
    radius_metric: str | None = None,
    voxel_size: float | None = None,
    include_soma: bool = False,
) -> dict:
    """
    Planar (x-y) dendritic-length density on a **square** grid.

    Parameters
    ----------
    extents
        Fixed window ``[xmin, xmax, ymin, ymax]`` (µm).  *None* → tight box.
    bin_size
        Side length of each square bin (µm).  `edges = lo + k·bin_size`.
        The window is *expanded* to the next multiple so that every bin is
        exactly `bin_size` wide.  (This guarantees comparability.)
    smooth
        σ of the Gaussian kernel (bins) applied to the histogram.
    measure:
        "length" – dendritic cable length per bin (histogram → Gaussian smooth).
        "volume" – **union-correct** morphology volume per bin (voxel union),
                   integrated along z, then binned in x-y.

    Returns
    -------
    dict  with keys ``x`` / ``y`` (centres), ``distribution`` (smoothed),
           ``histogram`` (raw), ``extents``, ``bin_size``, ``nbins`` …
    """

    if measure == "length":
        density, mid = segment_lengths(skel)
        units = skel.meta.get("unit", "µm")
    elif measure == "volume":
        density, mid = xy_column_volume(
            skel,
            voxel_size=voxel_size,
            include_soma=include_soma,
            radius_metric=radius_metric,
        )
        units = f"{skel.meta.get('unit', 'µm')}³"
    else:
        raise ValueError("measure must be one of {'length','volume'}")

    # 0) bounding box ----------------------------------------------------------
    if extents is None:
        x_all = np.concatenate((mid[:, 0], skel.nodes[:, 0]))
        y_all = np.concatenate((mid[:, 1], skel.nodes[:, 1]))
        xmin, xmax = x_all.min(), x_all.max()
        ymin, ymax = y_all.min(), y_all.max()

        xmin = np.floor(xmin / bin_size) * bin_size
        xmax = np.ceil(xmax / bin_size) * bin_size
        ymin = np.floor(ymin / bin_size) * bin_size
        ymax = np.ceil(ymax / bin_size) * bin_size

    else:
        xmin, xmax, ymin, ymax = extents

    # 1) build edges so that *every* cell is square (bin_size × bin_size) ------
    x_edges = _edges_from_bin_size(xmin, xmax, bin_size)
    y_edges = _edges_from_bin_size(ymin, ymax, bin_size)

    xy_hist, _, _ = np.histogram2d(
        mid[:, 0], mid[:, 1], bins=[x_edges, y_edges], weights=density
    )

    xy_dist = gaussian_filter(xy_hist, sigma=smooth, mode="nearest")
    xy_dist *= density.sum() / xy_dist.sum()

    x = 0.5 * (x_edges[:-1] + x_edges[1:])
    y = 0.5 * (y_edges[:-1] + y_edges[1:])

    return {
        "x": x,
        "y": y,
        "distribution": xy_dist,
        "histogram": xy_hist,
        "extents": [x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]],
        "n_bins": (len(x), len(y)),  # may differ if dx ≠ dy
        "bin_size": bin_size,
        "smooth": smooth,
        "measure": measure,
        "units": units,
        "include_soma": include_soma,
        "voxel_size": voxel_size,
        "radius_metric": radius_metric,
    }


def hdr(z_centres, z_density, mass=0.95):
    """
    High Density Region (HDR) estimator.

    Parameters
    ----------
    z_centres : (N,) bin centres (µm)
    z_density : (N,) density per bin (any units)
    mass      : float, 0 < mass ≤ 1 (e.g. 0.95)

    Example
    -------
    >>> hdr_multimodal(z, d, 0.95)
    [array([ -2.1,  1.7]),   # ON sheet
     array([ 10.3, 13.9])]   # OFF sheet
    """
    p = z_density / z_density.sum()  # normalise → probability
    order = np.argsort(p)[::-1]  # bins from high to low density

    selected = []
    cum = 0.0
    for idx in order:
        selected.append(idx)
        cum += p[idx]
        if cum >= mass:
            break

    sel = np.sort(selected)  # ascending bin indices
    # split where gaps > 1 bin
    gaps = np.where(np.diff(sel) > 1)[0]
    groups = [g for g in np.split(sel, gaps + 1)]

    intervals = [
        np.array([z_centres[g[0]], z_centres[g[-1]]]) for g in groups if len(g) > 0
    ]
    return intervals


class Warper:
    """High‑level interface around *pywarper* for IPL flattening."""

    def __init__(
        self,
        surfaces: dict[str, dict | tuple] | None = None,
        swc_path: str | None = None,
        *,
        off_sac_points=None,
        on_sac_points=None,
        voxel_resolution: list[float] = [1.0, 1.0, 1.0],
        verbose: bool = False,
    ) -> None:
        self.voxel_resolution = voxel_resolution
        self.verbose = verbose
        self.swc_path = swc_path

        # ---- resolve legacy keyword arguments --------------------------------
        if surfaces is None and (off_sac_points is not None or on_sac_points is not None):
            if off_sac_points is None or on_sac_points is None:
                raise ValueError(
                    "Both off_sac_points and on_sac_points must be provided together."
                )
            surfaces = {"on_sac": on_sac_points, "off_sac": off_sac_points}
        elif surfaces is not None and (off_sac_points is not None or on_sac_points is not None):
            raise ValueError(
                "Cannot provide both surfaces dict and off_sac_points/on_sac_points."
            )

        self.surface_points: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        if surfaces is not None:
            for tag, data in surfaces.items():
                self.surface_points[tag] = self._as_xyz(data)

        # Legacy convenience attributes
        if "off_sac" in self.surface_points:
            self.off_sac_points = self.surface_points["off_sac"]
        if "on_sac" in self.surface_points:
            self.on_sac_points = self.surface_points["on_sac"]

        if swc_path is not None:
            self.load_swc(swc_path)
        else:
            self.swc_path = None

    # ---------------------------- IO -------------------------------------
    def load_swc(self, swc_path: str | None = None) -> "Warper":
        """Load the skeleton from *swc_path*."""

        if self.verbose:
            print(f"[pywarper] Loading skeleton → {self.swc_path}")

        if swc_path is None:
            swc_path = self.swc_path

        if swc_path is not None:
            self.skeleton = sk.io.load_swc(swc_path)
        else:
            raise ValueError("SWC path must be provided to load the skeleton.")

        return self

    @staticmethod
    def _as_xyz(data) -> tuple[np.ndarray, np.ndarray, np.ndarray]:  # for load_sac()
        """Accept *dict* or tuple and return *(x, y, z)* numpy arrays."""
        if isinstance(data, dict):
            return np.asarray(data["x"]), np.asarray(data["y"]), np.asarray(data["z"])
        if isinstance(data, (tuple, list)) and len(data) == 3:
            return map(np.asarray, data)  # type: ignore[arg-type]
        raise TypeError(
            "SAC data must be a mapping with keys x/y/z or a 3‑tuple of arrays."
        )

    def load_surfaces(self, surfaces: dict) -> "Warper":
        """Load surface point clouds from a dict of tag -> data."""
        if self.verbose:
            print("[pywarper] Loading surface meshes …")
        self.surface_points = {}
        for tag, data in surfaces.items():
            self.surface_points[tag] = self._as_xyz(data)
        # Keep legacy attributes up to date
        if "off_sac" in self.surface_points:
            self.off_sac_points = self.surface_points["off_sac"]
        if "on_sac" in self.surface_points:
            self.on_sac_points = self.surface_points["on_sac"]
        return self

    def load_sac(self, off_sac_points, on_sac_points) -> "Warper":
        """Load the SAC meshes from *off_sac_points* and *on_sac_points*."""
        return self.load_surfaces({"on_sac": on_sac_points, "off_sac": off_sac_points})

    def load_warped_skeleton(
        self,
        filepath: str,
        med_z_on: float | None = None,
        med_z_off: float | None = None,
    ) -> None:
        """Load a warped skeleton from *swc_path*."""
        path = Path(filepath)

        if path.suffix.lower() == ".swc":
            self.warped_skeleton = sk.io.load_swc(path)

            if (med_z_on is not None) and (med_z_off is not None):
                self.warped_skeleton.extra["med_z_on"] = float(med_z_on)
                self.warped_skeleton.extra["med_z_off"] = float(med_z_off)
            else:
                self.warped_skeleton.extra["med_z_on"] = None
                self.warped_skeleton.extra["med_z_off"] = None
        elif path.suffix.lower() == ".npz":
            self.warped_skeleton = sk.io.load_npz(path)

        if self.verbose:
            print(f"[pywarper] Loaded warped skeleton → {path}")

    # ---------------------------- Core -----------------------------------

    def fit_surfaces(
        self,
        xmax: int | float | None = None,
        ymax: int | float | None = None,
        stride: int = 3,
        smoothness: int = 15,
        backward_compatible: bool = False,
    ) -> "Warper":
        """Fit all loaded surface point-clouds with *pygridfit*."""
        if self.verbose:
            print("[pywarper] Fitting surfaces …")

        if not self.surface_points:
            raise RuntimeError("No surface point clouds loaded.")

        if backward_compatible is False and (xmax is None or ymax is None):
            all_x = np.concatenate([pts[0] for pts in self.surface_points.values()])
            all_y = np.concatenate([pts[1] for pts in self.surface_points.values()])
            xmax = float(all_x.max())
            ymax = float(all_y.max())

        self.fitted_surfaces: dict[str, np.ndarray] = {}
        for tag, (x, y, z) in self.surface_points.items():
            _t0 = time.time()
            surface, *_ = fit_surface(
                x=x, y=y, z=z,
                stride=stride,
                smoothness=smoothness,
                xmax=xmax, ymax=ymax,
                backward_compatible=backward_compatible,
            )
            self.fitted_surfaces[tag] = surface
            if self.verbose:
                print(f"↳ fitting '{tag}' surface\n    done in {time.time() - _t0:.2f} seconds.")

        # Legacy attributes
        if "off_sac" in self.fitted_surfaces:
            self.off_sac_surface = self.fitted_surfaces["off_sac"]
        if "on_sac" in self.fitted_surfaces:
            self.on_sac_surface = self.fitted_surfaces["on_sac"]

        return self

    def build_mapping(
        self,
        bounds: np.ndarray | tuple | str | None = "local",
        conformal_jump: int = 2,
        n_anchors: int = 16,
        backward_compatible: bool = False,
    ) -> "Warper":
        """Create the quasi-conformal surface mapping."""
        if not hasattr(self, "fitted_surfaces") or not self.fitted_surfaces:
            raise RuntimeError("Surfaces not fitted. Call fit_surfaces() first.")

        if bounds is None or bounds == "local":
            xmin, xmax = (
                self.skeleton.nodes[:, 0].min(),
                self.skeleton.nodes[:, 0].max(),
            )
            ymin, ymax = (
                self.skeleton.nodes[:, 1].min(),
                self.skeleton.nodes[:, 1].max(),
            )
            bounds = np.array([xmin, xmax, ymin, ymax], dtype=float)
        elif bounds == "global":
            nx = max(s.shape[0] for s in self.fitted_surfaces.values())
            ny = max(s.shape[1] for s in self.fitted_surfaces.values())
            bounds = np.array([0, nx, 0, ny], dtype=float)
        else:
            bounds = np.asarray(bounds, dtype=float)
            if bounds.shape != (4,):
                raise ValueError(
                    "Bounds must be a 4-element array or tuple (x_min, x_max, y_min, y_max)."
                )

        if self.verbose:
            print("[pywarper] Building mapping …")
        self.mapping: dict = build_mapping(
            self.fitted_surfaces,
            bounds,
            conformal_jump=conformal_jump,
            n_anchors=n_anchors,
            backward_compatible=backward_compatible,
            verbose=self.verbose,
        )
        return self

    def warp_skeleton(
        self,
        on_sac_pos: float = 0.0,
        off_sac_pos: float = 12.0,
        z_profile_extent: list[float | int] | None = None,
        z_profile_bin_size: float | int = 1,  # um
        z_profile_hdr_mass: float | int = 0.95,
        z_profile_include_soma: bool = False,
        z_profile_voxel_size: float | None = None,
        xy_profile_extents: list[float | int] | None = None,
        xy_profile_bin_size: float | int = 20,  # um
        xy_profile_smooth: float | int = 1.0,
        xy_profile_include_soma: bool = False,
        xy_profile_voxel_size: float | None = None,
        radius_metric: str | None = None,
        skeleton_nodes_scale: float = 1.0,
        voxel_resolution: list[float | int] | None = None,
        conformal_jump: int | None = None,
        backward_compatible: bool = False,
    ) -> "Warper":
        """Apply the mapping to the skeleton."""
        if self.mapping is None:
            raise RuntimeError("Mapping missing. Call build_mapping() first.")

        if voxel_resolution is None:
            voxel_resolution = self.voxel_resolution

        self.warped_skeleton = warp_skeleton(
            self.skeleton,
            self.mapping,
            on_sac_pos=on_sac_pos,
            off_sac_pos=off_sac_pos,
            voxel_resolution=voxel_resolution,
            conformal_jump=conformal_jump,
            z_profile_extent=z_profile_extent,
            z_profile_bin_size=z_profile_bin_size,
            z_profile_hdr_mass=z_profile_hdr_mass,
            z_profile_include_soma=z_profile_include_soma,
            z_profile_voxel_size=z_profile_voxel_size,
            xy_profile_extents=xy_profile_extents,
            xy_profile_bin_size=xy_profile_bin_size,
            xy_profile_smooth=xy_profile_smooth,
            xy_profile_include_soma=xy_profile_include_soma,
            xy_profile_voxel_size=xy_profile_voxel_size,
            radius_metric=radius_metric,
            backward_compatible=backward_compatible,
            skeleton_nodes_scale=skeleton_nodes_scale,
            verbose=self.verbose,
        )
        return self

    def renormalize(
        self,
        on_sac_pos: float = 0.0,
        off_sac_pos: float = 12.0,
        z_profile_extent: list[float | int] | None = None,  # [z_min, z_max]
        z_profile_bin_size: float | int | None = None,
        z_profile_hdr_mass: float | int | None = None,
        z_profile_include_soma: bool | None = None,
        z_profile_voxel_size: float | None = None,
        xy_profile_extents: list[float | int]
        | None = None,  # [x_min, x_max, y_min, y_max]
        xy_profile_bin_size: float | int | None = None,
        xy_profile_smooth: float | int | None = None,
        xy_profile_include_soma: bool | None = None,
        xy_profile_voxel_size: float | None = None,
        radius_metric: str | None = None,
    ) -> Skeleton:
        """Renormalize the warped skeleton to the desired ON/OFF SAC positions."""
        if self.warped_skeleton is None:
            raise RuntimeError("Warped skeleton missing. Call warp_skeleton() first.")
        else:
            median_depths = self.warped_skeleton.extra.get(
                "median_depths",
                {"on_sac": self.warped_skeleton.extra["med_z_on"],
                 "off_sac": self.warped_skeleton.extra["med_z_off"]},
            )
            renormed_nodes = normalize_nodes(
                self.warped_skeleton.extra["prenormed_nodes"],
                median_depths=median_depths,
                anchor_pos=(on_sac_pos, off_sac_pos),
            )

        soma_renormed = deepcopy(self.warped_skeleton.soma)
        soma_renormed.center = renormed_nodes[0] * self.voxel_resolution

        skel_renormed = Skeleton(
            soma=soma_renormed,
            nodes=renormed_nodes,
            edges=self.warped_skeleton.edges,
            radii=self.warped_skeleton.radii,
            ntype=self.warped_skeleton.ntype,
            meta=self.warped_skeleton.meta.copy(),  # copy metadata
        )

        z_profiles = {
            measure: get_z_profile(
                skel_renormed,
                extent=z_profile_extent
                if z_profile_extent is not None
                else self.warped_skeleton.extra["z_profiles"][measure]["extent"],
                bin_size=z_profile_bin_size
                if z_profile_bin_size is not None
                else self.warped_skeleton.extra["z_profiles"][measure]["bin_size"],
                hdr_mass=z_profile_hdr_mass
                if z_profile_hdr_mass is not None
                else self.warped_skeleton.extra["z_profiles"][measure]["hdr_mass"],
                measure=measure,
                include_soma=z_profile_include_soma
                if z_profile_include_soma is not None
                else self.warped_skeleton.extra["z_profiles"][measure]["include_soma"],
                voxel_size=z_profile_voxel_size
                if z_profile_voxel_size is not None
                else self.warped_skeleton.extra["z_profiles"][measure]["voxel_size"],
                radius_metric=radius_metric
                if radius_metric is not None
                else self.warped_skeleton.extra["z_profiles"][measure]["radius_metric"],
            )
            for measure in ["length", "volume"]
        }
        xy_profiles = {
            measure: get_xy_profile(
                skel_renormed,
                extents=xy_profile_extents
                if xy_profile_extents is not None
                else self.warped_skeleton.extra["xy_profiles"][measure]["extents"],
                bin_size=xy_profile_bin_size
                if xy_profile_bin_size is not None
                else self.warped_skeleton.extra["xy_profiles"][measure]["bin_size"],
                smooth=xy_profile_smooth
                if xy_profile_smooth is not None
                else self.warped_skeleton.extra["xy_profiles"][measure]["smooth"],
                include_soma=xy_profile_include_soma
                if xy_profile_include_soma is not None
                else self.warped_skeleton.extra["xy_profiles"][measure]["include_soma"],
                voxel_size=xy_profile_voxel_size
                if xy_profile_voxel_size is not None
                else self.warped_skeleton.extra["xy_profiles"][measure]["voxel_size"],
                radius_metric=radius_metric
                if radius_metric is not None
                else self.warped_skeleton.extra["xy_profiles"][measure][
                    "radius_metric"
                ],
            )
            for measure in ["length", "volume"]
        }

        skel_renormed.extra = {
            "prenormed_nodes": self.warped_skeleton.extra["prenormed_nodes"],
            "median_depths": median_depths,
            "med_z_on": float(median_depths.get("on_sac", 0.0)),
            "med_z_off": float(median_depths.get("off_sac", 0.0)),
            "z_profiles": z_profiles,
            "xy_profiles": xy_profiles,
        }
        skel_renormed.meta.update(
            {
                "pywarper_version": _PYWARPER_VERSION,
                "renormed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
        )

        return skel_renormed
