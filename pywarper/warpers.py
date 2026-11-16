"""
pywarper.warpers
==============
Spatial warping and profiling utilities for **neuronal skeleton reconstructions**.

This module takes a neuronal tree (nodes+edges) and the previously-computed
ON/OFF Starburst Amacrine Cell (SAC) surface mapping in order to

1. **Warp the skeleton into the flattened SAC coordinate frame** (`warp_nodes`).
   Each node is locally re-registered with a polynomial least-squares fit
   (`local_ls_registration`) that references both SAC layers so that depth is
   preserved relative to the curved retina.
2. **Compute depth (z) profiles** (`get_z_profile`).  Edge lengths are first binned
   directly (histogram) and then re-estimated with a Kaiser–Bessel gridding
   kernel to obtain a smooth 1-D density across the inner plexiform layer.
3. **Compute planar (xy) density maps** (`get_xy_profile`).  Dendritic length is
   accumulated on a user-defined 2-D grid and optionally Gaussian-smoothed for
   visualisation or group statistics.

Key algorithms
--------------
* **Polynomial local registration** – For every node we fit a 2-D polynomial
  basis (up to a configurable `max_order`) to the positions of neighbouring
  SAC-band sample points, solving three separate least-squares systems in one
  go with `numpy.linalg.lstsq`.  A single **KDTree** (SciPy) accelerates the
  neighbourhood queries.
* **Kaiser–Bessel gridding** – The 1-D `gridder1d` function emulates the
  non-uniform FFT gridding scheme used by older MATLAB code, yielding the exact
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
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import gaussian_filter
from scipy.spatial import KDTree
from scipy.special import i0
from skeliner._core import _bfs_parents
from skeliner.dataclass import Skeleton
from skeliner.dx import _choose_voxel_size, _ellipsoid_aabb

from .surface import build_mapping, fit_sac_surface
from .utils import build_surface_correspondences, resolve_conformal_jump

_PYWARPER_VERSION = _metadata.version("pywarper")


def poly_basis_2d(x: np.ndarray, y: np.ndarray, max_order: int) -> np.ndarray:
    """
    Return the full 2-D polynomial basis up to total order *max_order*
    for coordinates (x, y).  Shape:  (len(x), n_terms)

    Order layout matches the original code:
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
    top_input_pos: np.ndarray,
    bot_input_pos: np.ndarray,
    top_output_pos: np.ndarray,
    bot_output_pos: np.ndarray,
    *,
    window: float,
    max_order: int,
) -> dict[str, np.ndarray | KDTree | float | int]:
    in_all = np.vstack((top_input_pos, bot_input_pos))
    out_all = np.vstack((top_output_pos, bot_output_pos))
    is_top = np.concatenate(
        (
            np.ones(len(top_input_pos), dtype=bool),
            np.zeros(len(bot_input_pos), dtype=bool),
        )
    )
    all_xy = in_all[:, :2]
    query_r = window * np.sqrt(2.0)
    tree = KDTree(all_xy)
    return {
        "in_all": in_all,
        "out_all": out_all,
        "is_top": is_top,
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
    is_top = state["is_top"]
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

        idx_top = idx[is_top[idx]]
        idx_bot = idx[~is_top[idx]]

        in_top, out_top = in_all[idx_top], out_all[idx_top]
        in_bot, out_bot = in_all[idx_bot], out_all[idx_bot]

        this_in = np.vstack((in_top, in_bot))
        this_out = np.vstack((out_top, out_bot))

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
    top_input_pos: np.ndarray,
    bot_input_pos: np.ndarray,
    top_output_pos: np.ndarray,
    bot_output_pos: np.ndarray,
    window: float = 5.0,
    max_order: int = 2,
    warn: bool = True,
) -> np.ndarray:
    """
    Same algorithm as before, but a **single KDTree** stores both
    surfaces.  The neighbour search is therefore performed once.
    """
    state = _build_local_ls_state(
        top_input_pos,
        bot_input_pos,
        top_output_pos,
        bot_output_pos,
        window=window,
        max_order=max_order,
    )
    return _apply_local_ls_state(nodes, state, warn=warn)


def warp_nodes(
    nodes: np.ndarray,
    surface_mapping: dict,
    conformal_jump: int | None = None,
    backward_compatible: bool = False,
) -> tuple[np.ndarray, float, float]:
    """Warp `nodes` (surface units) into the flattened frame, before normalization."""
    resolved_jump = resolve_conformal_jump(surface_mapping, conformal_jump)
    on_input_pts, off_input_pts, on_output_pts, off_output_pts, med_z_on, med_z_off = (
        build_surface_correspondences(
            surface_mapping,
            conformal_jump=resolved_jump,
            backward_compatible=backward_compatible,
        )
    )

    # Apply local least-squares registration to each node
    warped = local_ls_registration(
        nodes, on_input_pts, off_input_pts, on_output_pts, off_output_pts
    )
    if "extra_surfaces" in surface_mapping:
        rel = relative_depth(
            nodes, surface_mapping, backward_compatible=backward_compatible
        )
        warped[:, 2] = med_z_on + rel * (med_z_off - med_z_on)

    return warped, med_z_on, med_z_off


def normalize_nodes(
    nodes: np.ndarray,
    med_z_on: float,
    med_z_off: float,
    on_sac_pos: float = 0.0,
    off_sac_pos: float = 12.0,
) -> np.ndarray:
    """
    Normalize the z-coordinates of nodes based on the median z-values
    of the ON and OFF SAC surfaces.
    This function rescales the z-coordinates of the nodes to a normalized
    space where the ON SAC surface is at `on_sac_pos` and the OFF SAC
    surface is at `off_sac_pos`. The z-coordinates are adjusted based on
    the provided median z-values of the ON and OFF SAC surfaces.

    Parameters
    ----------
    nodes : np.ndarray
        (N, 3) array of [x, y, z] coordinates for the nodes to be normalized.
    med_z_on : float
        Median z-value of the ON SAC surface.
    med_z_off : float
        Median z-value of the OFF SAC surface.
    on_sac_pos : float, default=0.0
        Desired position of the ON SAC surface in the normalized space (µm).
    off_sac_pos : float, default=12.0
        Desired position of the OFF SAC surface in the normalized space (µm).
    z_res : float, default=1.0
        Spatial resolution along z (µm / voxel) after warping.
    Returns
    -------
    np.ndarray
        (N, 3) array of [x, y, z] coordinates with normalized z-coordinates.
    """
    normalized_nodes = nodes.copy().astype(float)

    # Compute the relative depth of each node
    rel_depth = (nodes[:, 2] - med_z_on) / (med_z_off - med_z_on)  # 0->ON, 1->OFF

    # Rescale the z-coordinates to the normalized space
    z_phys = on_sac_pos + rel_depth * (off_sac_pos - on_sac_pos)  # µm in global frame
    normalized_nodes[:, 2] = z_phys  # update the z-coordinate to the flattened space

    return normalized_nodes


def relative_depth(
    points: np.ndarray,
    surface_mapping: dict,
    *,
    backward_compatible: bool = False,
) -> np.ndarray:
    """
    Relative depth of points (0 at the ON band, 1 at the OFF band), per column.

    Without extra surfaces in the mapping this is the fraction of the way from
    the ON to the OFF band in the point's column, extended linearly outside
    them. With extra surfaces (see `build_mapping`) the depth is interpolated
    piecewise between the two layers that bracket the point and extended
    linearly with the outermost interval; columns where the layers are not in
    the order of their depths fall back to the two bands.

    Parameters
    ----------
    points : np.ndarray
        (N, 3) positions before warping, in the surfaces' units.
    surface_mapping : dict
        Output of `build_mapping`.
    backward_compatible : bool, default=False
        The surfaces' grid starts at 1 instead of 0.

    Returns
    -------
    np.ndarray
        (N,) relative depth.
    """
    points = np.asarray(points, dtype=float)
    on = np.asarray(surface_mapping["on_sac_surface"], dtype=float)
    off = np.asarray(surface_mapping["off_sac_surface"], dtype=float)
    first = 1.0 if backward_compatible else 0.0
    gx = np.arange(on.shape[0]) + first
    gy = np.arange(on.shape[1]) + first
    xy = np.column_stack(
        [np.clip(points[:, 0], gx[0], gx[-1]), np.clip(points[:, 1], gy[0], gy[-1])]
    )
    z_on = RegularGridInterpolator((gx, gy), on)(xy)
    z_off = RegularGridInterpolator((gx, gy), off)(xy)
    with np.errstate(divide="ignore", invalid="ignore"):
        two_band = (points[:, 2] - z_on) / (z_off - z_on)
    if "extra_surfaces" not in surface_mapping:
        return two_band

    depths = np.concatenate(
        [[0.0, 1.0], np.asarray(surface_mapping["extra_surface_depths"], float)]
    )
    layers = [z_on, z_off] + [
        RegularGridInterpolator((gx, gy), np.asarray(h, dtype=float))(xy)
        for h in surface_mapping["extra_surfaces"]
    ]
    order = np.argsort(depths)
    depths = depths[order]
    z = np.column_stack(layers)[:, order]
    # orient each column so that z grows with depth, as it does from ON to OFF
    sign = np.sign(z_off - z_on)
    z = z * sign[:, None]
    p = points[:, 2] * sign
    ordered = np.all(np.diff(z, axis=1) > 0, axis=1)

    lo = np.clip((p[:, None] > z).sum(axis=1) - 1, 0, len(depths) - 2)
    rows = np.arange(len(p))
    z_lo, z_hi = z[rows, lo], z[rows, lo + 1]
    with np.errstate(divide="ignore", invalid="ignore"):
        piecewise = depths[lo] + (p - z_lo) / (z_hi - z_lo) * (
            depths[lo + 1] - depths[lo]
        )
    return np.where(ordered, piecewise, two_band)


def node_coverage(
    nodes: np.ndarray,
    surface_mapping: dict,
    *,
    backward_compatible: bool = False,
) -> dict | None:
    """
    Where the imaged volume and the band observations end, per node.

    Parameters
    ----------
    nodes : np.ndarray
        (N, 3) node positions before warping, in the surfaces' units.
    surface_mapping : dict
        Output of `build_mapping`. Besides the surfaces, only its ``volume_z``,
        ``on_support`` and ``off_support`` entries are read.
    backward_compatible : bool, default=False
        The surfaces' grid starts at 1 instead of 0.

    Returns
    -------
    dict or None
        None when the mapping has none of those entries. Otherwise, each only
        when the mapping provides what it needs:

        ``depth_rel`` : (N, 2)
            The depth range the volume covers in each node's column, lowest
            first, in relative depth (0 at the ON band, 1 at the OFF band).
            ±inf on a side whose face is unknown.
        ``on_support``, ``off_support`` : (N,) bool
            Whether that band was observed in the node's column.
    """
    has_faces = "volume_z" in surface_mapping
    bands = [b for b in ("on", "off") if f"{b}_support" in surface_mapping]
    if not has_faces and not bands:
        return None

    nodes = np.asarray(nodes, dtype=float)
    on = np.asarray(surface_mapping["on_sac_surface"], dtype=float)
    first = 1.0 if backward_compatible else 0.0
    gx = np.arange(on.shape[0]) + first
    gy = np.arange(on.shape[1]) + first
    xy = np.column_stack(
        [np.clip(nodes[:, 0], gx[0], gx[-1]), np.clip(nodes[:, 1], gy[0], gy[-1])]
    )

    coverage: dict = {}
    if has_faces:
        off = np.asarray(surface_mapping["off_sac_surface"], dtype=float)
        z_on = RegularGridInterpolator((gx, gy), on)(xy)
        z_off = RegularGridInterpolator((gx, gy), off)(xy)
        # z grows with depth (+1) or against it (-1) in each column
        sign = np.sign(z_off - z_on)
        faces = np.asarray(surface_mapping["volume_z"], dtype=float)
        rel = np.empty((len(nodes), 2))
        for k, face in enumerate(faces):
            if np.isnan(face):
                # an unknown face does not bound the column on its side
                rel[:, k] = (-np.inf if k == 0 else np.inf) * sign
            else:
                at_face = np.column_stack([xy, np.full(len(nodes), face)])
                rel[:, k] = relative_depth(
                    at_face, surface_mapping, backward_compatible=backward_compatible
                )
        coverage["depth_rel"] = np.sort(rel, axis=1)

    ix = np.clip(np.rint(xy[:, 0] - first).astype(int), 0, on.shape[0] - 1)
    iy = np.clip(np.rint(xy[:, 1] - first).astype(int), 0, on.shape[1] - 1)
    for band in bands:
        support = np.asarray(surface_mapping[f"{band}_support"], dtype=bool)
        coverage[f"{band}_support"] = support[ix, iy]
    return coverage


def _covered_depth(
    depth_rel: np.ndarray,
    on_sac_pos: float,
    off_sac_pos: float,
    scale: float = 1.0,
    z_resolution: float = 1.0,
) -> np.ndarray:
    """`node_coverage`'s relative depth range in the warped output's z units."""
    with np.errstate(invalid="ignore"):
        depth = on_sac_pos + np.asarray(depth_rel) * (off_sac_pos - on_sac_pos)
    return np.sort(depth / scale * z_resolution, axis=1)


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
    4. When the mapping carries the volume's z-range or band support masks,
       ``extra["coverage"]`` holds `node_coverage`'s record for every node, plus
       ``depth``: the depth range the volume covers, in the output's z units.
       `get_coverage` summarises it per skeleton.
    """

    nodes = (
        skel.nodes.astype(float) * skeleton_nodes_scale
    )  # scale to the surface unit, which is often µm

    if verbose:
        print("[pywarper] Warping skeleton...")
        start_time = time.time()
    warped_nodes, med_z_on, med_z_off = warp_nodes(
        nodes,
        surface_mapping,
        conformal_jump=conformal_jump,
        backward_compatible=backward_compatible,
    )

    normalized_nodes = normalize_nodes(
        warped_nodes,
        med_z_on=med_z_on,
        med_z_off=med_z_off,
        on_sac_pos=on_sac_pos,
        off_sac_pos=off_sac_pos,
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
        "med_z_on": float(med_z_on),
        "med_z_off": float(med_z_off),
        # what `renormalize` needs to put renormalized nodes in these same units
        "voxel_resolution": np.broadcast_to(
            np.asarray(voxel_resolution, dtype=float), (3,)
        ).copy(),
        "skeleton_nodes_scale": float(skeleton_nodes_scale),
        "z_profiles": z_profiles,
        "xy_profiles": xy_profiles,
    }
    coverage = node_coverage(
        nodes, surface_mapping, backward_compatible=backward_compatible
    )
    if coverage is not None:
        if "depth_rel" in coverage:
            coverage["depth"] = _covered_depth(
                coverage["depth_rel"],
                on_sac_pos,
                off_sac_pos,
                skeleton_nodes_scale,
                skel_norm.extra["voxel_resolution"][2],
            )
        skel_norm.extra["coverage"] = coverage
    skel_norm.meta.update(
        {
            "pywarper_version": _PYWARPER_VERSION,
            "warped_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
    )

    return skel_norm


def warp_mesh(
    mesh: trimesh.Trimesh,  # mostly nm
    surface_mapping: dict,  # mostly µm
    conformal_jump: int | None = None,
    on_sac_pos: float = 0.0,  # µm
    off_sac_pos: float = 12.0,  # µm
    mesh_vertices_scale: float = 1.0,  # scale factor for mesh vertices, e.g., 1e-3 for nm to µm
    backward_compatible: bool = False,
    verbose: bool = False,
) -> trimesh.Trimesh:
    """
    Applies a local surface flattening (warp) to a 3D mesh using the results
    of previously computed surface mappings.
    """

    vertices = (
        mesh.vertices.astype(float) * mesh_vertices_scale
    )  # scale to the surface unit, which is often µm

    if verbose:
        print("[pywarper] Warping mesh...")
        start_time = time.time()
    warped_vertices, med_z_on, med_z_off = warp_nodes(
        vertices,
        surface_mapping,
        conformal_jump=conformal_jump,
        backward_compatible=backward_compatible,
    )

    normalized_vertices = normalize_nodes(
        warped_vertices,
        med_z_on=med_z_on,
        med_z_off=med_z_off,
        on_sac_pos=on_sac_pos,
        off_sac_pos=off_sac_pos,
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
    warped_mesh.metadata["med_z_on"] = float(med_z_on)
    warped_mesh.metadata["med_z_off"] = float(med_z_off)
    warped_mesh.metadata["conformal_jump"] = conformal_jump
    warped_mesh.metadata["surface_mapping"] = surface_mapping
    warped_mesh.metadata["on_sac_pos"] = on_sac_pos
    warped_mesh.metadata["off_sac_pos"] = off_sac_pos
    coverage = node_coverage(
        vertices, surface_mapping, backward_compatible=backward_compatible
    )
    if coverage is not None:
        if "depth_rel" in coverage:
            coverage["depth"] = _covered_depth(
                coverage["depth_rel"], on_sac_pos, off_sac_pos, mesh_vertices_scale
            )
        warped_mesh.metadata["coverage"] = coverage
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


def _voxelize_union(
    skel,
    radii: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    *,
    voxel_size: float | None,
    include_soma: bool,
):
    """
    Boolean occupancy grid for the union of edge frusta and (optionally) soma,
    inside [lo, hi].  Returns (occ, h, (nx,ny,nz), lo, hi).
    """
    h, (nx, ny, nz) = _choose_voxel_size(radii, lo, hi, user_voxel=voxel_size)
    xs = lo[0] + (np.arange(nx) + 0.5) * h
    ys = lo[1] + (np.arange(ny) + 0.5) * h
    zs = lo[2] + (np.arange(nz) + 0.5) * h

    occ = np.zeros((nx, ny, nz), dtype=bool)
    nodes = skel.nodes.astype(np.float64, copy=False)
    edges = skel.edges.astype(np.int64, copy=False)

    # ---------------- helpers ----------------
    def _range_x(x0, x1):
        i0 = int(max(0, np.floor((x0 - lo[0]) / h)))
        i1 = int(min(nx - 1, np.floor((x1 - lo[0]) / h)))
        return i0, i1

    def _range_y(y0, y1):
        j0 = int(max(0, np.floor((y0 - lo[1]) / h)))
        j1 = int(min(ny - 1, np.floor((y1 - lo[1]) / h)))
        return j0, j1

    def _range_z(z0, z1):
        k0 = int(max(0, np.floor((z0 - lo[2]) / h)))
        k1 = int(min(nz - 1, np.floor((z1 - lo[2]) / h)))
        return k0, k1

    # --------- precompute soma mask once (if there is a soma) ------------
    soma_slice = None
    soma_mask = None
    if getattr(skel, "soma", None) is not None:
        slo, shi = _ellipsoid_aabb(skel.soma)
        # clip to global bbox
        slo = np.maximum(slo, lo)
        shi = np.minimum(shi, hi)
        i0, i1 = _range_x(slo[0], shi[0])
        j0, j1 = _range_y(slo[1], shi[1])
        k0, k1 = _range_z(slo[2], shi[2])

        if (i1 >= i0) and (j1 >= j0) and (k1 >= k0):
            # broadcasted coordinate slabs (no big meshgrid; uses ogrid)
            xi = xs[i0 : i1 + 1][:, None, None]
            yj = ys[j0 : j1 + 1][None, :, None]
            zk = zs[k0 : k1 + 1][None, None, :]

            cx, cy, cz = skel.soma.center
            Rt = skel.soma.R.T  # 3x3
            ax, ay, az = skel.soma.axes
            ax2, ay2, az2 = ax * ax, ay * ay, az * az

            dx = xi - cx
            dy = yj - cy
            dz = zk - cz

            # rotate into soma body-frame: u = R^T (x - c)
            ux = Rt[0, 0] * dx + Rt[0, 1] * dy + Rt[0, 2] * dz
            uy = Rt[1, 0] * dx + Rt[1, 1] * dy + Rt[1, 2] * dz
            uz = Rt[2, 0] * dx + Rt[2, 1] * dy + Rt[2, 2] * dz

            soma_mask = (ux * ux) / ax2 + (uy * uy) / ay2 + (uz * uz) / az2 <= 1.0
            soma_slice = (slice(i0, i1 + 1), slice(j0, j1 + 1), slice(k0, k1 + 1))

    # If soma is to be included, OR it now.
    if include_soma and soma_mask is not None:
        occ[soma_slice] |= soma_mask

    # ---------------- rasterize every edge (broadcasted) -----------------
    for i, j in edges:
        a = nodes[i]
        b = nodes[j]
        r0 = float(radii[i])
        r1 = float(radii[j])

        rmax = max(r0, r1)
        if not np.isfinite(rmax) or rmax < 0.0:
            continue

        # edge AABB padded by rmax, clipped to [lo,hi]
        lo_e = np.maximum(np.minimum(a, b) - rmax, lo)
        hi_e = np.minimum(np.maximum(a, b) + rmax, hi)
        if np.any(lo_e > hi_e):
            continue

        ii0, ii1 = _range_x(lo_e[0], hi_e[0])
        jj0, jj1 = _range_y(lo_e[1], hi_e[1])
        kk0, kk1 = _range_z(lo_e[2], hi_e[2])
        if (ii1 < ii0) or (jj1 < jj0) or (kk1 < kk0):
            continue

        xi = xs[ii0 : ii1 + 1][:, None, None]
        yj = ys[jj0 : jj1 + 1][None, :, None]
        zk = zs[kk0 : kk1 + 1][None, None, :]

        v = b - a
        L2 = float(v @ v)
        if L2 <= 1e-24:
            # degenerate: paint a ball of radius rmax at 'a'
            dx = xi - a[0]
            dy = yj - a[1]
            dz = zk - a[2]
            d2 = dx * dx + dy * dy + dz * dz
            mask = d2 <= (rmax * rmax)
            occ[ii0 : ii1 + 1, jj0 : jj1 + 1, kk0 : kk1 + 1] |= mask
            continue

        vx, vy, vz = v
        # projection parameter s (broadcasted), then clamp to [0,1]
        dx = xi - a[0]
        dy = yj - a[1]
        dz = zk - a[2]
        s = (dx * vx + dy * vy + dz * vz) / L2
        # clip in-place to save a temporary
        np.clip(s, 0.0, 1.0, out=s)

        # distance from voxel center to closest point on segment
        rx = dx - s * vx
        ry = dy - s * vy
        rz = dz - s * vz
        d2 = rx * rx + ry * ry + rz * rz

        # linear radius along the frustum
        r = r0 + s * (r1 - r0)
        mask = d2 <= (r * r)

        occ[ii0 : ii1 + 1, jj0 : jj1 + 1, kk0 : kk1 + 1] |= mask

    # If soma is to be excluded, carve it out once (reuses precomputed mask).
    if (not include_soma) and (soma_mask is not None):
        occ[soma_slice] &= ~soma_mask

    return occ, float(h), (int(nx), int(ny), int(nz)), lo, hi


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
        Union volume per *non-empty* z-slice (in unit³).
    mid     : (K, 3) float
        Sample positions for each slice: (x_bar, y_bar, z_center), where x_bar,y_bar are
        slice area-weighted centroids and z_center is the slice center.
    """
    # choose radii column for voxelizer / bbox
    if radius_metric is None:
        radius_metric = skel.recommend_radius()[0]
    radii = np.asarray(skel.radii[radius_metric], dtype=np.float64).reshape(-1)

    # tight auto-bbox (like dx.volume/area)
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

    # keep only non-empty slices
    mask = vol_all > 0.0
    if not mask.any():
        return np.zeros(0, dtype=float), np.zeros((0, 3), dtype=float)
    vol = vol_all[mask]
    k = np.where(mask)[0]  # slice indices kept

    # z center for each kept slice
    zc = lo[2] + (k + 0.5) * h

    # area-weighted centroids per kept slice (optional but nice to have)
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
        (x_center, y_center, z_bar) for that column, where z_bar is the z-centroid
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
    Kaiser–Bessel gridding kernel in 1-D   (alpha=2, W=5)

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
    # myifft  ->  de-apodise  ->  abs(myfft3)  (unchanged)
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


def _split_path(
    path: list[int], xyz: np.ndarray, split_length: float
) -> list[list[int]]:
    """
    Divide an ordered node path into n roughly equal sub-paths.

    n = max(1, round(total_length / split_length)), so a 100 µm branch with
    split_length=30 becomes 3 sub-paths of ~33 µm rather than 30-30-40.
    Adjacent sub-paths share their boundary node.
    """
    coords = xyz[path]
    edge_lens = np.linalg.norm(np.diff(coords, axis=0), axis=1)
    L = float(edge_lens.sum())
    n = max(1, round(L / split_length))
    if n == 1:
        return [path]

    cum = np.concatenate([[0.0], np.cumsum(edge_lens)])
    targets = np.linspace(0.0, L, n + 1)

    # node index in path for each interior cut (skip 0 and L)
    cut_idx = [0]
    for t in targets[1:-1]:
        idx = int(np.argmin(np.abs(cum - t)))
        if idx > cut_idx[-1]:  # avoid duplicate split points
            cut_idx.append(idx)
    cut_idx.append(len(path) - 1)

    return [
        path[cut_idx[i] : cut_idx[i + 1] + 1]
        for i in range(len(cut_idx) - 1)
        if cut_idx[i + 1] > cut_idx[i]
    ]


def _orient_classify(
    skel: Skeleton,
    branch_split_length: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Branch-level orientation classifier.

    For each node i returns:
    - vert[i] : |dz| / L for the (sub-)segment containing edge (i -> parent[i]).
                NaN for the root (no parent).
    - tip[i]  : True if node i is in a terminal branch (ends at a leaf).

    A **branch** is a maximal chain of nodes where every interior node has
    exactly one child.  If branch_split_length is given, branches longer than
    that are split into n = round(L / branch_split_length) equal sub-segments
    so that orientation is evaluated at a consistent spatial scale.  Tip
    detection always uses the full branch length, not the sub-segment length.
    """
    raw_parent = _bfs_parents(skel.edges, len(skel.nodes), root=0)
    parent = np.asarray(raw_parent, dtype=np.int64)
    xyz = np.asarray(skel.nodes, dtype=np.float64)
    N = len(xyz)

    n_ch = np.zeros(N, dtype=int)
    child_nodes = np.where(parent != -1)[0]
    if len(child_nodes) == 0:
        return np.full(N, np.nan), np.zeros(N, dtype=bool)
    np.add.at(n_ch, parent[child_nodes], 1)

    ch: list[list[int]] = [[] for _ in range(N)]
    for c in map(int, child_nodes):
        ch[int(parent[c])].append(c)

    vert = np.full(N, np.nan)
    tip = np.zeros(N, dtype=bool)

    # DFS over branches.  Each stack entry is (first_node, branch_top_node).
    # first_node is a direct child of branch_top and is the entry to a new branch.
    stack: list[tuple[int, int]] = [(c, 0) for c in ch[0]]

    while stack:
        first, bstart = stack.pop()

        # Walk the single-child chain to the branch bottom (junction node).
        current = first
        while n_ch[current] == 1:
            current = ch[current][0]
        bend = current

        # Reconstruct ordered branch path by tracing back from bend to bstart.
        path: list[int] = [bend]
        cur = int(bend)
        while cur != bstart:
            cur = int(parent[cur])
            path.append(cur)
        path.reverse()  # [bstart, ..., bend]

        if len(path) >= 2:
            # All terminal branches are tips regardless of length or orientation.
            is_term = bool(n_ch[bend] == 0)
            qualifies_as_tip = is_term

            # Split into sub-segments (or keep as one if no split_length given).
            subs = (
                _split_path(path, xyz, branch_split_length)
                if branch_split_length is not None
                else [path]
            )

            for sub in subs:
                coords = xyz[sub]
                L_sub = float(np.linalg.norm(np.diff(coords, axis=0), axis=1).sum())
                dz_sub = abs(float(coords[-1, 2] - coords[0, 2]))
                bvert = dz_sub / L_sub if L_sub > 0.0 else 0.0

                for node in sub[1:]:  # child nodes in this sub-segment
                    vert[node] = bvert
                    if qualifies_as_tip:
                        tip[node] = True

        # Continue DFS into branches below branch points.
        if n_ch[bend] >= 2:
            for c in ch[bend]:
                stack.append((c, bend))

    return vert, tip


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
    orientation_threshold: float | None = None,
    branch_split_length: float | None = None,  # µm
) -> dict:
    """
    Compute a 1-D depth profile.

    measure:
        "length" – cable length per bin (histogram + KB-smoothed).
        "volume" – **union-correct** morphology volume per bin (voxel union).

    orientation_threshold:
        If set (float in [0, 1]), also compute split profiles for
        measure='length':
          vlength – edges where branch |dz|/L > threshold (vertical /
                    layer-bridging).
          hlength – all remaining edges + tip segments (horizontal /
                    synapse-making).
        The threshold is compared against the (sub-)segment |dz|/L.
        All terminal branches (leaf nodes) always go into hlength regardless
        of their orientation.
        branch_split_length: if set, each branch is divided into
        n = round(L / branch_split_length) equal sub-segments so that
        orientation is evaluated at a consistent spatial scale rather than
        over the full (possibly very long) branch.

    Notes
    -----
    * volume uses a single voxelization (_voxelize_union) and
      **do not double count** at branch junctions or soma contacts.
    * 'include_soma' defaults to False to match the 'length' convention
      (edges only). Set True if you want soma membrane/volume included.
    * For stable plots, keep bin_size >= voxel_size (if you set voxel_size).
    * orientation_threshold is only supported for measure='length'.
    """

    if measure == "length":
        density, mid = segment_lengths(skel)
    elif measure == "volume":
        density, mid = z_slince_volumes(
            skel,
            voxel_size=voxel_size,
            include_soma=include_soma,
            radius_metric=radius_metric,
        )
    else:
        raise ValueError("measure must be one of {'length','volume'}")

    z_vals = mid[:, 2]

    # window
    if extent is None:
        z_min, z_max = np.floor(z_vals.min()), np.ceil(z_vals.max())
        # a skeleton lying exactly on one integer depth still gets one bin
        z_max = max(z_max, z_min + bin_size)
    else:
        z_min, z_max = extent

    # histogram bins
    n_bins = max(1, int(np.ceil((z_max - z_min) / bin_size)))
    edges = z_min + np.arange(n_bins + 1) * bin_size
    edges[-1] = z_max
    x_um = 0.5 * (edges[1:] + edges[:-1])

    # shared KB-smoothing parameters
    centre = (z_min + z_max) / 2.0
    halfspan = (z_max - z_min) / 2.0
    z_samples = (z_vals - centre) / max(halfspan, np.finfo(float).eps)

    def _profile(w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Histogram + KB-smoothed profile for weight array w."""
        tot = float(w.sum())
        if tot == 0.0:
            return np.zeros(n_bins), np.zeros(n_bins)
        hist, _ = np.histogram(z_vals, bins=edges, weights=w)
        if hist.sum() > 0:
            hist = hist * (tot / hist.sum())
        dist = gridder1d(z_samples / 2.0, w, n_bins)
        if dist.sum() > 0:
            dist = dist * (tot / dist.sum())
        return hist, dist

    z_hist, z_dist = _profile(density)
    intervals = hdr(x_um, z_dist, mass=hdr_mass)
    unit = skel.meta.get("unit", "µm")

    result = {
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

    # Orientation split (length only)
    if orientation_threshold is not None:
        if measure != "length":
            raise ValueError("orientation_threshold requires measure='length'")

        vert_vals, tip_mask = _orient_classify(skel, branch_split_length)
        # vertical: not a tip AND branch |dz|/L > threshold
        mask_v = (
            np.isfinite(vert_vals) & (vert_vals > orientation_threshold) & ~tip_mask
        )

        v_hist, v_dist = _profile(density * mask_v)
        h_hist, h_dist = _profile(density * ~mask_v)

        result.update(
            {
                "vlength_distribution": v_dist,
                "vlength_histogram": v_hist,
                "hlength_distribution": h_dist,
                "hlength_histogram": h_hist,
                "orientation_threshold": orientation_threshold,
                "branch_split_length": branch_split_length,
            }
        )

    return result


def _edges_from_bin_size(lo: float, hi: float, bin_size: float) -> np.ndarray:
    """Generate edges >= bin_size wide, last bin clipped to *hi*."""
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
        Fixed window ``[xmin, xmax, ymin, ymax]`` (µm).  *None* -> tight box.
    bin_size
        Side length of each square bin (µm).  `edges = lo + k·bin_size`.
        The window is *expanded* to the next multiple so that every bin is
        exactly `bin_size` wide.  (This guarantees comparability.)
    smooth
        sigma of the Gaussian kernel (bins) applied to the histogram.
    measure:
        "length" – dendritic cable length per bin (histogram -> Gaussian smooth).
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
        "n_bins": (len(x), len(y)),  # may differ if dx != dy
        "bin_size": bin_size,
        "smooth": smooth,
        "measure": measure,
        "units": units,
        "include_soma": include_soma,
        "voxel_size": voxel_size,
        "radius_metric": radius_metric,
    }


def get_coverage(skel: Skeleton, extent: list[float] | tuple[float, float]) -> dict:
    """
    How much of a warped skeleton's cable lies where the data do not cover it.

    Parameters
    ----------
    skel : Skeleton
        Output of `warp_skeleton` or `Warper.renormalize`.
    extent : (z_min, z_max)
        Depth window in the warped skeleton's z units, e.g. its z-profile extent.

    Returns
    -------
    dict
        Shares of the cable length, each NaN when the skeleton has no record of it
        (see `node_coverage`):

        ``cut_below`` / ``cut_above``
            The imaged volume ends inside the window, below z_min / above z_max,
            in that cable's column.
        ``on_unsupported`` / ``off_unsupported``
            That band was not observed in the cable's column.
    """
    out = {
        k: np.nan
        for k in ("cut_below", "cut_above", "on_unsupported", "off_unsupported")
    }
    lengths, _ = segment_lengths(skel)
    total = lengths.sum()
    coverage = (skel.extra or {}).get("coverage") or {}
    if total <= 0:
        return out
    z_min, z_max = extent
    if "depth" in coverage:
        depth = np.asarray(coverage["depth"], dtype=float)
        out["cut_below"] = float(lengths[depth[:, 0] > z_min].sum() / total)
        out["cut_above"] = float(lengths[depth[:, 1] < z_max].sum() / total)
    for band in ("on", "off"):
        if f"{band}_support" in coverage:
            support = np.asarray(coverage[f"{band}_support"], dtype=bool)
            out[f"{band}_unsupported"] = float(lengths[~support].sum() / total)
    return out


def hdr(z_centres, z_density, mass=0.95):
    """
    High Density Region (HDR) estimator.

    Parameters
    ----------
    z_centres : (N,) bin centres (µm)
    z_density : (N,) density per bin (any units)
    mass      : float, 0 < mass <= 1 (e.g. 0.95)

    Example
    -------
    >>> hdr(z, d, 0.95)
    [array([ -2.1,  1.7]),   # ON sheet
     array([ 10.3, 13.9])]   # OFF sheet

    A profile without mass has no high-density region: the result is empty.
    """
    total = z_density.sum()
    if not total > 0:
        return []
    p = z_density / total  # normalise -> probability
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
    """High-level interface around *pywarper* for IPL flattening."""

    def __init__(
        self,
        off_sac_points: dict[str, np.ndarray]
        | tuple[np.ndarray, np.ndarray, np.ndarray]
        | None = None,
        on_sac_points: dict[str, np.ndarray]
        | tuple[np.ndarray, np.ndarray, np.ndarray]
        | None = None,
        swc_path: str | None = None,
        *,
        voxel_resolution: list[float] = [1.0, 1.0, 1.0],
        verbose: bool = False,
    ) -> None:
        self.voxel_resolution = voxel_resolution
        self.verbose = verbose
        self.swc_path = swc_path
        # what fit_surfaces() settled on, per band; folded into the mapping meta
        self.surface_fit: dict[str, dict] = {}

        if off_sac_points is not None:
            self.off_sac_points = self._as_xyz(off_sac_points)
        if on_sac_points is not None:
            self.on_sac_points = self._as_xyz(on_sac_points)

        if swc_path is not None:
            self.swc_path = swc_path
            self.load_swc(swc_path)  # raw SWC -> self.nodes / edges / radii
        else:
            self.swc_path = None

    # ---------------------------- IO -------------------------------------
    def load_swc(self, swc_path: str | None = None) -> "Warper":
        """Load the skeleton from *swc_path*."""

        if self.verbose:
            print(f"[pywarper] Loading skeleton -> {self.swc_path}")

        if swc_path is None:
            swc_path = self.swc_path

        if swc_path is not None:
            self.skeleton = sk.io.load_skeleton_swc(swc_path)
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
            "SAC data must be a mapping with keys x/y/z or a 3-tuple of arrays."
        )

    def load_sac(self, off_sac_points, on_sac_points) -> "Warper":
        """Load the SAC meshes from *off_sac_points* and *on_sac_points*."""
        if self.verbose:
            print("[pywarper] Loading SAC meshes...")
        self.off_sac_points = self._as_xyz(off_sac_points)
        self.on_sac_points = self._as_xyz(on_sac_points)
        return self

    def load_warped_skeleton(
        self,
        filepath: str,
        med_z_on: float | None = None,
        med_z_off: float | None = None,
    ) -> None:
        """Load a warped skeleton from *swc_path*."""
        path = Path(filepath)

        if path.suffix.lower() == ".swc":
            self.warped_skeleton = sk.io.load_skeleton_swc(path)

            if (med_z_on is not None) and (med_z_off is not None):
                self.warped_skeleton.extra["med_z_on"] = float(med_z_on)
                self.warped_skeleton.extra["med_z_off"] = float(med_z_off)
            else:
                self.warped_skeleton.extra["med_z_on"] = None
                self.warped_skeleton.extra["med_z_off"] = None
        elif path.suffix.lower() == ".npz":
            self.warped_skeleton = sk.io.load_npz(path)

        if self.verbose:
            print(f"[pywarper] Loaded warped skeleton -> {path}")

    # ---------------------------- Core -----------------------------------

    def fit_surfaces(
        self,
        xmax: int | float | None = None,
        ymax: int | float | None = None,
        method: str = "gridfit",
        stride: int | None = None,
        smoothness: int | float | str | None = None,
        gamma: float | None = None,
        k: int | tuple[int, int] | None = None,
        bs: str | None = None,
        backward_compatible: bool = False,
    ) -> "Warper":
        """Fit the ON / OFF SAC meshes.

        Parameters
        ----------
        method : {"gridfit", "gam"}, default="gridfit"
            Which fitter to use, see :func:`pywarper.surface.fit_sac_surface`.
        stride, smoothness : optional
            ``"gridfit"`` settings; default to 3 and 15 here. Pass
            ``smoothness="auto"`` to select it from the data instead -- what each
            band settled on lands in ``self.surface_fit[band]``, since the two are
            fitted separately and need not agree. Rejected under ``"gam"``.
        gamma : float, optional
            ``"gridfit"`` under ``smoothness="auto"`` only; the length scale the
            selection is optimal for. See :func:`pywarper.surface.fit_sac_surface`.
        k, bs : optional
            ``"gam"`` settings; default to 100 and "tp". Rejected under
            ``"gridfit"``.
        """
        if self.verbose:
            print("[pywarper] Fitting SAC surfaces...")

        if backward_compatible is False and (xmax is None or ymax is None):
            # use the bounding box of the skeleton
            xmax = max(self.off_sac_points[0].max(), self.on_sac_points[0].max())
            ymax = max(self.off_sac_points[1].max(), self.on_sac_points[1].max())

        if method == "gridfit":
            stride = 3 if stride is None else stride
            smoothness = 15 if smoothness is None else smoothness

        self.surface_fit = {}
        for band, points, label in (
            ("off", self.off_sac_points, "OFF (max)"),
            ("on", self.on_sac_points, "ON (min)"),
        ):
            _t0 = time.time()
            surface = fit_sac_surface(
                x=points[0],
                y=points[1],
                z=points[2],
                xmax=xmax,
                ymax=ymax,
                method=method,
                stride=stride,
                smoothness=smoothness,
                gamma=gamma,
                k=k,
                bs=bs,
                backward_compatible=backward_compatible,
            )
            setattr(self, f"{band}_sac_surface", surface.zmesh)
            setattr(self, f"{band}_sac_surface_se", surface.se)
            self.surface_fit[band] = surface.summary
            if self.verbose:
                print(
                    f"-> fitting {label} surface\n    done in {time.time() - _t0:.2f} seconds."
                )
        return self

    def build_mapping(
        self,
        bounds: np.ndarray | tuple | str | None = "local",
        conformal_jump: int = 2,
        n_anchors: int = 16,
        backward_compatible: bool = False,
        *,
        volume_z: tuple[float | None, float | None] | None = None,
        on_support: np.ndarray | None = None,
        off_support: np.ndarray | None = None,
    ) -> "Warper":
        """Create the quasi-conformal surface mapping.

        `volume_z`, `on_support` and `off_support` are passed on to
        :func:`pywarper.surface.build_mapping`; warped skeletons then record
        their coverage.
        """
        if self.off_sac_surface is None or self.on_sac_surface is None:
            raise RuntimeError("Surfaces not fitted. Call fit_surfaces() first.")

        if bounds is None or bounds == "local":
            # skeleton-derived box (rounded to int so it plays nicely with
            # backward-compatible 1-based code paths)
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
            # use whichever SAC fit is larger in each axis
            nx = max(self.on_sac_surface.shape[0], self.off_sac_surface.shape[0])
            ny = max(self.on_sac_surface.shape[1], self.off_sac_surface.shape[1])
            bounds = np.array([0, nx, 0, ny], dtype=float)
        else:
            bounds = np.asarray(bounds, dtype=float)
            if bounds.shape != (4,):
                raise ValueError(
                    "Bounds must be a 4-element array or tuple (x_min, x_max, y_min, y_max)."
                )

        if self.verbose:
            print("[pywarper] Building mapping...")
        self.mapping: dict = build_mapping(
            self.on_sac_surface,
            self.off_sac_surface,
            bounds,
            conformal_jump=conformal_jump,
            n_anchors=n_anchors,
            volume_z=volume_z,
            on_support=on_support,
            off_support=off_support,
            backward_compatible=backward_compatible,
            verbose=self.verbose,
        )
        if self.surface_fit:
            # so a warped skeleton records how its surfaces were fitted
            self.mapping["meta"]["surface_fit"] = self.surface_fit
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
        """Renormalize the warped skeleton to the desired ON/OFF SAC positions.

        The result is in the same units as the output of `warp_skeleton`: the
        voxel resolution and node scale that the warp recorded are undone before
        renormalizing and applied again after. Skeletons warped before those were
        recorded fall back to this Warper's `voxel_resolution` and a scale of 1.
        """
        if self.warped_skeleton is None:
            raise RuntimeError("Warped skeleton missing. Call warp_skeleton() first.")
        extra = self.warped_skeleton.extra
        voxel_resolution = np.broadcast_to(
            np.asarray(
                extra.get("voxel_resolution", self.voxel_resolution), dtype=float
            ),
            (3,),
        )
        scale = float(extra.get("skeleton_nodes_scale", 1.0))
        renormed_nodes = (
            normalize_nodes(
                np.asarray(extra["prenormed_nodes"], dtype=float) / voxel_resolution,
                med_z_on=extra["med_z_on"],
                med_z_off=extra["med_z_off"],
                on_sac_pos=on_sac_pos,
                off_sac_pos=off_sac_pos,
            )
            / scale
            * voxel_resolution
        )

        soma_renormed = deepcopy(self.warped_skeleton.soma)
        soma_renormed.center = renormed_nodes[0]

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
            "prenormed_nodes": self.warped_skeleton.extra[
                "prenormed_nodes"
            ],  # keep the pre-normed warped nodes for future use
            "med_z_on": float(self.warped_skeleton.extra["med_z_on"]),
            "med_z_off": float(self.warped_skeleton.extra["med_z_off"]),
            "voxel_resolution": voxel_resolution.copy(),
            "skeleton_nodes_scale": scale,
            "z_profiles": z_profiles,
            "xy_profiles": xy_profiles,
        }
        if "coverage" in extra:
            coverage = dict(extra["coverage"])
            if "depth_rel" in coverage:
                coverage["depth"] = _covered_depth(
                    coverage["depth_rel"],
                    on_sac_pos,
                    off_sac_pos,
                    scale,
                    voxel_resolution[2],
                )
            skel_renormed.extra["coverage"] = coverage
        skel_renormed.meta.update(
            {
                "pywarper_version": _PYWARPER_VERSION,
                "renormed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
        )

        return skel_renormed
