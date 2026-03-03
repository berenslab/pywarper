"""
pywarper.surface
================
Numerical utilities for **flattening the Starburst Amacrine Cell (SAC) layers** of a retina.

Given two depth maps—one for the ON SAC band and one for the OFF SAC band—this module performs

1. **Surface fitting** (`fit_surface`) – smooths scattered ChAT-band samples or arbor node coordinates
   into regular height-fields using *PyGridFit*.
2. **Uniform resampling** (`resample_zgrid`) – converts the irregular fit to a unit-spaced integer grid,
   matching MATLAB’s historical conventions.
3. **Diagonal length measurement** (`calculate_diag_length`) – computes the true 3-D lengths of the main
   and skew diagonals; these serve as scale anchors for the conformal map.
4. **Quasi-conformal mapping** (`conformal_map_indep_fixed_diagonals`) – straightens the two diagonals
   while optimally preserving local angles, yielding 2-D coordinates for every voxel.
5. **Map alignment** (`align_mapped_surface`) – rigidly shifts the OFF map so that its local slope
   pattern best matches the ON map via patch-wise gradient minimisation.
6. **Map building** (`build_mapping`) – runs the whole pipeline and returns the flattened
   mapping along with diagnostic metadata.

The resulting 2-D coordinates mapping can be applied to any neurite morphology located between the SAC layers
so that axonal and dendritic trees can be visualised *as if* the inner plexiform layer were perfectly
flat.
"""
import time

import numpy as np
from pygridfit import GridFit
from scipy.interpolate import RegularGridInterpolator
from scipy.signal import convolve2d
from scipy.sparse import coo_matrix, hstack, vstack
from scipy.sparse.linalg import spsolve

try:
    from sksparse.cholmod import cholesky
    HAS_CHOLMOD = True
except ImportError:
    HAS_CHOLMOD = False
    _WARN_MSG = (
        "[pywarper.surface] Optional dependency 'scikit-sparse' (CHOLMOD bindings) not found. "
        "Falling back to SciPy's sparse linear solver, which is ≈5–10× slower for large problems.\n\n"
        "For platform-specific instructions see the project README:\n"
        "\thttps://github.com/berenslab/pywarper#installation"
    )
    print(_WARN_MSG)

from importlib import metadata as _metadata

_PYWARPER_VERSION = _metadata.version("pywarper")

def fit_surface(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    xmax: int | float | None = None,
    ymax: int | float | None = None,
    stride: int = 3, 
    smoothness: int = 1,
    extend: str = "warning",
    interp: str = "triangle",
    regularizer: str = "gradient",
    solver: str = "normal",
    maxiter: int | None = None,
    autoscale: str = "on",
    xscale: float = 1.0,
    yscale: float = 1.0,
    backward_compatible: bool = False
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Fits a surface to scattered data points (x, y, z) using grid-based interpolation
    and smoothing. Internally uses a GridFit-based approach to produce a 2D surface.

    Parameters
    ----------
    x : np.ndarray
        The x-coordinates of input data points.
    y : np.ndarray
        The y-coordinates of input data points.
    z : np.ndarray
        The z-values at each (x, y) coordinate.
    xmax : int, optional
        Maximum value along the x-axis used to define the interpolation grid.
        If None, the max value from x is used.
    ymax : int, optional
        Maximum value along the y-axis used to define the interpolation grid.
        If None, the max value from y is used.
    smoothness : int, default=1
        Amount of smoothing applied during fitting.
    extend : str, default="warning"
        Determines how to handle extrapolation outside data boundaries.
        Possible values include "warning", "fill", etc. (see GridFit docs).
    interp : str, default="triangle"
        Type of interpolation to apply (e.g., "triangle", "bilinear").
    regularizer : str, default="gradient"
        Regularization method used in the solver (e.g., "gradient", "laplacian").
    solver : str, default="normal"
        Solver backend (e.g., "normal" for normal equations).
    maxiter : int, optional
        Maximum number of solver iterations. If None, defaults to solver-based value.
    autoscale : str, default="on"
        Autoscaling setting for the solver.
    xscale : float, default=1.0
        Additional scaling factor applied to the x-dimension during fitting.
    yscale : float, default=1.0
        Additional scaling factor applied to the y-dimension during fitting.
    backward_compatible : bool, default=False
        If True, use the same node spacing as the original MATLAB implementation.
        
    Returns
    -------
    zmesh: np.ndarray (xmax, ymax)
        2D array of interpolated z-values over the fitted surface / Interpolated surface heights.
    xmesh, ymesh: np.ndarray (xma, ymax)
        Grid coordinate matrices matching zmesh.
    """
    if xmax is None:
        xmax = np.max(x).astype(float)
    if ymax is None:
        ymax = np.max(y).astype(float)

    if backward_compatible:
        # MATLAB-style nodes
        xnodes = np.hstack([np.arange(1., xmax, stride), np.array([xmax])])
        ynodes = np.hstack([np.arange(1., ymax, stride), np.array([ymax])])
    else:
        xnodes = np.arange(0, xmax + stride, stride)
        ynodes = np.arange(0, ymax + stride, stride)

    g = GridFit(x, y, z, xnodes, ynodes, 
                    smoothness=smoothness,
                    extend=extend,
                    interp=interp,
                    regularizer=regularizer,
                    solver=solver,
                    maxiter=maxiter,
                    autoscale=autoscale,
                    xscale=xscale,
                    yscale=yscale,
        ).fit()
    zgrid = np.asarray(g.zgrid)

    zmesh, xmesh, ymesh = resample_zgrid(
        xnodes, ynodes, zgrid, xmax, ymax, backward_compatible
    )

    return zmesh, xmesh, ymesh

def resample_zgrid(
    xnodes: np.ndarray,
    ynodes: np.ndarray,
    zgrid: np.ndarray,
    xmax: int | float,
    ymax: int | float,
    backward_compatible: bool = False
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Resamples a 2D grid (zgrid) at integer coordinates up to xmax and ymax.
    Uses a linear RegularGridInterpolator under the hood.

    Parameters
    ----------
    xnodes : np.ndarray
        Sorted 1D array of x-coordinates defining the original grid.
    ynodes : np.ndarray
        Sorted 1D array of y-coordinates defining the original grid.
    zgrid : np.ndarray
        2D array of shape (len(ynodes), len(xnodes)), representing z-values
        on a regular grid with axes (y, x).
    xmax : int
        The maximum x-coordinate (inclusive) for the resampling.
    ymax : int
        The maximum y-coordinate (inclusive) for the resampling.

    Returns
    -------
    vzmesh : np.ndarray
        2D array of shape (xmax, ymax), containing interpolated z-values at
        integer (x, y) positions.
    xi : np.ndarray
        2D array of shape (xmax, ymax), representing the x-coordinates used for
        interpolation.
    yi : np.ndarray
        2D array of shape (xmax, ymax), representing the y-coordinates used for
        interpolation.

    Notes
    -----
    In Python, arrays are typically indexed as (row, column) which maps to
    (y, x) in a 2D sense. This function transposes the meshgrid from
    `np.meshgrid(..., indexing='xy')` to match the MATLAB style of indexing.
    """

    # 0) Check that xmax, ymax are integers.
    #    If not, round to nearest integer.
    xmax = round(xmax)
    ymax = round(ymax)

    # 1) Build the interpolator, 
    #    specifying x= xnodes (ascending), y= ynodes (ascending).
    #    Note that in Python, the first axis in zgrid is y, second is x.
    #    So pass (ynodes, xnodes) in that order:
    rgi = RegularGridInterpolator(
        (ynodes, xnodes),  # (y-axis, x-axis)
        zgrid, 
        method="linear", 
        bounds_error=False, 
        fill_value=np.nan  # or e.g. zgrid.mean()
    )

    # 2) Make xi, yi as in MATLAB, 
    #    then do xi=xi', yi=yi' => shape (xmax, ymax).
    if backward_compatible:
        xi_m, yi_m = np.meshgrid(
            np.arange(1, xmax+1), 
            np.arange(1, ymax+1), 
            indexing='xy'
        )
    else:
        xi_m, yi_m = np.meshgrid(
            np.arange(0, xmax), 
            np.arange(0, ymax), 
            indexing='xy'
        )
    xi = xi_m.T  # shape (xmax, ymax)
    yi = yi_m.T  # shape (xmax, ymax)

    # 3) Flatten the coordinate arrays to shape (N, 2) for RGI.
    XYi = np.column_stack((yi.ravel(), xi.ravel()))
    # We must pass (y, x) in that order since RGI is (y-axis, x-axis).

    # 4) Interpolate.
    vmesh_flat = rgi(XYi)  # 1D array, length xmax*ymax

    # 5) Reshape to (xmax, ymax).
    vzmesh = vmesh_flat.reshape((xmax, ymax))

    return vzmesh, xi, yi


def calculate_diag_length(
    xpos: np.ndarray,
    ypos: np.ndarray,
    VZmesh: np.ndarray
) -> tuple[float, float]:
    """
    Computes the 3D length along the main and skew diagonals of VZmesh
    (exactly the same result as the original implementation).

    Parameters
    ----------
    xpos, ypos, VZmesh : see original docstring.

    Returns
    -------
    main_diag_dist, skew_diag_dist : float
    """
    M, N = VZmesh.shape  # M = len(xpos), N = len(ypos)

    # Build regular-grid interpolators
    interp_x = RegularGridInterpolator(
        (xpos, ypos),
        np.meshgrid(xpos, ypos, indexing="ij")[0],
        method="linear"
    )
    interp_y = RegularGridInterpolator(
        (xpos, ypos),
        np.meshgrid(xpos, ypos, indexing="ij")[1],
        method="linear"
    )
    interp_z = RegularGridInterpolator(
        (xpos, ypos), VZmesh, method="linear"
    )

    if N >= M:
        # vectors of length N
        x_diag = np.linspace(xpos[0], xpos[-1], N)
        y_main = ypos
        y_skew = y_main[::-1]

        pts_main = np.column_stack((x_diag, y_main))
        pts_skew = np.column_stack((x_diag, y_skew))
    else:
        # vectors of length M
        y_diag = np.linspace(ypos[0], ypos[-1], M)
        x_main = xpos
        x_skew = x_main[::-1]

        pts_main = np.column_stack((x_main, y_diag))
        pts_skew = np.column_stack((x_skew, y_diag))

    # Evaluate coordinates on both diagonals
    x_main_v = interp_x(pts_main)
    y_main_v = interp_y(pts_main)
    z_main_v = interp_z(pts_main)

    x_skew_v = interp_x(pts_skew)
    y_skew_v = interp_y(pts_skew)
    z_skew_v = interp_z(pts_skew)

    # Stack, diff, and accumulate Euclidean distances (vectorised, no Python loop)
    diffs_main = np.diff(
        np.stack((x_main_v, y_main_v, z_main_v), axis=1), axis=0
    )
    diffs_skew = np.diff(
        np.stack((x_skew_v, y_skew_v, z_skew_v), axis=1), axis=0
    )

    main_diag_dist = np.sqrt((diffs_main ** 2).sum(1)).sum()
    skew_diag_dist = np.sqrt((diffs_skew ** 2).sum(1)).sum()

    return main_diag_dist, skew_diag_dist


def assign_local_coordinates(triangles: np.ndarray) -> tuple[np.ndarray, ...]:
    """
    Vectorised local complex coordinates for many triangles at once.

    Parameters
    ----------
    triangles : np.ndarray
        Shape (T, 3, 3).  triangles[:, i, :] is the (x,y,z) of vertex i.

    Returns
    -------
    w1, w2, w3 : np.ndarray, shape (T,)
    zeta       : np.ndarray, shape (T,)
    """
    v1 = triangles[:, 0, :]
    v2 = triangles[:, 1, :]
    v3 = triangles[:, 2, :]

    d12 = np.linalg.norm(v1 - v2, axis=1)
    d13 = np.linalg.norm(v1 - v3, axis=1)
    d23 = np.linalg.norm(v2 - v3, axis=1)

    y3 = ((-d12) ** 2 + d13 ** 2 - d23 ** 2) / (2 * -d12)
    x3 = np.sqrt(np.maximum(0.0, d13 ** 2 - y3 ** 2))

    w2 = -x3 - 1j * y3
    w1 =  x3 + 1j * (y3 + d12)
    w3 = 1j * (-d12)

    zeta = np.abs(np.real(1j * (np.conj(w2) * w1 - np.conj(w1) * w2)))
    return w1, w2, w3, zeta


def conformal_map_indep_fixed_diagonals(
    mainDiagDist: float,
    skewDiagDist: float,
    xpos: np.ndarray,
    ypos: np.ndarray,
    VZmesh: np.ndarray,
    *,
    n_anchors: int = 16,        # 4, 8 (default) or 16
    backward_compatible: bool = False
) -> np.ndarray:
    """
    Creates a quasi-conformal 2D mapping of the surface in VZmesh. 
    Diagonal constraints are fixed using mainDiagDist and skewDiagDist 
    for consistent scaling.

    Parameters
    ----------
    mainDiagDist : float
        Target distance along the main diagonal for the mapped surface.
    skewDiagDist : float
        Target distance along the skew (reverse) diagonal for the mapped surface.
    xpos : np.ndarray
        1D array of x-coordinates (length M).
    ypos : np.ndarray
        1D array of y-coordinates (length N).
    VZmesh : np.ndarray
        2D array of shape (M, N), representing z-values for each (x, y).
    n_anchors : int, default=16
        Number of anchor points to use for the conformal mapping.
        Options are 4, 8 (default), or 16 anchors.
            - 4   → original behaviour (two separate solves, then average)  
            - 8   → add horizontal/vertical mid-lines (single solve)  
            - 16  → also add the quarter-lines (single solve)
        
    Returns
    -------
    mappedPositions : np.ndarray
        2D array of shape (M*N, 2). Each row corresponds to the (x, y) position
        in the conformal map for the corresponding vertex in the original mesh.

    Notes
    -----
    The mapping is generated by splitting each cell of the grid into two triangles,
    constructing a sparse system to enforce approximate conformality, and then
    solving for new vertex positions subject to diagonally fixed boundaries.
    The final 2D layout merges two separate diagonal constraints.
    """  
    M, N = VZmesh.shape

    if backward_compatible:
        xpos_new = xpos + 1
        ypos_new = ypos + 1
    else:
        xpos_new = xpos
        ypos_new = ypos
    vertexCount   = M * N
    triangleCount = (2 * M - 2) * (N - 1)

    # -----------------------------------------------------------
    # 1. triangulation on the regular grid
    # -----------------------------------------------------------
    col1   = np.kron([1, 1], np.arange(M - 1))
    temp1  = np.kron([1, M + 1], np.ones(M - 1))
    temp2  = np.kron([M + 1, M], np.ones(M - 1))
    onecol = np.stack([col1, col1 + temp1, col1 + temp2], axis=1).astype(int)

    triangulation = np.tile(onecol, (N - 1, 1))
    triangulation += np.repeat(np.arange(N - 1), 2 * M - 2)[:, None] * M
    rows = triangulation % M
    cols = triangulation // M

    # -----------------------------------------------------------
    # 2. complex local coordinates
    # -----------------------------------------------------------
    tri_xyz = np.empty((triangleCount, 3, 3), dtype=np.float64)
    tri_xyz[:, :, 0] = xpos_new[rows]
    tri_xyz[:, :, 1] = ypos_new[cols]
    tri_xyz[:, :, 2] = VZmesh[rows, cols]

    w1, w2, w3, zeta = assign_local_coordinates(tri_xyz)
    denom = np.sqrt(zeta / 2.0)

    ws_real = np.column_stack([np.real(w1), np.real(w2), np.real(w3)]) / denom[:, None]
    ws_imag = np.column_stack([np.imag(w1), np.imag(w2), np.imag(w3)]) / denom[:, None]

    ridx = np.repeat(np.arange(triangleCount), 3)
    cidx = triangulation.ravel()

    Mreal = coo_matrix((ws_real.ravel(), (ridx, cidx)),
                       shape=(triangleCount, vertexCount)).tocsr()
    Mimag = coo_matrix((ws_imag.ravel(), (ridx, cidx)),
                       shape=(triangleCount, vertexCount)).tocsr()

    # -----------------------------------------------------------
    # 3. linear solver helper
    # -----------------------------------------------------------
    def solve_mapping(fixed_pts: list[int],
                      fixed_vals: np.ndarray,
                      free_pts: np.ndarray) -> np.ndarray:

        A = vstack([
            hstack([Mreal[:, free_pts], -Mimag[:, free_pts]]),
            hstack([Mimag[:, free_pts],  Mreal[:, free_pts]])
        ])

        b_real = Mreal[:, fixed_pts] @ fixed_vals[:, 0] - \
                 Mimag[:, fixed_pts] @ fixed_vals[:, 1]
        b_imag = Mimag[:, fixed_pts] @ fixed_vals[:, 0] + \
                 Mreal[:, fixed_pts] @ fixed_vals[:, 1]
        b = -np.concatenate([b_real, b_imag])

        AtA = (A.T @ A).tocsc()
        Atb = A.T @ b
        if HAS_CHOLMOD:
            sol = cholesky(AtA)(Atb)
        else:
            sol = spsolve(AtA, Atb)

        nf = len(free_pts)
        mapped = np.zeros((vertexCount, 2))
        mapped[fixed_pts]    = fixed_vals
        mapped[free_pts, 0]  = sol[:nf]
        mapped[free_pts, 1]  = sol[nf:]
        return mapped

    # -----------------------------------------------------------
    # 4. set up diagonal anchors (always present)
    # -----------------------------------------------------------
    diag_scale = M / np.sqrt(M**2 + N**2)

    main_fixed_pts  = [0, vertexCount - 1]
    main_fixed_vals = np.array([
        [xpos_new[0], ypos_new[0]],
        [xpos_new[0] + mainDiagDist * diag_scale,
         ypos_new[0] + mainDiagDist * diag_scale * N / M]
    ])

    skew_fixed_pts  = [M - 1, vertexCount - M]
    skew_fixed_vals = np.array([
        [xpos_new[0] + skewDiagDist * diag_scale, ypos_new[0]],
        [xpos_new[0],
         ypos_new[0] + skewDiagDist * diag_scale * N / M]
    ])

    # -----------------------------------------------------------
    # 5. branch on anchor count
    # -----------------------------------------------------------
    if n_anchors == 4:
        # --- historical behaviour: two solves, then average ----------
        free_main = np.setdiff1d(np.arange(vertexCount), main_fixed_pts)
        map_main  = solve_mapping(main_fixed_pts, main_fixed_vals, free_main)

        free_skew = np.setdiff1d(np.arange(vertexCount), skew_fixed_pts)
        map_skew  = solve_mapping(skew_fixed_pts, skew_fixed_vals, free_skew)

        mappedPositions = 0.5 * (map_main + map_skew)

    else:
        # --- single solve with additional anchors -------------------
        fixed_pts  : list[int]       = main_fixed_pts + skew_fixed_pts
        fixed_vals : list[np.ndarray] = [main_fixed_vals, skew_fixed_vals]

        # add mid-lines (8 anchors) and quarter-lines (16 anchors)
        if n_anchors >= 8:
            mid_cols = [N // 2]
            mid_rows = [M // 2]
            if n_anchors == 16:
                mid_cols += [N // 4, 3 * N // 4]
                mid_rows += [M // 4, 3 * M // 4]

            # horizontals
            for c in mid_cols:
                idx_left  = 0       + c * M
                idx_right = (M - 1) + c * M
                dz = VZmesh[M - 1, c] - VZmesh[0, c]
                length = np.sqrt((xpos[-1] - xpos[0])**2 + dz**2)
                fixed_pts += [idx_left, idx_right]
                fixed_vals.append(np.array([
                    [xpos_new[0],                 ypos_new[c]],
                    [xpos_new[0] + length,        ypos_new[c]]
                ]))

            # verticals
            for r in mid_rows:
                idx_top    = r + 0 * M
                idx_bottom = r + (N - 1) * M
                dz = VZmesh[r, N - 1] - VZmesh[r, 0]
                length = np.sqrt((ypos[-1] - ypos[0])**2 + dz**2)
                fixed_pts += [idx_top, idx_bottom]
                fixed_vals.append(np.array([
                    [xpos_new[r], ypos_new[0]],
                    [xpos_new[r], ypos_new[0] + length]
                ]))

        fixed_vals = np.vstack(fixed_vals)
        free_pts   = np.setdiff1d(np.arange(vertexCount), fixed_pts)
        mappedPositions = solve_mapping(fixed_pts, fixed_vals, free_pts)

    return mappedPositions


def align_mapped_surface(
    ref_surface: np.ndarray,
    target_surface: np.ndarray,
    ref_mapped: np.ndarray,
    target_mapped: np.ndarray,
    xborders: list[int],
    yborders: list[int],
    conformal_jump: int = 1,
    patch_size: int = 21
) -> np.ndarray:
    """
    Shifts *target_mapped* so that its local gradients align best with
    those of *ref_mapped*.

    Parameters
    ----------
    ref_surface : np.ndarray
        2D height map (X, Y) of the reference surface.
    target_surface : np.ndarray
        2D height map (X, Y) of the surface to shift.
    ref_mapped : np.ndarray
        (X*Y, 2) conformally mapped coordinates for the reference surface.
    target_mapped : np.ndarray
        (X*Y, 2) conformally mapped coordinates for the target surface
        (will be shifted in-place).
    xborders : list of int
        [x_min, x_max] bounding indices used to focus the alignment region.
    yborders : list of int
        [y_min, y_max] bounding indices used to focus the alignment region.
    conformal_jump : int, default=1
        Subsampling step in x and y dimensions for alignment calculations.
    patch_size : int, default=21
        Size of the local 2D window used for minimizing gradient differences.

    Returns
    -------
    target_mapped : np.ndarray
        Updated (X*Y, 2) for the target surface, after alignment.
    """
    patch_size = int(np.ceil(patch_size / conformal_jump))

    # Pad surfaces to preserve shape after differencing
    pad_val_ref = 10 * np.max(ref_surface)
    pad_val_tgt = 10 * np.max(target_surface)

    ref_padded = np.pad(ref_surface, ((0, 1), (0, 1)), constant_values=pad_val_ref)
    tgt_padded = np.pad(target_surface, ((0, 1), (0, 1)), constant_values=pad_val_tgt)

    # Gradient differences (dx + i*dy)
    dref_dx = np.diff(ref_padded, axis=0)[:, :-1]
    dref_dy = np.diff(ref_padded, axis=1)[:-1, :]
    dRefSurface = np.abs(dref_dx + 1j * dref_dy)

    dtgt_dx = np.diff(tgt_padded, axis=0)[:, :-1]
    dtgt_dy = np.diff(tgt_padded, axis=1)[:-1, :]
    dTgtSurface = np.abs(dtgt_dx + 1j * dtgt_dy)

    # Region of interest
    x1, x2 = xborders
    y1, y2 = yborders

    dRefSurface_roi = dRefSurface[x1:x2+1:conformal_jump, y1:y2+1:conformal_jump]
    dTgtSurface_roi = dTgtSurface[x1:x2+1:conformal_jump, y1:y2+1:conformal_jump]

    combined_slope = dRefSurface_roi + dTgtSurface_roi

    # Patch cost = sum of local gradients over patch
    kernel = np.ones((patch_size, patch_size))
    patch_costs = convolve2d(combined_slope, kernel, mode='valid')

    min_index = np.argmin(patch_costs)
    row0, col0 = np.unravel_index(min_index, patch_costs.shape)

    row_center_0b = int(round(row0 + (patch_size - 1) / 2))
    col_center_0b = int(round(col0 + (patch_size - 1) / 2))

    flat_index = col_center_0b * dRefSurface_roi.shape[0] + row_center_0b

    # Then do the shift
    shift_x = target_mapped[flat_index, 0] - ref_mapped[flat_index, 0]
    shift_y = target_mapped[flat_index, 1] - ref_mapped[flat_index, 1]

    target_mapped[:, 0] -= shift_x
    target_mapped[:, 1] -= shift_y

    return target_mapped


def build_mapping(
    surfaces: dict[str, np.ndarray],
    bounds: np.ndarray | tuple[int, int, int, int],
    conformal_jump: int = 1,
    n_anchors: int = 16,
    alignment_patch_size: int = 21,
    *,
    verbose: bool = False,
    backward_compatible: bool = False,
) -> dict:
    """
    Create a 2D conformal map that **flattens** N tagged depth surfaces onto
    a common plane.

    Parameters
    ----------
    surfaces : dict[str, np.ndarray] | None
        Mapping of tag -> (X, Y) height map.  Surfaces are auto-sorted by
        median depth (shallowest first).
    bounds : tuple[int, int, int, int] | np.ndarray
        *(xmin, xmax, ymin, ymax)* bounds of the region of interest.
    conformal_jump : int, default 1
        Sub-sampling stride.
    n_anchors : int, default 16
        Number of anchor points for the conformal mapping (4, 8, or 16).
    alignment_patch_size : int, default 21
        Patch size for surface alignment.
    on_sac_surface : np.ndarray | None
        LEGACY keyword-only. If *surfaces* is None, this and *off_sac_surface*
        are used to construct ``surfaces = {"on_sac": ..., "off_sac": ...}``.
    off_sac_surface : np.ndarray | None
        LEGACY keyword-only. See *on_sac_surface*.
    verbose : bool, default False
        Print timing info.
    backward_compatible : bool, default False
        Use MATLAB-compatible indexing.

    Returns
    -------
    dict
        New-format mapping with keys: ``surfaces``, ``mapped_surfaces``,
        ``surface_order``, ``main_diag_dist``, ``skew_diag_dist``,
        ``sampled_x_idx``, ``sampled_y_idx``, ``n_anchors``,
        ``conformal_jump``, ``meta``.
        Legacy keys ``mapped_on``, ``mapped_off``, ``on_sac_surface``,
        ``off_sac_surface`` are also included for backward compatibility.
    """
    # ---- validate inputs -----------------------------------------------------
    if not surfaces:
        raise ValueError("surfaces dict must be provided and non-empty.")

    # ---- auto-sort surfaces by median depth --------------------------------
    med_depths = {tag: float(np.nanmedian(s)) for tag, s in surfaces.items()}
    surface_order = sorted(med_depths, key=lambda t: med_depths[t])
    surfaces = {tag: surfaces[tag] for tag in surface_order}

    # ---- subsample ----------------------------------------------------------
    if backward_compatible:
        xmin, xmax, ymin, ymax = np.asarray(bounds) - 1
    else:
        xmin, xmax, ymin, ymax = np.asarray(bounds)

    # Use the first surface shape for bounds; all should be the same shape
    first_surface = next(iter(surfaces.values()))
    nx, ny = first_surface.shape
    sampled_x_idx = np.arange(max(xmin - 1, 0), min(xmax + 1, nx - 1) + 1,
                              conformal_jump, dtype=int)
    sampled_y_idx = np.arange(max(ymin - 1, 0), min(ymax + 1, ny - 1) + 1,
                              conformal_jump, dtype=int)

    # ensure within bounds of all surfaces
    for tag, s in surfaces.items():
        sampled_x_idx = sampled_x_idx[(sampled_x_idx >= 0) & (sampled_x_idx < s.shape[0])]
        sampled_y_idx = sampled_y_idx[(sampled_y_idx >= 0) & (sampled_y_idx < s.shape[1])]

    # ---- subsample each surface --------------------------------------------
    subsampled = {}
    for tag, s in surfaces.items():
        subsampled[tag] = s[np.ix_(sampled_x_idx, sampled_y_idx)]

    # ---- diagonal distances (average across all surfaces) ------------------
    all_main = []
    all_skew = []
    for tag in surface_order:
        m, s = calculate_diag_length(sampled_x_idx, sampled_y_idx, subsampled[tag])
        all_main.append(m)
        all_skew.append(s)
    main_diag_dist = float(np.mean(all_main))
    skew_diag_dist = float(np.mean(all_skew))

    # ---- conformal map each surface independently --------------------------
    mapped_surfaces: dict[str, np.ndarray] = {}
    for tag in surface_order:
        if verbose:
            print(f"↳ mapping '{tag}' surface …")
            _t0 = time.time()
        mapped_surfaces[tag] = conformal_map_indep_fixed_diagonals(
            main_diag_dist, skew_diag_dist,
            sampled_x_idx, sampled_y_idx, subsampled[tag],
            n_anchors=n_anchors, backward_compatible=backward_compatible,
        )
        if verbose:
            print(f"    done in {time.time() - _t0:.2f} seconds.")

    # ---- align all surfaces to the first (shallowest) ----------------------
    x_limits = [sampled_x_idx.min(), sampled_x_idx.max()]
    y_limits = [sampled_y_idx.min(), sampled_y_idx.max()]

    ref_tag = surface_order[0]
    for tag in surface_order[1:]:
        mapped_surfaces[tag] = align_mapped_surface(
            surfaces[ref_tag], surfaces[tag],
            mapped_surfaces[ref_tag], mapped_surfaces[tag],
            x_limits, y_limits, conformal_jump, alignment_patch_size,
        )

    # ---- build result dict -------------------------------------------------
    result: dict = {
        "surfaces": surfaces,
        "mapped_surfaces": mapped_surfaces,
        "surface_order": surface_order,
        "main_diag_dist": main_diag_dist,
        "skew_diag_dist": skew_diag_dist,
        "sampled_x_idx": sampled_x_idx,
        "sampled_y_idx": sampled_y_idx,
        "n_anchors": n_anchors,
        "conformal_jump": conformal_jump,
        "meta": {
            "mapped_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "pywarper_version": _PYWARPER_VERSION,
        },
    }

    # ---- legacy keys for backward compat -----------------------------------
    if "on_sac" in surfaces:
        result["on_sac_surface"] = surfaces["on_sac"]
        result["mapped_on"] = mapped_surfaces["on_sac"]
    if "off_sac" in surfaces:
        result["off_sac_surface"] = surfaces["off_sac"]
        result["mapped_off"] = mapped_surfaces["off_sac"]

    return result