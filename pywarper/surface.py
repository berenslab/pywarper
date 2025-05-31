"""
pywarper.surface
================
Numerical utilities for **flattening the Starburst Amacrine Cell (SAC) layers** of a retina.

Given two depth maps—one for the ON SAC band and one for the OFF SAC band—this module performs

1. **Surface fitting** (`fit_sac_surface`) – smooths scattered ChAT-band samples or arbor node coordinates
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


def fit_sac_surface(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    xmax: int | float | None = None,
    ymax: int | float | None = None,
    smoothness: int = 1,
    extend: str = "warning",
    interp: str = "triangle",
    regularizer: str = "gradient",
    solver: str = "normal",
    maxiter: int | None = None,
    autoscale: str = "on",
    xscale: float = 1.0,
    yscale: float = 1.0,
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

    xnodes = np.hstack([np.arange(1., xmax, 3.), np.array([xmax])])
    ynodes = np.hstack([np.arange(1., ymax, 3.), np.array([ymax])])

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
        xnodes, ynodes, zgrid, xmax, ymax
    )

    return zmesh, xmesh, ymesh

def resample_zgrid(
    xnodes: np.ndarray,
    ynodes: np.ndarray,
    zgrid: np.ndarray,
    xmax: int | float,
    ymax: int | float,
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
    xi_m, yi_m = np.meshgrid(
        np.arange(1, xmax+1), 
        np.arange(1, ymax+1), 
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
    VZmesh: np.ndarray
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
    xpos_new = xpos + 1
    ypos_new = ypos + 1
    vertexCount = M * N
    triangleCount = (2 * M - 2) * (N - 1)

    # --- build triangulation -------------------------------------------------
    col1 = np.kron([1, 1], np.arange(M - 1))
    temp1 = np.kron([1, M + 1], np.ones(M - 1))
    temp2 = np.kron([M + 1, M], np.ones(M - 1))
    one_column = np.stack([col1, col1 + temp1, col1 + temp2], axis=1).astype(int)

    triangulation = np.tile(one_column, (N - 1, 1))
    offsets = np.repeat(np.arange(N - 1), 2 * M - 2)[:, None] * M
    triangulation += offsets
    # triangulation.shape == (triangleCount, 3)

    # --- vectorised local-coordinate calculation ----------------------------
    rows = triangulation % M
    cols = triangulation // M

    triangles_xyz = np.empty((triangleCount, 3, 3), dtype=np.float64)
    triangles_xyz[:, :, 0] = xpos_new[rows]
    triangles_xyz[:, :, 1] = ypos_new[cols]
    triangles_xyz[:, :, 2] = VZmesh[rows, cols]

    w1, w2, w3, zeta = assign_local_coordinates(triangles_xyz)
    denom = np.sqrt(zeta / 2.0)

    ws_real = np.column_stack([np.real(w1), np.real(w2), np.real(w3)]) / denom[:, None]
    ws_imag = np.column_stack([np.imag(w1), np.imag(w2), np.imag(w3)]) / denom[:, None]

    row_indices = np.repeat(np.arange(triangleCount), 3)
    col_indices = triangulation.ravel()

    Mreal_csr = coo_matrix((ws_real.ravel(), (row_indices, col_indices)),
                           shape=(triangleCount, vertexCount)).tocsr()
    Mimag_csr = coo_matrix((ws_imag.ravel(), (row_indices, col_indices)),
                           shape=(triangleCount, vertexCount)).tocsr()

    def solve_mapping(fixed_pts, fixed_vals, free_pts):
        A = vstack([
            hstack([Mreal_csr[:, free_pts], -Mimag_csr[:, free_pts]]),
            hstack([Mimag_csr[:, free_pts], Mreal_csr[:, free_pts]])
        ])

        b_real = Mreal_csr[:, fixed_pts] @ fixed_vals[:, 0] - Mimag_csr[:, fixed_pts] @ fixed_vals[:, 1]
        b_imag = Mimag_csr[:, fixed_pts] @ fixed_vals[:, 0] + Mreal_csr[:, fixed_pts] @ fixed_vals[:, 1]
        b = -np.concatenate([b_real, b_imag])

        AtA = (A.T @ A).tocsc()
        Atb = A.T @ b

        if HAS_CHOLMOD:
            sol = cholesky(AtA)(Atb)
        else:
            sol = spsolve(AtA, Atb)

        num_free = len(free_pts)
        mapped = np.zeros((vertexCount, 2))
        mapped[fixed_pts] = fixed_vals
        mapped[free_pts, 0] = sol[:num_free]
        mapped[free_pts, 1] = sol[num_free:]
        return mapped

    diag_scale = M / np.sqrt(M**2 + N**2)
    main_diag_fixed_pts = [0, vertexCount - 1]
    main_diag_fixed_vals = np.array([
        [xpos_new[0], ypos_new[0]],
        [xpos_new[0] + mainDiagDist * diag_scale, ypos_new[0] + mainDiagDist * diag_scale * N / M]
    ])
    main_diag_free_pts = np.setdiff1d(np.arange(vertexCount), main_diag_fixed_pts)
    mapped_main = solve_mapping(main_diag_fixed_pts, main_diag_fixed_vals, main_diag_free_pts)

    skew_diag_fixed_pts = [M - 1, vertexCount - M]
    skew_diag_fixed_vals = np.array([
        [xpos_new[0] + skewDiagDist * diag_scale, ypos_new[0]],
        [xpos_new[0], ypos_new[0] + skewDiagDist * diag_scale * N / M]
    ])
    skew_diag_free_pts = np.setdiff1d(np.arange(vertexCount), skew_diag_fixed_pts)
    mapped_skew = solve_mapping(skew_diag_fixed_pts, skew_diag_fixed_vals, skew_diag_free_pts)

    # Final averaged result
    mappedPositions = (mapped_main + mapped_skew) / 2
    return mappedPositions

def align_mapped_surface(    
    thisVZminmesh: np.ndarray,
    thisVZmaxmesh: np.ndarray,
    mappedMinPositions: np.ndarray,
    mappedMaxPositions: np.ndarray,
    xborders: list[int],
    yborders: list[int],
    conformal_jump: int = 1,
    patch_size: int = 21
) -> np.ndarray:
    """
    Shifts the second mapped surface (mappedMaxPositions) so that its local
    gradients align best with those of the first (mappedMinPositions).

    Parameters
    ----------
    thisVZminmesh : np.ndarray
        2D array of shape (X, Y), representing the first (minimum) surface.
    thisVZmaxmesh : np.ndarray
        2D array of shape (X, Y), representing the second (maximum) surface.
    mappedMinPositions : np.ndarray
        2D array of shape (X*Y, 2), the conformally mapped coordinates 
        corresponding to the min surface.
    mappedMaxPositions : np.ndarray
        2D array of shape (X*Y, 2), the conformally mapped coordinates 
        corresponding to the max surface.
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
    mappedMaxPositions : np.ndarray
        Updated 2D array of shape (X*Y, 2) for the max surface, 
        after alignment to the min surface.

    Notes
    -----
    This step finds an offset (shift in x and y) that best aligns local slope
    features from the two surfaces, by comparing gradients in a restricted region 
    and choosing the position with minimal combined gradient magnitude.
    """
    patch_size = int(np.ceil(patch_size / conformal_jump))

    # Pad surfaces to preserve shape after differencing
    pad_val_min = 10 * np.max(thisVZminmesh)
    pad_val_max = 10 * np.max(thisVZmaxmesh)

    VZminmesh_padded = np.pad(thisVZminmesh, ((0, 1), (0, 1)), constant_values=pad_val_min)
    VZmaxmesh_padded = np.pad(thisVZmaxmesh, ((0, 1), (0, 1)), constant_values=pad_val_max)

    # Gradient differences (dx + i*dy)
    dmin_dx = np.diff(VZminmesh_padded, axis=0)[:, :-1]
    dmin_dy = np.diff(VZminmesh_padded, axis=1)[:-1, :]
    dMinSurface = np.abs(dmin_dx + 1j * dmin_dy)

    dmax_dx = np.diff(VZmaxmesh_padded, axis=0)[:, :-1]
    dmax_dy = np.diff(VZmaxmesh_padded, axis=1)[:-1, :]
    dMaxSurface = np.abs(dmax_dx + 1j * dmax_dy)

    # Region of interest
    x1, x2 = xborders
    y1, y2 = yborders

    dMinSurface_roi = dMinSurface[x1:x2+1:conformal_jump, y1:y2+1:conformal_jump]
    dMaxSurface_roi = dMaxSurface[x1:x2+1:conformal_jump, y1:y2+1:conformal_jump]

    combined_slope = dMinSurface_roi + dMaxSurface_roi

    # Patch cost = sum of local gradients over patch
    kernel = np.ones((patch_size, patch_size))
    patch_costs = convolve2d(combined_slope, kernel, mode='valid')

    # # Map back to flattened index in 2D mesh
    # row, col are 0-based from Python
    # Convert them to 1-based to mimic MATLAB
    min_index = np.argmin(patch_costs)
    row0, col0 = np.unravel_index(min_index, patch_costs.shape)
    # (row0, col0) is 0-based, which correspond to x,y in MATLAB if the array shape is (num_x, num_y).

    # Now replicate the step:
    #   row = round(row + (patchSize - 1)/2)
    #   col = round(col + (patchSize - 1)/2)
    row_center_0b = int(round(row0 + (patch_size - 1) / 2))
    col_center_0b = int(round(col0 + (patch_size - 1) / 2))

    # Now we want the same linear index that MATLAB would get from
    # sub2ind([num_x, num_y], row_center, col_center),
    # except sub2ind is 1-based. In 0-based form, that is:
    #   linearInd = col_center_0b * num_x + row_center_0b
    flat_index = col_center_0b * dMinSurface_roi.shape[0] + row_center_0b

    # Then do the shift
    shift_x = mappedMaxPositions[flat_index, 0] - mappedMinPositions[flat_index, 0]
    shift_y = mappedMaxPositions[flat_index, 1] - mappedMinPositions[flat_index, 1]

    mappedMaxPositions[:, 0] -= shift_x
    mappedMaxPositions[:, 1] -= shift_y

    return mappedMaxPositions


def build_mapping(
    on_sac_surface: np.ndarray, # original `thisVZminmesh`  (ON‑Starburst layer)
    off_sac_surface: np.ndarray, # original `thisVZmaxmesh`  (OFF‑Starburst layer)
    arbor_xy_bounds: np.ndarray | tuple[int, int, int, int],  # original `arborBoundaries`
    conformal_jump: int = 1, # original `conformalJump`
    *,
    verbose: bool = False
) -> dict:
    """
    Create a 2D conformal map that **flattens** the ON‑ and OFF‑Starburst Amacrine Cell (SAC) 
    layers onto a common plane so their geometry can later be imposed on retinal arbors.

    This is a refactored port of MATLAB **`calcWarpedSACsurfaces`**.  
    The mathematics and return values are preserved exactly; only names and documentation are clearer.

    Workflow
    --------
    1. **Subsample** both SAC height‑fields within `arbor_xy_bounds` at every `conformal_jump` pixels.
    2. Measure the true 3D lengths of the main and skew diagonals on each subsampled surface.
    3. **Conformally map** the ON and OFF surfaces independently so those diagonals become straight with the measured lengths.
    4. **Align** the OFF map to the ON map by finding the x/y shift that minimises local slope mismatches.
    5. Return the two mapped coordinate sets plus diagnostic metadata.

    Parameters
    ----------
    on_sac_surface : np.ndarray
        Height map of the ON ("minimum") SAC layer, shape *(X, Y).*  (Formerly `thisVZminmesh`).
    off_sac_surface : np.ndarray
        Height map of the OFF ("maximum") SAC layer, shape *(X, Y).*  (Formerly `thisVZmaxmesh`).
    arbor_xy_bounds : tuple[int, int, int, int] | np.ndarray
        *(xmin, xmax, ymin, ymax)* bounds of the region that actually contains the arbor.  (Formerly `arborBoundaries`).
    conformal_jump : int, default 1
        Sub‑sampling stride when reading the SAC surfaces.  A larger value speeds things up at the cost of resolution.  (Formerly `conformalJump`).
    verbose : bool, default False
        If *True*, print timing information.

    Returns
    -------
    dict
        ``mapped_on``
            *(N × 2)* xy coordinates of each sampled vertex on the flattened ON surface.
        ``mapped_off``
            Same for the OFF surface **after alignment**.
        ``main_diag_dist`` / ``skew_diag_dist``
            Mean physical lengths of the main and skew diagonals used as conformal constraints.
        ``sampled_x_idx`` / ``sampled_y_idx``
            The x‑ and y‑indices that were actually sampled and mapped.
        ``on_sac_surface`` / ``off_sac_surface``
            The original (subsampled) height fields kept for debugging or visualisation.
    """

    xmin, xmax, ymin, ymax = np.asarray(arbor_xy_bounds) - 1 # Convert to 0-based indexing

    nx, ny = off_sac_surface.shape
    sampled_x_idx = np.arange(max(xmin - 1, 0),  min(xmax + 1, nx - 1) + 1,
                    conformal_jump, dtype=int)
    sampled_y_idx = np.arange(max(ymin - 1, 0),  min(ymax + 1, ny - 1) + 1,
                    conformal_jump, dtype=int)

    # probably not necessary but better ensure that sampled_x_idx, sampled_y_idx are within bounds
    sampled_x_idx = sampled_x_idx[(sampled_x_idx >= 0) & (sampled_x_idx < on_sac_surface.shape[0])]
    sampled_y_idx = sampled_y_idx[(sampled_y_idx >= 0) & (sampled_y_idx < on_sac_surface.shape[1])]

    on_subsampled  =  on_sac_surface[np.ix_(sampled_x_idx, sampled_y_idx)]
    off_subsampled = off_sac_surface[np.ix_(sampled_x_idx, sampled_y_idx)]

    # calculate the traveling distances on the diagonals of the two SAC surfaces
    start_time = time.time()
    main_diag_dist_on, skew_diag_dist_on = calculate_diag_length(sampled_x_idx, sampled_y_idx, on_subsampled)
    main_diag_dist_off, skew_diag_dist_off = calculate_diag_length(sampled_x_idx, sampled_y_idx, off_subsampled)

    main_diag_dist = np.mean([main_diag_dist_on, main_diag_dist_off])
    skew_diag_dist = np.mean([skew_diag_dist_on, skew_diag_dist_off])

    # quasi-conformally map individual SAC surfaces to planes
    if verbose:
        print("Mapping ON (min) surface …")    
        start_time = time.time()
    mapped_on = conformal_map_indep_fixed_diagonals(
        float(main_diag_dist), float(skew_diag_dist), sampled_x_idx, sampled_y_idx, on_subsampled
    )
    if verbose:
        print(f"Completed in {time.time() - start_time:.2f} seconds.")

    if verbose:
        print("Mapping OFF (max) surface …")
        start_time = time.time()
    mapped_off = conformal_map_indep_fixed_diagonals(
        float(main_diag_dist), float(skew_diag_dist), sampled_x_idx, sampled_y_idx, off_subsampled
    )
    if verbose:
        print(f"Completed in {time.time() - start_time:.2f} seconds.")

    x_limits = [sampled_x_idx.min(), sampled_x_idx.max()]  # original `xborders`
    y_limits = [sampled_y_idx.min(), sampled_y_idx.max()]  # original `yborders`

    # Align OFF map to ON map (patch matching)
    map_off_aligned = align_mapped_surface(
        on_sac_surface, off_sac_surface,
        mapped_on, mapped_off,
        x_limits, y_limits, conformal_jump
    )

    return {
        "mapped_on": mapped_on,  # formerly `mappedMinPositions`
        "mapped_off": map_off_aligned, # formerly `mappedMaxPositions`
        "main_diag_dist": main_diag_dist, # same as MATLAB `mainDiagDist`
        "skew_diag_dist": skew_diag_dist, # same as MATLAB `skewDiagDist`
        "sampled_x_idx": sampled_x_idx, # formerly `thisx`
        "sampled_y_idx": sampled_y_idx, # formerly `thisy`
        "on_sac_surface": on_sac_surface, # formerly `thisVZminmesh`
        "off_sac_surface": off_sac_surface, # formerly `thisVZmaxmesh`
    }