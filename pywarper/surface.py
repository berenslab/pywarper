import time
from typing import Optional

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
    print("[Info] scikit-sparse not found. Falling back to scipy.sparse.linalg.spsolve.")


def fit_surface(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    xmax: Optional[int] = None,
    ymax: Optional[int] = None,
    smoothness: int = 1,
    extend: str = "warning",
    interp: str = "triangle",
    regularizer: str = "gradient",
    solver: str = "normal",
    maxiter: Optional[int] = None,
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
    zmesh : np.ndarray
        2D array of interpolated z-values over the fitted surface.
    xmesh : np.ndarray
        2D array of x-coordinates corresponding to zmesh.
    ymesh : np.ndarray
        2D array of y-coordinates corresponding to zmesh.
    """
    if xmax is None:
        xmax = np.max(x).astype(float)
    if ymax is None:
        ymax = np.max(y).astype(float)

    xnodes = np.hstack([np.arange(1., xmax, 3), np.array([xmax])])
    ynodes = np.hstack([np.arange(1., ymax, 3), np.array([ymax])])
    # xnodes = np.arange(0, xmax+skip, skip)
    # ynodes = np.arange(0, ymax+skip, skip)

    gf = GridFit(x, y, z, xnodes, ynodes, 
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
    zgrid = gf.zgrid

    zmesh, xmesh, ymesh = resample_zgrid(
        xnodes, ynodes, zgrid, xmax, ymax
    )

    return zmesh, xmesh, ymesh

def resample_zgrid(
    xnodes: np.ndarray,
    ynodes: np.ndarray,
    zgrid: np.ndarray,
    xMax: int,
    yMax: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Resamples a 2D grid (zgrid) at integer coordinates up to xMax and yMax.
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
    xMax : int
        The maximum x-coordinate (inclusive) for the resampling.
    yMax : int
        The maximum y-coordinate (inclusive) for the resampling.

    Returns
    -------
    vzmesh : np.ndarray
        2D array of shape (xMax, yMax), containing interpolated z-values at
        integer (x, y) positions.
    xi : np.ndarray
        2D array of shape (xMax, yMax), representing the x-coordinates used for
        interpolation.
    yi : np.ndarray
        2D array of shape (xMax, yMax), representing the y-coordinates used for
        interpolation.

    Notes
    -----
    In Python, arrays are typically indexed as (row, column) which maps to
    (y, x) in a 2D sense. This function transposes the meshgrid from
    `np.meshgrid(..., indexing='xy')` to match the MATLAB style of indexing.
    """

    # 0) Check that xMax, yMax are integers.
    #    If not, round to nearest integer.
    xMax = int(xMax)
    yMax = int(yMax)

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
    #    then do xi=xi', yi=yi' => shape (xMax, yMax).
    xi_m, yi_m = np.meshgrid(
        np.arange(1, xMax+1), 
        np.arange(1, yMax+1), 
        indexing='xy'
    )
    xi = xi_m.T  # shape (xMax, yMax)
    yi = yi_m.T  # shape (xMax, yMax)

    # 3) Flatten the coordinate arrays to shape (N, 2) for RGI.
    XYi = np.column_stack((yi.ravel(), xi.ravel()))
    # We must pass (y, x) in that order since RGI is (y-axis, x-axis).

    # 4) Interpolate.
    vmesh_flat = rgi(XYi)  # 1D array, length xMax*yMax

    # 5) Reshape to (xMax, yMax).
    vzmesh = vmesh_flat.reshape((xMax, yMax))

    return vzmesh, xi, yi


def calculate_diag_length(    
    xpos: np.ndarray,
    ypos: np.ndarray,
    VZmesh: np.ndarray
) -> tuple[float, float]:
    """
    Computes the 3D length along the main diagonal and skew diagonal of the
    surface given by VZmesh, where xpos and ypos define the coordinate axes.

    Parameters
    ----------
    xpos : np.ndarray
        1D array of x-coordinates (length M).
    ypos : np.ndarray
        1D array of y-coordinates (length N).
    VZmesh : np.ndarray
        2D array of shape (M, N), representing z-values at the grid points
        formed by xpos and ypos.

    Returns
    -------
    main_diag_dist : float
        The summed 3D distance along the main diagonal of the surface.
    skew_diag_dist : float
        The summed 3D distance along the skew (reverse) diagonal of the surface.

    Notes
    -----
    The function interpolates coordinates along whichever dimension is larger,
    then sums Euclidean distances in 3D for each diagonal path.
    """
    M, N = VZmesh.shape  # M = len(xpos), N = len(ypos)

    # Build interpolators over the regular (xpos,ypos) grid
    interp_x = RegularGridInterpolator((xpos, ypos), 
                                       np.meshgrid(xpos, ypos, indexing='ij')[0],
                                       method='linear')
    interp_y = RegularGridInterpolator((xpos, ypos), 
                                       np.meshgrid(xpos, ypos, indexing='ij')[1],
                                       method='linear')
    interp_z = RegularGridInterpolator((xpos, ypos), VZmesh, 
                                       method='linear')

    main_diag_dist = 0.0
    skew_diag_dist = 0.0

    # if N >= M, we step N times in x, and use the full range of ypos in y
    if N >= M:
        # Build vectors for diagonal queries, matching the MATLAB approach
        x_diag = np.linspace(xpos[0], xpos[-1], N)  # length N
        y_main_diag = np.array(ypos)               # also length N
        y_skew_diag = y_main_diag[::-1]

        # Evaluate on main diagonal
        pts_main = np.column_stack((x_diag, y_main_diag))  # shape (N,2)
        x_knots_main = interp_x(pts_main)
        y_knots_main = interp_y(pts_main)
        z_knots_main = interp_z(pts_main)

        # Evaluate on skew diagonal
        pts_skew = np.column_stack((x_diag, y_skew_diag))
        x_knots_skew = interp_x(pts_skew)
        y_knots_skew = interp_y(pts_skew)
        z_knots_skew = interp_z(pts_skew)

        # Accumulate distances
        for kk in range(N - 1):
            dx_main = x_knots_main[kk] - x_knots_main[kk + 1]
            dy_main = y_knots_main[kk] - y_knots_main[kk + 1]
            dz_main = z_knots_main[kk] - z_knots_main[kk + 1]
            main_diag_dist += np.sqrt(dx_main**2 + dy_main**2 + dz_main**2)

            dx_skew = x_knots_skew[kk] - x_knots_skew[kk + 1]
            dy_skew = y_knots_skew[kk] - y_knots_skew[kk + 1]
            dz_skew = z_knots_skew[kk] - z_knots_skew[kk + 1]
            skew_diag_dist += np.sqrt(dx_skew**2 + dy_skew**2 + dz_skew**2)

    else:
        # M > N
        y_diag = np.linspace(ypos[0], ypos[-1], M)  # length M
        x_main_diag = np.array(xpos)               # also length M
        x_skew_diag = x_main_diag[::-1]

        # Evaluate on main diagonal
        pts_main = np.column_stack((x_main_diag, y_diag))  # shape (M,2)
        x_knots_main = interp_x(pts_main)
        y_knots_main = interp_y(pts_main)
        z_knots_main = interp_z(pts_main)

        # Evaluate on skew diagonal
        pts_skew = np.column_stack((x_skew_diag, y_diag))
        x_knots_skew = interp_x(pts_skew)
        y_knots_skew = interp_y(pts_skew)
        z_knots_skew = interp_z(pts_skew)

        # Accumulate distances
        for kk in range(M - 1):
            dx_main = x_knots_main[kk] - x_knots_main[kk + 1]
            dy_main = y_knots_main[kk] - y_knots_main[kk + 1]
            dz_main = z_knots_main[kk] - z_knots_main[kk + 1]
            main_diag_dist += np.sqrt(dx_main**2 + dy_main**2 + dz_main**2)

            dx_skew = x_knots_skew[kk] - x_knots_skew[kk + 1]
            dy_skew = y_knots_skew[kk] - y_knots_skew[kk + 1]
            dz_skew = z_knots_skew[kk] - z_knots_skew[kk + 1]
            skew_diag_dist += np.sqrt(dx_skew**2 + dy_skew**2 + dz_skew**2)

    return main_diag_dist, skew_diag_dist


def assign_local_coordinates(triangle: np.ndarray) -> tuple[complex, complex, complex, float]:
    """
    Assigns local complex coordinates (w1, w2, w3) to the three vertices of a
    triangle in 3D space, used for conformal mapping calculations.

    Parameters
    ----------
    triangle : np.ndarray
        Array of shape (3, 3), where each row corresponds to a vertex in (x, y, z).
        The three vertices define one triangular face.

    Returns
    -------
    w1 : complex
        Complex representation of the local coordinate for vertex 1.
    w2 : complex
        Complex representation of the local coordinate for vertex 2.
    w3 : complex
        Complex representation of the local coordinate for vertex 3.
    zeta : float
        A normalization factor based on the triangle’s geometry, typically used
        to scale further computations (e.g., for quasi-conformal maps).

    Notes
    -----
    Each vertex is measured relative to the first vertex, establishing a local
    coordinate system. The calculations ensure an appropriate scale and orientation
    for the subsequent mapping steps.
    """    
    d12 = np.linalg.norm(triangle[0] - triangle[1])
    d13 = np.linalg.norm(triangle[0] - triangle[2])
    d23 = np.linalg.norm(triangle[1] - triangle[2])
    y3 = ((-d12)**2 + d13**2 - d23**2) / (2 * -d12)
    x3 = np.sqrt(np.maximum(0, d13**2 - y3**2))
    w2 = -x3 - 1j * y3
    w1 = x3 + 1j * (y3 + d12)
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
    vertexCount:int = M * N
    triangleCount = (2 * M - 2) * (N - 1)

    # Efficient triangulation construction
    col1 = np.kron([1, 1], np.arange(M - 1)).reshape(-1, 1)
    temp1 = np.kron([1, M + 1], np.ones(M - 1)).reshape(-1, 1)
    temp2 = np.kron([M + 1, M], np.ones(M - 1)).reshape(-1, 1)
    one_column = np.hstack([col1, col1 + temp1, col1 + temp2]).astype(int)

    # Corrected broadcasting logic
    one_column_tiled = np.tile(one_column, (N - 1, 1))
    offsets = (np.repeat(np.arange(N - 1), 2 * M - 2) * M).reshape(-1, 1)
    triangulation = (one_column_tiled + offsets).astype(int)

    # Precompute arrays
    row_indices = np.repeat(np.arange(triangleCount), 3)
    col_indices = triangulation.flatten()

    Mreal_data = np.zeros(triangleCount * 3)
    Mimag_data = np.zeros(triangleCount * 3)

    # Loop once to fill the data
    for tri_idx in range(triangleCount):
        vertices = triangulation[tri_idx]
        coords = np.column_stack((
            xpos_new[vertices % M],
            ypos_new[vertices // M],
            VZmesh[vertices % M, vertices // M]
        ))
        w1, w2, w3, zeta = assign_local_coordinates(coords)
        denom = np.sqrt(zeta / 2)
        ws = [w1, w2, w3]

        idx = slice(tri_idx * 3, (tri_idx + 1) * 3)
        Mreal_data[idx] = np.real(ws) / denom
        Mimag_data[idx] = np.imag(ws) / denom

    # Build sparse matrices in one step
    Mreal_csr = coo_matrix((Mreal_data, (row_indices, col_indices)), shape=(triangleCount, vertexCount)).tocsr()
    Mimag_csr = coo_matrix((Mimag_data, (row_indices, col_indices)), shape=(triangleCount, vertexCount)).tocsr()

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


def warp_surface(
    thisvzminmesh: np.ndarray,
    thisvzmaxmesh: np.ndarray,
    arbor_boundaries: np.ndarray,
    conformal_jump: int = 1,
    verbose: bool = False
) -> dict:
    """
    Generates a conformal warp of two Starburst Amacrine Cell (SAC) surfaces 
    (min and max) and aligns them. This function is a higher-level wrapper 
    that uses diagonal distance calculations, conformal mapping, and alignment.

    Parameters
    ----------
    thisvzminmesh : np.ndarray
        2D array of shape (X, Y), representing the “minimum” / ON SAC surface.
    thisvzmaxmesh : np.ndarray
        2D array of shape (X, Y), representing the “maximum” / OFF SAC surface.
    arbor_boundaries : tuple of int
        (xmin, xmax, ymin, ymax) specifying the region of interest 
        over which to warp the surfaces.
    conformal_jump : int, default=1
        Subsampling step for reducing the resolution during mapping 
        (e.g., conformal_jump=2 uses every other pixel).
    verbose : bool, default=False
        Whether to print timing and debug information.

    Returns
    -------
    dict
        Dictionary containing:
        - "mapped_min_positions": np.ndarray
            Mapped coordinates for the min surface.
        - "mapped_max_positions": np.ndarray
            Mapped coordinates for the max surface (aligned to min).
        - "main_diag_dist": float
            Average main diagonal distance used during conformal mapping.
        - "skew_diag_dist": float
            Average skew diagonal distance used during conformal mapping.
        - "thisx": np.ndarray
            Subsampled x indices used for mapping.
        - "thisy": np.ndarray
            Subsampled y indices used for mapping.
        - "thisVZminmesh": np.ndarray
            Original min surface data used in mapping (subsampled).
        - "thisVZmaxmesh": np.ndarray
            Original max surface data used in mapping (subsampled).

    Notes
    -----
    This routine is tailored to Starburst Amacrine Cell layers but can be 
    generalized to other layered surfaces. It:
    1. Subsamples the surfaces by conformal_jump.
    2. Calculates average diagonal distances from each surface.
    3. Performs two independent conformal mappings (min and max).
    4. Aligns the “max” mapping to “min” based on local gradient differences.
    5. Returns a dictionary of intermediate results for further inspection.
    """

    xmin, xmax, ymin, ymax = arbor_boundaries

    thisx = np.round(np.arange(np.maximum(xmin-2, 0), np.minimum(xmax+1, thisvzmaxmesh.shape[0]), conformal_jump)).astype(int)
    thisy = np.round(np.arange(np.maximum(ymin-2, 0), np.minimum(ymax+1, thisvzmaxmesh.shape[1]), conformal_jump)).astype(int)

    thisminmesh = thisvzminmesh[thisx[:, None], thisy]
    thismaxmesh = thisvzmaxmesh[thisx[:, None], thisy]
    # calculate the traveling distances on the diagonals of the two SAC surfaces 
    start_time = time.time()
    main_diag_dist_min, skew_diag_dist_min = calculate_diag_length(thisx, thisy, thisminmesh)
    main_diag_dist_max, skew_diag_dist_max = calculate_diag_length(thisx, thisy, thismaxmesh)

    main_diag_dist = np.mean([main_diag_dist_min, main_diag_dist_max])
    skew_diag_dist = np.mean([skew_diag_dist_min, skew_diag_dist_max])

    # quasi-conformally map individual SAC surfaces to planes
    if verbose:
        print("Mapping min position (On SAC layer)...")    
        start_time = time.time()
    mapped_min_positions = conformal_map_indep_fixed_diagonals(
        main_diag_dist, skew_diag_dist, thisx, thisy, thisminmesh
    )
    if verbose:
        print(f"Mapping min position completed in {time.time() - start_time:.2f} seconds.")

    if verbose:
        print("Mapping max position (Off SAC layer)...")
        start_time = time.time()
    mapped_max_positions = conformal_map_indep_fixed_diagonals(
        main_diag_dist, skew_diag_dist, thisx, thisy, thismaxmesh
    )
    if verbose:
        print(f"Mapping max position completed in {time.time() - start_time:.2f} seconds.")

    xborders = [thisx.min(), thisx.max()]
    yborders = [thisy.min(), thisy.max()]

    # align the mapped max surface to the mapped min surface
    mapped_max_positions = align_mapped_surface(
        thisvzminmesh, thisvzmaxmesh,
        mapped_min_positions, mapped_max_positions,
        xborders, yborders, conformal_jump
    )

    return {
        "mapped_min_positions": mapped_min_positions,
        "mapped_max_positions": mapped_max_positions,
        "main_diag_dist": main_diag_dist,
        "skew_diag_dist": skew_diag_dist,
        "thisx": thisx,
        "thisy": thisy,
        "thisVZminmesh": thisvzminmesh,
        "thisVZmaxmesh": thisvzmaxmesh,
    }