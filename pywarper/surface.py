import time

import numpy as np
from pygridfit import GridFit
from scipy.interpolate import RegularGridInterpolator
from scipy.signal import convolve2d
from scipy.sparse import csr_matrix, hstack, lil_matrix, vstack
from scipy.sparse.linalg import spsolve


def fit_surface(x: np.ndarray, y: np.ndarray, z: np.ndarray, 
                xmax: int|None = None, ymax: int|None = None,
                smoothness: int = 1,
                # skip:int = 3
                ):

    if xmax is None:
        xmax = np.max(x).astype(float)
    if ymax is None:
        ymax = np.max(y).astype(float)

    xnodes = np.hstack([np.arange(1., xmax, 3), np.array([xmax])])
    ynodes = np.hstack([np.arange(1., ymax, 3), np.array([ymax])])
    # xnodes = np.arange(0, xmax+skip, skip)
    # ynodes = np.arange(0, ymax+skip, skip)

    gf = GridFit(x, y, z, xnodes, ynodes, smoothness=smoothness)
    gf.fit()
    zgrid = gf.zgrid

    zmesh, xmesh, ymesh = resample_zgrid(
        xnodes, ynodes, zgrid, xmax, ymax
    )

    return zmesh, xmesh, ymesh

def resample_zgrid(
    xnodes,    # 1D array of x-coordinates, length nx
    ynodes,    # 1D array of y-coordinates, length ny
    zgrid,     # shape (ny, nx), as if from np.meshgrid(xnodes, ynodes, indexing='xy')
    xMax, yMax
):
    """
    Replicate something like:
      [xi, yi] = meshgrid(1:xMax, 1:yMax); 
      xi=xi'; yi=yi';
      vzmesh=interp2(xgrid,ygrid,zgrid, xi, yi, '*linear', fillval)
    using RegularGridInterpolator with method='linear'.
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


def calculate_diag_length(xpos, ypos, VZmesh):
    """
    A closer match to the original MATLAB code using RegularGridInterpolator
    for xKnots, yKnots (instead of RectBivariateSpline) 
    and for zKnots (instead of griddata).
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


def assign_local_coordinates(triangle):
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

def conformal_map_indep_fixed_diagonals(mainDiagDist, skewDiagDist, xpos, ypos, VZmesh):
    M, N = VZmesh.shape
    xpos_new = xpos + 1
    ypos_new = ypos + 1
    triangleCount = (2*M - 2)*(N - 1)
    vertexCount = M * N

    # Build triangulation
    col1 = np.kron([1, 1], np.arange(1, M)).reshape(-1, 1)
    temp1 = np.kron([1, M+1], np.ones(M-1)).reshape(-1, 1)
    temp2 = np.kron([M+1, M], np.ones(M-1)).reshape(-1, 1)
    one_column = np.hstack([col1, col1 + temp1, col1 + temp2]) - 1

    triangulation = np.zeros((triangleCount, 3), dtype=int)
    for kk in range(N - 1):
        offset = kk * M
        start = kk * (2*M - 2)
        end = start + (2*M - 2)
        triangulation[start:end] = one_column + offset

    # Sparse matrices
    Mreal = lil_matrix((triangleCount, vertexCount))
    Mimag = lil_matrix((triangleCount, vertexCount))

    # Assign local coordinates and fill sparse matrices
    for tri_idx in range(triangleCount):
        triangle = []
        for vertex in triangulation[tri_idx]:
            xind = vertex % M
            yind = vertex // M
            triangle.append([xpos_new[xind], ypos_new[yind], VZmesh[xind, yind]])
        triangle = np.array(triangle)
        w1, w2, w3, zeta = assign_local_coordinates(triangle)
        denom = np.sqrt(zeta / 2)
        ws = [w1, w2, w3]
        for j, vertex in enumerate(triangulation[tri_idx]):
            Mreal[tri_idx, vertex] = np.real(ws[j]) / denom
            Mimag[tri_idx, vertex] = np.imag(ws[j]) / denom

    # Fix points for main diagonal
    mainDiagXdist = mainDiagDist * M / np.sqrt(M**2 + N**2)
    mainDiagYdist = mainDiagXdist * N / M

    fixed = [0, vertexCount - 1]
    free = list(range(1, vertexCount - 1))
    Mreal_csr = Mreal.tocsr()
    Mimag_csr = Mimag.tocsr()

    A_main = vstack([
        hstack([Mreal_csr[:, free], -Mimag_csr[:, free]]),
        hstack([Mimag_csr[:, free],  Mreal_csr[:, free]])
    ])

    fixed_vec = np.array([[xpos_new[0]], [xpos_new[0] + mainDiagXdist], [ypos_new[0]], [ypos_new[0] + mainDiagYdist]])

    xr = Mreal_csr[:, fixed] @ fixed_vec[:2]
    yr = Mimag_csr[:, fixed] @ fixed_vec[2:]
    xi = Mimag_csr[:, fixed] @ fixed_vec[:2]
    yi = Mreal_csr[:, fixed] @ fixed_vec[2:]

    top = csr_matrix(xr - yr)
    bottom = csr_matrix(xi + yi)

    b_main = -vstack([top, bottom]).toarray().ravel()
    
    sol_main = spsolve(A_main.T @ A_main, A_main.T @ b_main)
    
    mapped_main = np.zeros((vertexCount, 2))
    mapped_main[0] = [xpos_new[0], ypos_new[0]]
    mapped_main[1:-1, 0] = sol_main[:len(sol_main)//2]
    mapped_main[1:-1, 1] = sol_main[len(sol_main)//2:]
    mapped_main[-1] = [xpos_new[0] + mainDiagXdist, ypos_new[0] + mainDiagYdist]

    # Fix points for skew diagonal
    skewDiagXdist = skewDiagDist * M / np.sqrt(M**2 + N**2)
    skewDiagYdist = skewDiagXdist * N / M

    # Match MATLAB's freeVar: [1:M-1, M+1:M*N-M, M*N-M+2:M*N] (0-based)
    freeVar = np.concatenate([
        np.arange(0, M - 1),
        np.arange(M, vertexCount - M),
        np.arange(vertexCount - M + 1, vertexCount)
    ])

    fixed_skew = [M - 1, vertexCount - M]

    A_skew = vstack([
        hstack([Mreal_csr[:, freeVar], -Mimag_csr[:, freeVar]]),
        hstack([Mimag_csr[:, freeVar],  Mreal_csr[:, freeVar]])
    ])

    fixed_vec_skew = np.array([[xpos_new[0] + skewDiagXdist], [xpos_new[0]], [ypos_new[0]], [ypos_new[0] + skewDiagYdist]])

    xr_skew = Mreal_csr[:, fixed_skew] @ fixed_vec_skew[:2]
    yr_skew = Mimag_csr[:, fixed_skew] @ fixed_vec_skew[2:]
    xi_skew = Mimag_csr[:, fixed_skew] @ fixed_vec_skew[:2]
    yi_skew = Mreal_csr[:, fixed_skew] @ fixed_vec_skew[2:]

    top_skew = csr_matrix(xr_skew - yr_skew)
    bottom_skew = csr_matrix(xi_skew + yi_skew)

    b_skew = -vstack([top_skew, bottom_skew]).toarray().ravel()

    # Solve using direct least squares
    sol_skew = spsolve(A_skew.T @ A_skew, A_skew.T @ b_skew)

    # Reassemble mapped_skew like in MATLAB
    mapped_skew = np.zeros((vertexCount, 2))

    # Get total free variable count
    n_free = len(freeVar)

    # X part
    x_part = sol_skew[:n_free]

    mapped_skew[:M - 1, 0] = x_part[:M - 1]
    mapped_skew[M - 1, 0] = xpos_new[0] + skewDiagXdist
    mapped_skew[M:vertexCount - M, 0] = x_part[M - 1:n_free - (M - 1)]
    mapped_skew[vertexCount - M, 0] = xpos_new[0]
    mapped_skew[vertexCount - M + 1:, 0] = x_part[n_free - (M - 1):]

    # Y part
    y_part = sol_skew[n_free:]
    mapped_skew[:M - 1, 1] = y_part[:M - 1]
    mapped_skew[M - 1, 1] = ypos_new[0]
    mapped_skew[M:vertexCount - M, 1] = y_part[M - 1:n_free - (M - 1)]
    mapped_skew[vertexCount - M, 1] = ypos_new[0] + skewDiagYdist
    mapped_skew[vertexCount - M + 1:, 1] = y_part[n_free - (M - 1):]

    # Average both results
    mappedPositions = (mapped_main + mapped_skew) / 2
    return mappedPositions


def align_mapped_surface(thisVZminmesh, thisVZmaxmesh,
                         mappedMinPositions, mappedMaxPositions,
                         xborders, yborders, conformal_jump=1, patch_size=21):
    """
    Aligns mappedMaxPositions to mappedMinPositions by minimizing difference in slope
    between VZminmesh and VZmaxmesh over a local region.
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


def warp_surface(thisvzminmesh, thisvzmaxmesh, arbor_boundaries, conformal_jump = 1, verbose=False):

    if verbose:
        print("Warping surface...")
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