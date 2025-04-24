import time
from typing import Union

import numpy as np
from numpy.linalg import lstsq
from scipy.special import i0


def local_ls_registration(
    nodes: np.ndarray,
    top_input_pos: np.ndarray,
    bot_input_pos: np.ndarray,
    top_output_pos: np.ndarray,
    bot_output_pos: np.ndarray,
    window: float = 5.0,
    max_order: int = 2
) -> np.ndarray:
    """
    Applies a local least-squares polynomial transformation to each node based on nearby
    surface correspondences from the top and bottom “bands” (layers).

    Parameters
    ----------
    nodes : np.ndarray
        (N, 3) array of [x, y, z] positions to be transformed.
    top_input_pos : np.ndarray
        (M, 3) array of [x, y, z] coordinates for the top band (original space).
    bot_input_pos : np.ndarray
        (M, 3) array of [x, y, z] coordinates for the bottom band (original space).
    top_output_pos : np.ndarray
        (M, 3) array of mapped [x, y, z] coordinates for the top band (flattened space).
    bot_output_pos : np.ndarray
        (M, 3) array of mapped [x, y, z] coordinates for the bottom band (flattened space).
    window : float, default=5.0
        Neighborhood radius (in pixels/units) for local polynomial fitting.
    max_order : int, default=2
        Maximum total polynomial order for the local transformation model.
        For example, 2 indicates quadratic terms.

    Returns
    -------
    np.ndarray
        (N, 3) array of transformed [x, y, z] positions, the result of applying
        the local least-squares fits to each node.

    Notes
    -----
    The function constructs a polynomial basis (constant, linear, up to max_order
    in both x and y), plus terms modulated by z. A least-squares system is solved
    locally for each node, using points in the top and bottom surfaces within
    the specified window. If insufficient points are found, the node remains
    unchanged.
    """
    transformed_nodes = np.zeros_like(nodes)
    
    # Combine top/bottom input positions with outputs once
    top_in_xy = top_input_pos[:, :2]
    bot_in_xy = bot_input_pos[:, :2]

    for k, (x, y, z) in enumerate(nodes):
        lx, ux = x - window, x + window
        ly, uy = y - window, y + window

        # Find indices once (reuse masks)
        top_mask = ((top_in_xy[:, 0] >= lx) & (top_in_xy[:, 0] <= ux) &
                    (top_in_xy[:, 1] >= ly) & (top_in_xy[:, 1] <= uy))
        bot_mask = ((bot_in_xy[:, 0] >= lx) & (bot_in_xy[:, 0] <= ux) &
                    (bot_in_xy[:, 1] >= ly) & (bot_in_xy[:, 1] <= uy))

        in_top, out_top = top_input_pos[top_mask], top_output_pos[top_mask]
        in_bot, out_bot = bot_input_pos[bot_mask], bot_output_pos[bot_mask]

        this_in = np.vstack([in_top, in_bot])
        this_out = np.vstack([out_top, out_bot])

        if len(this_in) < 12:
            transformed_nodes[k] = nodes[k]
            continue

        # Center coordinates (in-place avoided)
        x_shift, y_shift = np.mean(this_in[:, :2], axis=0)
        this_in_centered = this_in.copy()
        this_out_centered = this_out.copy()

        this_in_centered[:, 0] -= x_shift
        this_out_centered[:, 0] -= x_shift
        this_in_centered[:, 1] -= y_shift
        this_out_centered[:, 1] -= y_shift

        # Efficient polynomial basis creation
        xin, yin, zin = this_in_centered.T
        basis_cols = []

        # Constant term
        basis_cols.append(np.ones_like(xin))

        # Linear terms
        basis_cols.extend([xin, yin])

        # Higher-order terms
        for order in range(2, max_order + 1):
            for ox in range(order + 1):
                oy = order - ox
                basis_cols.append((xin ** ox) * (yin ** oy))

        # Stack basis columns once
        base_terms = np.vstack(basis_cols).T  # shape: (n_points, n_terms)

        # Z-modulated terms
        z_modulated = base_terms * zin[:, np.newaxis]

        # Combined X matrix
        X = np.hstack([base_terms, z_modulated])

        # Solve linear system efficiently
        T, _, _, _ = lstsq(X, this_out_centered, rcond=None)

        # Build basis for current node (single-step, no insertions)
        node_xy = np.array([x - x_shift, y - y_shift])
        nx, ny = node_xy
        basis_eval = [1.0, nx, ny]

        for order in range(2, max_order + 1):
            for ox in range(order + 1):
                oy = order - ox
                basis_eval.append((nx ** ox) * (ny ** oy))

        basis_eval = np.array(basis_eval)
        z_modulated_eval = z * basis_eval

        final_input = np.concatenate([basis_eval, z_modulated_eval])
        new_pos = final_input @ T

        new_pos[0] += x_shift
        new_pos[1] += y_shift
        transformed_nodes[k] = new_pos

    return transformed_nodes

def warp_arbor(
    nodes: np.ndarray,
    edges: np.ndarray,
    radii: np.ndarray,
    surface_mapping: dict,
    voxel_resolution: Union[float, list] = [1., 1., 1.],
    conformal_jump: int = 1,
    verbose: bool = False
) -> dict:
    """
    Applies a local surface flattening (warp) to a neuronal arbor using the results
    of previously computed surface mappings.

    Parameters
    ----------
    nodes : np.ndarray
        (N, 3) array of [x, y, z] coordinates for the arbor to be warped.
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

    # Unpack mappings and surfaces
    mapped_min = surface_mapping["mapped_min_positions"]
    mapped_max = surface_mapping["mapped_max_positions"]
    VZmin = surface_mapping["thisVZminmesh"]
    VZmax = surface_mapping["thisVZmaxmesh"]
    thisx = surface_mapping["thisx"] + 1 
    thisy = surface_mapping["thisy"] + 1 
    # this is one ugly hack: thisx and thisy are 1-based in MATLAB
    # but 0-based in Python; the rest of the code is to produce exact
    # same results as MATLAB given the SAME input, that means thisx and 
    # thisy needs to be 1-based, but we need to shift it back to 0-based 
    # when slicing
    
    # Convert MATLAB 1-based inclusive ranges to Python slices
    # If thisx/thisy are consecutive integer indices:
    # x_vals = np.arange(thisx[0], thisx[-1] + 1)  # matches [thisx(1):thisx(end)] in MATLAB
    # y_vals = np.arange(thisy[0], thisy[-1] + 1)  # matches [thisy(1):thisy(end)] in MATLAB
    x_vals = np.arange(thisx[0], thisx[-1] + 1, conformal_jump)
    y_vals = np.arange(thisy[0], thisy[-1] + 1, conformal_jump)

    # Create a meshgrid shaped like MATLAB's [tmpymesh, tmpxmesh] = meshgrid(yRange, xRange).
    # This means we want shape (len(x_vals), len(y_vals)) for each array, with row=“x”, col=“y”:
    tmpxmesh, tmpymesh = np.meshgrid(x_vals, y_vals, indexing="ij")
    # tmpxmesh.shape == tmpymesh.shape == (len(x_vals), len(y_vals))

    # Extract the corresponding subregion of the surfaces so it also has shape (len(x_vals), len(y_vals)).
    # In MATLAB: tmpminmesh = thisVZminmesh(xRange, yRange)
    tmp_min = VZmin[x_vals[:, None]-1, y_vals-1]  # shape (len(x_vals), len(y_vals))
    tmp_max = VZmax[x_vals[:, None]-1, y_vals-1]  # shape (len(x_vals), len(y_vals))

    # Now flatten in column-major order (like MATLAB’s A(:)) to line up with tmpxmesh(:), etc.
    top_input_pos = np.column_stack([
        tmpxmesh.ravel(order="F"),
        tmpymesh.ravel(order="F"),
        tmp_min.ravel(order="F")
    ])

    bot_input_pos = np.column_stack([
        tmpxmesh.ravel(order="F"),
        tmpymesh.ravel(order="F"),
        tmp_max.ravel(order="F")
    ])

    # Finally, the “mapped” output is unaffected by the flattening order mismatch,
    # but we keep it consistent with MATLAB’s final step:
    top_output_pos = np.column_stack([
        mapped_min[:, 0],
        mapped_min[:, 1],
        np.median(tmp_min) * np.ones(mapped_min.shape[0])
    ])

    bot_output_pos = np.column_stack([
        mapped_max[:, 0],
        mapped_max[:, 1],
        np.median(tmp_max) * np.ones(mapped_max.shape[0])
    ])

    # return top_input_pos, bot_input_pos, top_output_pos, bot_output_pos

    # Apply local least-squares registration to each node
    if verbose:
        print("Warping nodes...")
        start_time = time.time()
    warped_nodes = local_ls_registration(nodes, top_input_pos, bot_input_pos, top_output_pos, bot_output_pos)
    if verbose:
        print(f"Nodes warped in {time.time() - start_time:.2f} seconds.")

    # Compute median Z-planes
    med_VZmin = np.median(tmp_min)
    med_VZmax = np.median(tmp_max)

    # Build output dictionary
    warped_arbor = {
        'nodes': warped_nodes * voxel_resolution,
        'edges': edges,
        'radii': radii,
        'medVZmin': med_VZmin,
        'medVZmax': med_VZmax,
    }

    return warped_arbor


# =====================================================================
# helpers for get_zprofile()
# =====================================================================

def segment_lengths(
    nodes: np.ndarray,
    edges: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Edge length at each child node and the corresponding mid-point.

    `edges` are the raw SWC child/parent pairs *1-based*.
    """

    # check if soma is first node
    if edges[0, 1] == -1:
        # remove soma from edges
        edges = edges[1:]

    child = edges[:, 0].astype(int) - 1     # → 0-based
    parent = edges[:, 1].astype(int) - 1

    density = np.zeros(nodes.shape[0], dtype=float)
    mid     = nodes.copy()

    vec      = nodes[parent] - nodes[child]
    seg_len  = np.linalg.norm(vec, axis=1)

    density[child]  = seg_len
    mid[child]      = nodes[child] + 0.5 * vec

    return density, mid

def gridder1d(
    z_samples: np.ndarray,
    density:   np.ndarray,
    n: int,
) -> np.ndarray:
    """
    Kaiser–Bessel gridding kernel in 1-D   (α=2, W=5 as in the paper)

    Port of Uygar Sümbül’s MATLAB `gridder1d`
    All comments match the original line-for-line.
    """
    if z_samples.shape != density.shape:
        raise ValueError("z_samples and density must have the same shape")

    # ------------------------------------------------------------------
    # Constants
    # ------------------------------------------------------------------
    alpha, W, err = 2, 5, 1e-3
    S    = int(np.ceil(0.91 / err / alpha))
    beta = np.pi * np.sqrt((W / alpha * (alpha - 0.5))**2 - 0.8)

    # ------------------------------------------------------------------
    # Pre-computed Kaiser–Bessel lookup table (LUT)
    # ------------------------------------------------------------------
    s      = np.linspace(-1, 1, 2 * S * W + 1)
    F_kbZ  = i0(beta * np.sqrt(1 - s**2))
    F_kbZ /= F_kbZ.max()

    # ------------------------------------------------------------------
    # Fourier transform of the 1-D kernel
    # ------------------------------------------------------------------
    Gz   = alpha * n
    z    = np.arange(-Gz // 2, Gz // 2)
    arg  = (np.pi * W * z / Gz)**2 - beta**2

    kbZ  = np.empty_like(arg, dtype=float)
    pos, neg = arg > 1e-12, arg < -1e-12
    kbZ[pos] = np.sin(np.sqrt(arg[pos]))    / np.sqrt(arg[pos])
    kbZ[neg] = np.sinh(np.sqrt(-arg[neg]))  / np.sqrt(-arg[neg])
    kbZ[~(pos | neg)] = 1.0
    kbZ *= np.sqrt(Gz)

    # ------------------------------------------------------------------
    # Oversampled grid and accumulation
    # ------------------------------------------------------------------
    n_os = Gz
    out  = np.zeros(n_os, dtype=float)

    centre = n_os / 2 + 1                   # 1-based like MATLAB
    nz     = centre + n_os * z_samples      # fractional indices

    half_w = (W - 1) // 2
    for lz in range(-half_w, half_w + 1):
        nzt = np.round(nz + lz).astype(int)
        zpos = S * ((nz - nzt) + W / 2)
        kw   = F_kbZ[np.round(zpos).astype(int)]

        nzt = np.clip(nzt, 0, n_os - 1)     # clamp out-of-range
        np.add.at(out, nzt, density * kw)

    out[0] = out[-1] = 0.0                  # edge artefacts

    # ------------------------------------------------------------------
    # myifft  →  de-apodise  →  abs(myfft3)
    # ------------------------------------------------------------------
    u = n
    f  = np.fft.ifftshift(np.fft.ifft(np.fft.ifftshift(out))) * np.sqrt(u)
    f  = f[int(np.ceil((f.size - u) / 2)) : int(np.ceil((f.size + u) / 2))]

    f /= kbZ[u // 2 : 3 * u // 2]           # de-apodisation

    F  = np.fft.fftshift(np.fft.fftn(np.fft.fftshift(f))) / np.sqrt(f.size)
    return np.abs(F)

# =====================================================================
# helpers for get_zprofile() END
# =====================================================================

def get_zprofile(
    warped_arbor: dict,
    z_res: float = 1.,
    grid_point_count: int = 120,
    z_ext: float = 5.,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute a 1-D depth profile (length per z-bin) from a warped arbor.

    Parameters
    ----------
    warped_arbor
        Dict returned by ``warp_arbor()``. Must contain
            'nodes'   – (N, 3) xyz coordinates in µm,
            'edges'   – (E, 2) SWC child/parent pairs (1-based),
            'medVZmin', 'medVZmax'  – median of the ON and OFF SAC surfaces.
    z_res
        Spatial resolution along z (µm / voxel) *after* warping. Depends on
        what unit the medvzmin/max were in. If they were already in µm, 
        then z_res = 1. If they were in pixels, then z_res = voxel_size.
    grid_point_count
        Number of evenly-spaced output bins along z.
    z_ext
        Depth range (in “SAC units”) that was mapped to the unit interval
        during warping; see Sumbul et al. 2014 for details.

    Returns
    -------
    x_um
        Bin centres in micrometres (depth in IPL).
    z_dist
        Dendritic length contained in each bin (same units as input nodes).
    """
    # ------------------------------------------------------------------
    # 1.  Edge lengths and mid-points  ---------------------------------
    # ------------------------------------------------------------------
    density, mid_nodes = segment_lengths(
        warped_arbor["nodes"], warped_arbor["edges"]
    )

    # ------------------------------------------------------------------
    # 2.  Normalise z to [-0.5, +0.5] for the gridding kernel ----------
    # ------------------------------------------------------------------
    vz_min, vz_max = warped_arbor["medVZmin"], warped_arbor["medVZmax"]
    z_samples = (
        (mid_nodes[:, 2] / z_res - vz_min) / (vz_max - vz_min) / z_ext
    )

    # ------------------------------------------------------------------
    # 3.  Non-uniform → uniform resampling (Kaiser–Bessel gridding) ----
    # ------------------------------------------------------------------
    z_dist = gridder1d(z_samples, density, grid_point_count)

    # ------------------------------------------------------------------
    # 4.  Re-scale so that Σ z_dist · z_res  equals total dendritic len.
    # ------------------------------------------------------------------
    z_dist *= density.sum() / (z_dist.sum() * z_res)

    # ------------------------------------------------------------------
    # 5.  Bin centres on a metric x-axis (µm) for plotting -------------
    # ------------------------------------------------------------------
    dz = z_ext / grid_point_count
    bins = -z_ext / 2 + np.arange(grid_point_count) * dz
    x_um = bins * (grid_point_count * z_res / z_ext)

    return x_um, z_dist
