import time

import numpy as np
from numpy.linalg import lstsq


def local_ls_registration(nodes, top_input_pos, bot_input_pos, top_output_pos, bot_output_pos, window=5, max_order=2):
    """
    Applies a local least-squares polynomial transformation to each node based on nearby surface correspondences.

    Parameters:
    - nodes: Nx3 array of [x, y, z] positions to be transformed
    - top_input_pos: Mx3 array of surface coordinates on the top band (original)
    - bot_input_pos: Mx3 array of surface coordinates on the bottom band (original)
    - top_output_pos: Mx3 array of mapped coordinates on the top band (flattened)
    - bot_output_pos: Mx3 array of mapped coordinates on the bottom band (flattened)
    - window: neighborhood radius in pixels
    - max_order: maximum total polynomial order (e.g., 2 for quadratic)

    Returns:
    - transformed_nodes: Nx3 array of transformed [x, y, z] positions
    """
    transformed_nodes = np.zeros_like(nodes)

    for k in range(nodes.shape[0]):
        x, y, z = nodes[k]

        # Extract local neighborhood
        lx, ux = int(round(x - window)), int(round(x + window))
        ly, uy = int(round(y - window)), int(round(y + window))

        in_top = top_input_pos[(top_input_pos[:, 0] >= lx) & (top_input_pos[:, 0] <= ux) &
                               (top_input_pos[:, 1] >= ly) & (top_input_pos[:, 1] <= uy)]
        in_bot = bot_input_pos[(bot_input_pos[:, 0] >= lx) & (bot_input_pos[:, 0] <= ux) &
                               (bot_input_pos[:, 1] >= ly) & (bot_input_pos[:, 1] <= uy)]
        out_top = top_output_pos[(top_input_pos[:, 0] >= lx) & (top_input_pos[:, 0] <= ux) &
                                 (top_input_pos[:, 1] >= ly) & (top_input_pos[:, 1] <= uy)]
        out_bot = bot_output_pos[(bot_input_pos[:, 0] >= lx) & (bot_input_pos[:, 0] <= ux) &
                                 (bot_input_pos[:, 1] >= ly) & (bot_input_pos[:, 1] <= uy)]

        this_in = np.vstack([in_top, in_bot])
        this_out = np.vstack([out_top, out_bot])

        if len(this_in) < 12:
            # Not enough points for a stable fit
            transformed_nodes[k] = nodes[k]
            continue

        # Center coordinates
        x_shift = np.mean(this_in[:, 0])
        y_shift = np.mean(this_in[:, 1])
        this_in[:, 0] -= x_shift
        this_out[:, 0] -= x_shift
        this_in[:, 1] -= y_shift
        this_out[:, 1] -= y_shift

        # Build polynomial basis
        X = []
        for i in range(this_in.shape[0]):
            terms = [this_in[i, 0], this_in[i, 1], 1.0]
            for total_order in range(2, max_order + 1):
                for o in range(total_order + 1):
                    terms.insert(0, this_in[i, 0]**o * this_in[i, 1]**(total_order - o))
            base_terms = np.array(terms)
            z_modulated = this_in[i, 2] * base_terms
            X.append(np.concatenate([base_terms, z_modulated]))
        X = np.vstack(X)

        # Solve local linear system
        T, _, _, _ = lstsq(X, this_out, rcond=None)

        # Build transform for current node
        node_xy = np.array([x - x_shift, y - y_shift])
        terms = [node_xy[0], node_xy[1], 1.0]
        for total_order in range(2, max_order + 1):
            for o in range(total_order + 1):
                terms.insert(0, node_xy[0]**o * node_xy[1]**(total_order - o))
        base_terms = np.array(terms)
        z_modulated = z * base_terms
        final_input = np.concatenate([base_terms, z_modulated])
        new_pos = final_input @ T
        new_pos[0] += x_shift
        new_pos[1] += y_shift
        transformed_nodes[k] = new_pos

    return transformed_nodes

def warp_arbor(nodes, edges, radii, surface_mapping, verbose=False):
    """
    Applies a local surface flattening to a neuronal arbor using surface mapping results.

    Parameters:
    - nodes: Nx3 array of [x, y, z] coordinates of the arbor
    - edges: Ex2 array of arbor connectivity
    - radii: N-length array of node radii
    - surface_mapping: dict with keys:
        - 'mappedMinPositions', 'mappedMaxPositions'
        - 'thisVZminmesh', 'thisVZmaxmesh'
        - 'thisx', 'thisy'
    - voxel_dim: 1x3 list or array with physical voxel dimensions in µm
    - conformal_jump: downsampling step size used during conformal mapping

    Returns:
    - warped_arbor: dict with updated 'nodes', 'edges', 'radii', 'medVZmin', 'medVZmax'
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
    x_vals = np.arange(thisx[0], thisx[-1] + 1)  # matches [thisx(1):thisx(end)] in MATLAB
    y_vals = np.arange(thisy[0], thisy[-1] + 1)  # matches [thisy(1):thisy(end)] in MATLAB

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
        'nodes': warped_nodes,
        'edges': edges,
        'radii': radii,
        'medVZmin': med_VZmin,
        'medVZmax': med_VZmax
    }

    return warped_arbor
