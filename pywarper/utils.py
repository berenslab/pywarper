import numpy as np
import pandas as pd

# import scipy.spatial

def read_arbor_trace(
    datapath: str,
    downsample_factor: int = 1
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """
    Reads a neuronal arbor trace from an SWC file and optionally downsamples the data.

    Parameters
    ----------
    datapath : str
        Path to the SWC file to read.
    downsample_factor : int, default=1
        If greater than 1, every n-th row is selected (in row order) to reduce
        the total number of points in the returned data.

    Returns
    -------
    df : pd.DataFrame
        A DataFrame containing all columns read from the SWC file:
        'n', 'type', 'x', 'y', 'z', 'radius', 'parent'.
    coords : np.ndarray
        An (N, 3) array of node coordinates [x, y, z].
    edges : np.ndarray
        An (N, 2) array where each row is [node_id, parent_id].
        Note that ids and parents may be updated if downsample_factor > 1.
    radii : np.ndarray
        An (N,) array of radii corresponding to each node.

    Notes
    -----
    1. Comment lines in the SWC file (prefixed by '#') are automatically skipped.
    2. The downsampled data is re-indexed so that the 'parent' field reflects
       the new node numbering.
    """
    # Read the SWC file into a DataFrame
    df = pd.read_csv(datapath, comment='#',
                     names=['n', 'type', 'x', 'y', 'z', 'radius', 'parent'],
                     index_col=False, sep=r'\s+')

    if downsample_factor > 1:
        # Downsample the DataFrame by selecting every nth point
        downsampled_df = df.iloc[::downsample_factor].copy()

        # Update the indices
        id_map = {old_id: new_id for new_id, old_id in enumerate(downsampled_df['n'], start=1)}
        downsampled_df['n'] = downsampled_df['n'].map(id_map)
        downsampled_df['parent'] = downsampled_df['parent'].map(lambda x: id_map.get(x, -1))

        df = downsampled_df

    return df, df[["x", "y", "z"]].values, df[["n", "parent"]].values, df["radius"].values


# def read_arbor_trace(datapath):

#     df = pd.read_csv(datapath, comment='#',
#                       names=['n', 'type', 'x', 'y', 'z', 'radius', 'parent'], index_col=False, sep=r'\s+')

#     return df, df[["x", "y", "z"]].values, df[["n", "parent"]].values, df["radius"].values

# def summarize_z_at_sampled_xy(x, y, z, grid_spacing=10, radius=5, percentiles=[10, 25, 50, 75, 90]):
#     """
#     Sample a sparse subset of original (x, y) points and compute z summary stats
#     over a local neighborhood (radius).

#     Parameters:
#         x, y, z (np.ndarray): Original point cloud arrays (N,)
#         grid_spacing (float): Sampling interval for (x, y) grid
#         radius (float): Radius around each sample point to summarize z values
#         percentiles (list): Percentiles to compute for z

#     Returns:
#         x_samp, y_samp: Sampled locations (M,)
#         z_mean: mean Z value around each (x, y)
#         z_stats: dict of z_pXX arrays (same length as x_samp)
#     """
#     # Build spatial tree on original points
#     xy = np.column_stack((x, y))
#     tree = scipy.spatial.cKDTree(xy)

#     # Generate sparse sampling grid
#     x_grid = np.arange(x.min(), x.max(), grid_spacing)
#     y_grid = np.arange(y.min(), y.max(), grid_spacing)

#     xx, yy = np.meshgrid(x_grid, y_grid)
#     sample_points = np.column_stack((xx.ravel(), yy.ravel()))

#     # For each sample point, find neighbors within radius
#     neighbors = tree.query_ball_point(sample_points, r=radius)

#     x_samp, y_samp = [], []
#     z_mean = []
#     z_percentiles = {f'z_p{p}': [] for p in percentiles}

#     # Get max edges (float values)
#     for (xi, yi), idxs in zip(sample_points, neighbors):
#         if len(idxs) == 0:
#             continue  # skip interior point with no neighbors
#         else:
#             z_vals = z[idxs]

#         x_samp.append(xi)
#         y_samp.append(yi)
#         z_mean.append(np.mean(z_vals))
#         for p in percentiles:
#             z_percentiles[f'z_p{p}'].append(np.percentile(z_vals, p))

#     return (
#         np.array(x_samp),
#         np.array(y_samp),
#         np.array(z_mean),
#         {k: np.array(v) for k, v in z_percentiles.items()}
#     )
