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

