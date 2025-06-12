"""pywarper.utils"""
import numpy as np


def read_sumbul_et_al_chat_bands(fname: str, unit="voxel") -> dict[str, np.ndarray]:
    """
    Read a ChAT-band point cloud exported by KNOSSOS/FiJi.

    Parameters
    ----------
    fname : str
        Plain-text file with columns Area, Mean, Min, Max, X, Y, Slice
        plus an unlabeled first column (row index).

    Returns
    -------
    dict
        Keys ``x``, ``y``, ``z`` (1-based index, float64).
    """
    # The file has eight numeric columns; we only need X (col 5),
    # Y (col 6) and Slice (col 7). 0-based indices: 5, 6, 7.
    data = np.loadtxt(
        fname,
        comments="#",
        skiprows=1,        # skip the header line
        usecols=(5, 7, 6), # X, Slice, Y in desired order
        dtype=np.float64,
    )

    x = data[:, 0] + 1          # KNOSSOS X  → +1 for MATLAB convention
    y = data[:, 1]              # Slice (already 1-based)
    z = data[:, 2] + 1          # KNOSSOS Y  → +1

    if unit == "voxel":
        return {"x": x, "y": y, "z": z}
    elif unit == "physical":
        # Convert to physical units (in micrometers)
        voxel_resolution = [0.4, 0.4, 0.5]
        return {
            "x": x * voxel_resolution[0],
            "y": y * voxel_resolution[1],
            "z": z * voxel_resolution[2],
        }
    else:
        raise ValueError(f"Unknown unit: {unit}. Use 'voxel' or 'physical'.")