import numpy as np
import pandas as pd
import scipy.io

from pywarper import Warper


def test_warper():
    """
    Test the warping of arbor against the expected values from MATLAB.

    Given the same input, the output of the Python code should match the output of the MATLAB code.
    """
    
    def read_chat(fname: str) -> dict[str, np.ndarray]:
        """Read a ChAT‐band point cloud exported by KNOSSOS/FiJi.

        Parameters
        ----------
        fname
            Plain‐text file with at least the columns *X*, *Y*, *Slice* as in the
            Sümbül *et al.* (2014) dataset.

        Returns
        -------
        dict
            With keys ``x``, ``y``, ``z`` (1‑based index, *float64*).
        """
        df = pd.read_csv(fname, comment="#", sep=r"\s+")

        # KNOSSOS axes → (x, y, z) in µm; +1 to mimic MATLAB 1‑based convention
        return {
            "x": df["X"].to_numpy(float) + 1,
            "y": df["Slice"].to_numpy(float),
            "z": df["Y"].to_numpy(float) + 1,
        }
    
    chat_top = read_chat("./tests/data/Image013-009_01_ChAT-TopBand-Mike.txt") # should be the off sac layer
    chat_bottom = read_chat("./tests/data/Image013-009_01_ChAT-BottomBand-Mike.txt") # should be the on sac layer
    # but the image can be flipped
    if chat_top["z"].mean() > chat_bottom["z"].mean():
        off_sac = chat_top
        on_sac = chat_bottom
    else:
        off_sac = chat_bottom
        on_sac = chat_top

    cell_path = "./tests/data/Image013-009_01_raw_latest_Uygar.swc"
    voxel_resolution = [0.4, 0.4, 0.5]
    w = Warper(off_sac, on_sac, cell_path, voxel_resolution=voxel_resolution, verbose=False)
    w.nodes += 1  # unnecessary, but to match the matlab behavior
    w.fit_surfaces()
    w.build_mapping()
    w.warp_arbor()

    warped_arbor_mat = scipy.io.loadmat("./tests/data/warpedArbor_jump.mat", squeeze_me=True, struct_as_record=False)
    warped_nodes_mat = warped_arbor_mat["warpedArbor"].nodes

    assert np.allclose(w.warped_arbor["nodes"], warped_nodes_mat, rtol=1e-5, atol=1e-8), "Warped nodes do not match expected values."
    assert np.isclose(w.warped_arbor["medVZmin"], warped_arbor_mat["warpedArbor"].medVZmin), "Minimum VZ does not match expected value."
    assert np.isclose(w.warped_arbor["medVZmax"], warped_arbor_mat["warpedArbor"].medVZmax), "Maximum VZ does not match expected value."
    assert w.warped_arbor["medVZmin"] < w.warped_arbor["medVZmax"], "Minimum VZ should be less than maximum VZ."
