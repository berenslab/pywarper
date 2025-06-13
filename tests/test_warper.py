import numpy as np
import scipy.io

from pywarper import Warper
from pywarper.utils import read_sumbul_et_al_chat_bands


def test_warper():
    """
    Test the warping of arbor against the expected values from MATLAB.

    Given the same input, the output of the Python code should match the output of the MATLAB code.
    """
    
    chat_top = read_sumbul_et_al_chat_bands("./tests/data/Image013-009_01_ChAT-TopBand-Mike.txt") # should be the off sac layer
    chat_bottom = read_sumbul_et_al_chat_bands("./tests/data/Image013-009_01_ChAT-BottomBand-Mike.txt") # should be the on sac layer
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
    w.skel.nodes += 1  # unnecessary, but to match the matlab behavior
    w.fit_surfaces(backward_compatible=True)
    w.build_mapping(n_anchors=4, backward_compatible=True)
    w.warp_arbor(backward_compatible=True)

    warped_arbor_mat = scipy.io.loadmat("./tests/data/warpedArbor_jump.mat", squeeze_me=True, struct_as_record=False)
    warped_nodes_mat = warped_arbor_mat["warpedArbor"].nodes

    assert np.allclose(w.warped_arbor.extra["warped_nodes"], warped_nodes_mat, rtol=1e-5, atol=1e-8), "Warped nodes do not match expected values."
    assert np.isclose(w.warped_arbor.extra["med_z_on"], warped_arbor_mat["warpedArbor"].medVZmin), "Minimum VZ does not match expected value."
    assert np.isclose(w.warped_arbor.extra["med_z_off"], warped_arbor_mat["warpedArbor"].medVZmax), "Maximum VZ does not match expected value."
    assert w.warped_arbor.extra["med_z_on"] < w.warped_arbor.extra["med_z_off"], "Minimum VZ should be less than maximum VZ."
