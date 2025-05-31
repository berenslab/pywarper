import numpy as np
import pandas as pd
import scipy.io
import skeliner as sk

from pywarper.arbor import warp_arbor
from pywarper.surface import build_mapping, fit_sac_surface


def test_arbor():
    """
    Test the warping of arbor against the expected values from MATLAB.

    Given the same input, the output of the Python code should match the output of the MATLAB code.
    """
    
    def read_ChAT(filename):
        
        df = pd.read_csv(filename, comment='#', sep=r'\s+')
        x = df["X"].values.astype(float)
        y = df["Slice"].values.astype(float)
        z = df["Y"].values.astype(float)

        x = x + 1
        z = z + 1

        return x, y ,z
    
    chat_top = read_ChAT("./tests/data/Image013-009_01_ChAT-TopBand-Mike.txt") # should be the off sac layer
    chat_bottom = read_ChAT("./tests/data/Image013-009_01_ChAT-BottomBand-Mike.txt") # should be the on sac layer
    # but the image can be flipped
    if chat_top[2].mean() > chat_bottom[2].mean():
        off_sac = chat_top
        on_sac = chat_bottom
    else:
        off_sac = chat_bottom
        on_sac = chat_top

    # _, nodes, edges, radii = read_arbor_trace("./tests/data/Image013-009_01_raw_latest_Uygar.swc")
    skel = sk.io.load_swc("./tests/data/Image013-009_01_raw_latest_Uygar.swc")
    # +1 to match MATLAB indexing (1-based)
    skel.nodes += 1

    off_sac_surface, _, _ = fit_sac_surface(x=off_sac[0], y=off_sac[1], z=off_sac[2], smoothness=15)
    on_sac_surface, _, _ = fit_sac_surface(x=on_sac[0], y=on_sac[1], z=on_sac[2], smoothness=15)
    arbor_boundaries = np.array([skel.nodes[:, 0].min(), skel.nodes[:, 0].max(), skel.nodes[:, 1].min(), skel.nodes[:, 1].max()])
    surface_mapping = build_mapping(on_sac_surface, off_sac_surface, arbor_boundaries, conformal_jump=2, verbose=True)
    
    # to me it makes more sense to use physical units from the start but this is how the original code works
    # so I will keep it like this: only convert the warped arbor to physical units at the `warp_arbor` function
    voxel_resolution = [0.4, 0.4, 0.5]
    # warped_arbor = warp_arbor(nodes, edges, radii, surface_mapping, voxel_resolution=voxel_resolution, conformal_jump=2, verbose=True)
    warped_arbor = warp_arbor(skel, surface_mapping, voxel_resolution=voxel_resolution, conformal_jump=2, verbose=True)
    warped_nodes = warped_arbor.nodes

    warped_arbor_mat = scipy.io.loadmat("./tests/data/warpedArbor_jump.mat", squeeze_me=True, struct_as_record=False)
    warped_nodes_mat = warped_arbor_mat["warpedArbor"].nodes

    assert np.allclose(warped_nodes, warped_nodes_mat, rtol=1e-5, atol=1e-8), "Warped nodes do not match expected values."
    assert np.isclose(warped_arbor.extra["med_z_on"], warped_arbor_mat["warpedArbor"].medVZmin), "Minimum VZ does not match expected value."
    assert np.isclose(warped_arbor.extra["med_z_off"], warped_arbor_mat["warpedArbor"].medVZmax), "Maximum VZ does not match expected value."
    assert warped_arbor.extra["med_z_on"] < warped_arbor.extra["med_z_off"], "Minimum VZ should be less than maximum VZ."
