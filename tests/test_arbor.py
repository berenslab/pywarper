import numpy as np
import pandas as pd
import scipy.io

from pywarper.arbor import warp_arbor
from pywarper.surface import fit_surface, warp_surface
from pywarper.utils import read_arbor_trace


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

    off_sac = read_ChAT("./tests/data/Image013-009_01_ChAT-TopBand-Mike.txt")
    on_sac = read_ChAT("./tests/data/Image013-009_01_ChAT-BottomBand-Mike.txt")
    rgc, nodes, edges, radii = read_arbor_trace("./tests/data/Image013-009_01_raw_latest_Uygar.swc")
    nodes += 1
    thisvzmaxmesh, xgridmax, ygridmax = fit_surface(x=off_sac[0], y=off_sac[1], z=off_sac[2], smoothness=15)
    thisvzminmesh, xgridmin, ygridmin = fit_surface(x=on_sac[0], y=on_sac[1], z=on_sac[2], smoothness=15)
    arbor_boundaries = np.array([nodes[:, 0].min(), nodes[:, 0].max(), nodes[:, 1].min(), nodes[:, 1].max()])
    surface_mapping = warp_surface(thisvzminmesh, thisvzmaxmesh, arbor_boundaries, conformal_jump=2, verbose=True)
    warped_arbor = warp_arbor(nodes, edges, radii, surface_mapping, conformal_jump=2, verbose=True)
    warped_nodes = warped_arbor["nodes"]

    warped_arbor_mat = scipy.io.loadmat("./tests/data/warpedArbor_jump.mat", squeeze_me=True, struct_as_record=False)
    warped_nodes_mat = warped_arbor_mat["warpedArbor"].nodes


    assert np.allclose(warped_nodes, warped_nodes_mat, rtol=1e-5, atol=1e-8), "Warped nodes do not match expected values."