import numpy as np
import scipy.io

from pywarper import Warper
from pywarper.utils import read_sumbul_et_al_chat_bands


def test_warper():
    """
    Test the warping of arbor against the expected values from MATLAB.

    Given the same input, the output of the Python code should match the output of the MATLAB code.
    """

    chat_top = read_sumbul_et_al_chat_bands(
        "./tests/data/Image013-009_01_ChAT-TopBand-Mike.txt"
    )  # should be the off sac layer
    chat_bottom = read_sumbul_et_al_chat_bands(
        "./tests/data/Image013-009_01_ChAT-BottomBand-Mike.txt"
    )  # should be the on sac layer
    # but the image can be flipped
    if chat_top["z"].mean() > chat_bottom["z"].mean():
        off_sac = chat_top
        on_sac = chat_bottom
    else:
        off_sac = chat_bottom
        on_sac = chat_top

    cell_path = "./tests/data/Image013-009_01_raw_latest_Uygar.swc"
    voxel_resolution = [0.4, 0.4, 0.5]
    w = Warper(
        off_sac, on_sac, cell_path, voxel_resolution=voxel_resolution, verbose=False
    )
    w.skeleton.nodes += 1  # unnecessary, but to match the matlab behavior
    w.fit_surfaces(backward_compatible=True)
    w.build_mapping(n_anchors=4, backward_compatible=True)
    w.warp_skeleton(backward_compatible=True)

    warped_skeleton_mat = scipy.io.loadmat(
        "./tests/data/warpedArbor_jump.mat", squeeze_me=True, struct_as_record=False
    )
    warped_nodes_mat = warped_skeleton_mat["warpedArbor"].nodes

    assert np.allclose(
        w.warped_skeleton.extra["prenormed_nodes"],
        warped_nodes_mat,
        rtol=1e-5,
        atol=1e-8,
    ), "Warped nodes do not match expected values."
    assert np.isclose(
        w.warped_skeleton.extra["med_z_on"], warped_skeleton_mat["warpedArbor"].medVZmin
    ), "Minimum VZ does not match expected value."
    assert np.isclose(
        w.warped_skeleton.extra["med_z_off"],
        warped_skeleton_mat["warpedArbor"].medVZmax,
    ), "Maximum VZ does not match expected value."
    assert w.warped_skeleton.extra["med_z_on"] < w.warped_skeleton.extra["med_z_off"], (
        "Minimum VZ should be less than maximum VZ."
    )

    # Ensure the warped skeleton preserves the same attributes as the input skeleton
    input_skel = w.skeleton
    output_skel = w.warped_skeleton

    expected_attrs = [
        "nodes",
        "edges",
        "radii",
        "ntype",
        "soma",
        "meta",
        "node2verts",
        "vert2node",
        "extra",
    ]
    for attr in expected_attrs:
        assert hasattr(output_skel, attr), f"Warped skeleton missing attribute '{attr}'"

    # Attributes that should remain identical to the input skeleton
    assert np.array_equal(output_skel.edges, input_skel.edges), (
        "Edges should be preserved."
    )
    assert np.array_equal(output_skel.radii, input_skel.radii), (
        "Radii should be preserved."
    )

    # ntype may be None or an array-like; preserve equality
    if getattr(input_skel, "ntype", None) is None:
        assert output_skel.ntype is None, "ntype should be None if input ntype is None."
    else:
        if isinstance(input_skel.ntype, np.ndarray):
            assert np.array_equal(output_skel.ntype, input_skel.ntype), (
                "ntype should be preserved."
            )
        else:
            assert output_skel.ntype == input_skel.ntype, "ntype should be preserved."

    # node<->vertex mappings should be preserved (None, arrays or dicts)
    def _assert_mapping_equal(a, b, name):
        if a is None:
            assert b is None, f"{name} should be None if input {name} is None."
        elif isinstance(a, dict):
            assert isinstance(b, dict), f"{name} should be a dict."
            assert set(a.keys()) == set(b.keys()), f"{name} keys should be preserved."
            for k in a:
                av, bv = a[k], b[k]
                if isinstance(av, np.ndarray):
                    assert np.array_equal(bv, av), f"{name}[{k}] should be preserved."
                elif isinstance(av, (list, tuple)):
                    assert list(bv) == list(av), f"{name}[{k}] should be preserved."
                else:
                    assert bv == av, f"{name}[{k}] should be preserved."
        elif isinstance(a, np.ndarray):
            assert np.array_equal(a, b), f"{name} should be preserved."
        else:
            assert a == b, f"{name} should be preserved."

    _assert_mapping_equal(input_skel.node2verts, output_skel.node2verts, "node2verts")
    _assert_mapping_equal(input_skel.vert2node, output_skel.vert2node, "vert2node")
