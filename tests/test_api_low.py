import numpy as np
import scipy.io
import skeliner as sk

from pywarper.surface import build_mapping, fit_sac_surface
from pywarper.utils import read_sumbul_et_al_chat_bands
from pywarper.warpers import warp_skeleton


def test_skeleton():
    """
    Test the warping of skeleton against the expected values from MATLAB.

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

    skel = sk.io.load_swc("./tests/data/Image013-009_01_raw_latest_Uygar.swc")
    skel.nodes += 1  # to match MATLAB indexing (1-based)

    off_sac_surface, _, _ = fit_sac_surface(
        x=off_sac["x"],
        y=off_sac["y"],
        z=off_sac["z"],
        smoothness=15,
        backward_compatible=True,
    )
    on_sac_surface, _, _ = fit_sac_surface(
        x=on_sac["x"],
        y=on_sac["y"],
        z=on_sac["z"],
        smoothness=15,
        backward_compatible=True,
    )
    skeleton_boundaries = np.array(
        [
            skel.nodes[:, 0].min(),
            skel.nodes[:, 0].max(),
            skel.nodes[:, 1].min(),
            skel.nodes[:, 1].max(),
        ]
    )
    surface_mapping = build_mapping(
        on_sac_surface,
        off_sac_surface,
        skeleton_boundaries,
        conformal_jump=2,
        n_anchors=4,
        backward_compatible=True,
        verbose=True,
    )

    # to me it makes more sense to use physical units from the start but this is how the original code works
    # so I will keep it like this: only convert the warped skeleton to physical units at the `warp_skeleton` function
    voxel_resolution = [0.4, 0.4, 0.5]
    warped_skeleton = warp_skeleton(
        skel,
        surface_mapping,
        voxel_resolution=voxel_resolution,
        conformal_jump=2,
        backward_compatible=True,
        verbose=True,
    )
    warped_nodes = warped_skeleton.extra["prenormed_nodes"]

    warped_skeleton_mat = scipy.io.loadmat(
        "./tests/data/warpedArbor_jump.mat", squeeze_me=True, struct_as_record=False
    )
    warped_nodes_mat = warped_skeleton_mat["warpedArbor"].nodes

    assert np.allclose(warped_nodes, warped_nodes_mat, rtol=1e-5, atol=1e-8), (
        "Warped nodes do not match expected values."
    )
    assert np.isclose(
        warped_skeleton.extra["med_z_on"], warped_skeleton_mat["warpedArbor"].medVZmin
    ), "Minimum VZ does not match expected value."
    assert np.isclose(
        warped_skeleton.extra["med_z_off"], warped_skeleton_mat["warpedArbor"].medVZmax
    ), "Maximum VZ does not match expected value."
    assert warped_skeleton.extra["med_z_on"] < warped_skeleton.extra["med_z_off"], (
        "Minimum VZ should be less than maximum VZ."
    )

    # Ensure the warped skeleton preserves the same attributes as the input skeleton
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
        assert hasattr(warped_skeleton, attr), (
            f"Warped skeleton missing attribute '{attr}'"
        )

    # Attributes that should remain identical to the input skeleton
    assert np.array_equal(warped_skeleton.edges, skel.edges), (
        "Edges should be preserved."
    )
    assert np.array_equal(warped_skeleton.radii, skel.radii), (
        "Radii should be preserved."
    )

    # ntype may be None or an array-like; preserve equality
    if getattr(skel, "ntype", None) is None:
        assert warped_skeleton.ntype is None, (
            "ntype should be None if input ntype is None."
        )
    else:
        if isinstance(skel.ntype, np.ndarray):
            assert np.array_equal(warped_skeleton.ntype, skel.ntype), (
                "ntype should be preserved."
            )
        else:
            assert warped_skeleton.ntype == skel.ntype, "ntype should be preserved."

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

    _assert_mapping_equal(skel.node2verts, warped_skeleton.node2verts, "node2verts")
    _assert_mapping_equal(skel.vert2node, warped_skeleton.vert2node, "vert2node")
