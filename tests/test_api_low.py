import numpy as np
import pytest
import scipy.io
import skeliner as sk

from pywarper.surface import build_mapping, fit_sac_surface, map_distortion
from pywarper.utils import read_sumbul_et_al_chat_bands
from pywarper.warpers import get_coverage, warp_skeleton


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

    skel = sk.io.load_skeleton_swc("./tests/data/Image013-009_01_raw_latest_Uygar.swc")
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


def _two_dendrites() -> sk.Skeleton:
    """Two straight dendrites at mid-depth, x = 10 and x = 30, joined at y = 30."""
    from skeliner.dataclass import Skeleton, Soma

    a = np.column_stack([np.full(21, 10.0), np.arange(10.0, 31.0), np.full(21, 15.0)])
    b = np.column_stack(
        [np.full(21, 30.0), np.arange(30.0, 9.0, -1.0), np.full(21, 15.0)]
    )
    nodes = np.vstack([a, b])
    n = len(nodes)
    radii = np.full(n, 1.0)
    return Skeleton(
        soma=Soma(center=nodes[0].copy(), axes=np.ones(3), R=np.eye(3), verts=None),
        nodes=nodes,
        radii={"median": radii, "mean": radii.copy(), "trim": radii.copy()},
        edges=np.column_stack([np.arange(n - 1), np.arange(1, n)]),
        ntype=np.ones(n, dtype=int),
    )


def test_coverage_records_volume_faces_and_unobserved_band():
    on = np.full((40, 40), 10.0)
    off = np.full((40, 40), 20.0)
    off_support = np.ones((40, 40), dtype=bool)
    off_support[20:, :] = False
    mapping = build_mapping(
        on,
        off,
        [0, 39, 0, 39],
        conformal_jump=1,
        volume_z=(5.0, 22.0),
        off_support=off_support,
    )
    skel = _two_dendrites()
    warped = warp_skeleton(skel, mapping)

    # a skeleton lying on a single depth still gets a finite profile
    assert np.isfinite(warped.extra["z_profiles"]["length"]["distribution"]).all()

    coverage = warped.extra["coverage"]
    # faces at z = 5 and 22 around bands 10 apart, ON at 0 and OFF at 12
    assert np.allclose(coverage["depth"], [-6.0, 14.4])
    assert np.array_equal(coverage["off_support"], skel.nodes[:, 0] < 20)
    assert "on_support" not in coverage

    inside = get_coverage(warped, extent=(-5.0, 14.0))
    assert inside["cut_below"] == 0.0 and inside["cut_above"] == 0.0
    outside = get_coverage(warped, extent=(-8.0, 16.0))
    assert outside["cut_below"] == 1.0 and outside["cut_above"] == 1.0
    # the x = 30 dendrite and the edge that reaches it: 40 of 60 of cable
    assert np.isclose(outside["off_unsupported"], 40 / 60)
    assert np.isnan(outside["on_unsupported"])

    one_face = build_mapping(
        on, off, [0, 39, 0, 39], conformal_jump=1, volume_z=(None, 22.0)
    )
    depth = warp_skeleton(skel, one_face).extra["coverage"]["depth"]
    assert np.all(depth[:, 0] == -np.inf) and np.allclose(depth[:, 1], 14.4)

    with pytest.raises(ValueError, match="off_support"):
        build_mapping(on, off, [0, 39, 0, 39], off_support=off_support[:-1])


def test_tilted_plane_flattens_without_stretch():
    """A tilted plane is developable: every anchoring keeps lengths and angles."""
    x, y = np.meshgrid(np.arange(60.0), np.arange(60.0), indexing="ij")
    slope = np.tan(np.radians(15.0))
    on = slope * (0.8 * x + 0.6 * y) + 10.0
    off = on + 12.0
    for n_anchors, tol in ((16, 1e-6), (8, 1e-6), (4, 1e-3)):
        mapping = build_mapping(
            on, off, [0, 59, 0, 59], conformal_jump=1, n_anchors=n_anchors
        )
        for band in ("on", "off"):
            summary = map_distortion(mapping)[band]["summary"]
            assert abs(summary["length_scale_p1"] - 1.0) < tol, (n_anchors, summary)
            assert abs(summary["length_scale_p99"] - 1.0) < tol, (n_anchors, summary)
            assert summary["anisotropy_p99"] < 1.0 + tol, (n_anchors, summary)


def test_extra_surfaces_set_depth_piecewise():
    from pywarper.unwarper import unwarp_nodes
    from pywarper.warpers import normalize_nodes, warp_nodes

    on = np.full((40, 40), 10.0)
    off = np.full((40, 40), 20.0)
    extra = {
        "deep": (np.full((40, 40), 4.0), -0.5),  # lands at -6 with the bands at 0, 12
        "outer": (np.full((40, 40), 26.0), 1.75),  # lands at 21
    }
    mapping = build_mapping(
        on,
        off,
        [0, 39, 0, 39],
        conformal_jump=1,
        volume_z=(2.0, 30.0),
        extra_surfaces=extra,
    )
    z = np.array([2.0, 4.0, 7.0, 10.0, 15.0, 20.0, 23.0, 26.0, 30.0])
    nodes = np.column_stack([np.full(len(z), 20.0), np.full(len(z), 20.0), z])
    warped, med_on, med_off = warp_nodes(nodes, mapping)
    depth = normalize_nodes(warped, med_on, med_off)[:, 2]
    # each layer lands on its depth, straight lines between, outermost interval beyond
    assert np.allclose(depth, [-8.0, -6.0, -3.0, 0.0, 6.0, 12.0, 16.5, 21.0, 27.0])

    coverage = warp_skeleton(_two_dendrites(), mapping).extra["coverage"]
    assert np.allclose(coverage["depth"], [-8.0, 27.0])

    # flat bands map x, y onto themselves, so the piecewise depth inverts exactly
    flat = normalize_nodes(warped, med_on, med_off)
    assert np.allclose(unwarp_nodes(flat, mapping), nodes, atol=1e-6)

    with pytest.raises(ValueError, match="depths"):
        build_mapping(on, off, [0, 39, 0, 39], extra_surfaces={"x": (on, 1.0)})
    with pytest.raises(ValueError, match="'on' or 'off'"):
        build_mapping(on, off, [0, 39, 0, 39], extra_surfaces={"on": (on - 5, -0.5)})


def test_extra_surface_out_of_order_is_left_out_only_where_it_is():
    from pywarper.warpers import (
        height_at_depth,
        normalize_nodes,
        relative_depth,
        warp_nodes,
    )

    on = np.full((40, 40), 10.0)
    off = np.full((40, 40), 20.0)
    deep = np.full((40, 40), 4.0)
    deep[:20] = 12.0  # above the ON band for x < 20: no longer under it
    mapping = build_mapping(
        on,
        off,
        [0, 39, 0, 39],
        conformal_jump=1,
        extra_surfaces={
            "deep": (deep, -0.5),
            "outer": (np.full((40, 40), 26.0), 1.75),
        },
    )
    nodes = np.array(
        [[10.0, 20.0, 7.0], [10.0, 20.0, 23.0], [30.0, 20.0, 7.0], [30.0, 20.0, 23.0]]
    )
    warped, med_on, med_off = warp_nodes(nodes, mapping)
    depth = normalize_nodes(warped, med_on, med_off)[:, 2]
    # x = 10: under the ON band the two bands alone, above OFF the outer layer still
    assert np.allclose(depth, [-3.6, 16.5, -3.0, 16.5])

    relative = relative_depth(nodes, mapping)
    assert np.allclose(height_at_depth(nodes[:, :2], relative, mapping), nodes[:, 2])


def test_coverage_names_the_extra_layers_that_were_not_observed():
    on = np.full((40, 40), 10.0)
    off = np.full((40, 40), 20.0)
    seen = np.ones((40, 40), dtype=bool)
    seen[20:, :] = False
    mapping = build_mapping(
        on,
        off,
        [0, 39, 0, 39],
        conformal_jump=1,
        extra_surfaces={
            "deep": (np.full((40, 40), 4.0), -0.5, seen),
            "outer": (np.full((40, 40), 26.0), 1.75),
        },
    )
    assert mapping["extra_surface_supports"].shape == (2, 40, 40)
    skel = _two_dendrites()
    warped = warp_skeleton(skel, mapping)
    coverage = warped.extra["coverage"]
    assert coverage["extra_names"].tolist() == ["deep", "outer"]
    assert np.array_equal(coverage["extra_support"][:, 0], skel.nodes[:, 0] < 20)
    assert coverage["extra_support"][:, 1].all()  # given without a mask: observed

    shares = get_coverage(warped, extent=(-8.0, 16.0))
    assert np.isclose(shares["deep_unsupported"], 40 / 60)
    assert shares["outer_unsupported"] == 0.0

    with pytest.raises(ValueError, match="mask of extra surface 'deep'"):
        build_mapping(
            on, off, [0, 39, 0, 39], extra_surfaces={"deep": (on - 6, -0.5, seen[:-1])}
        )
