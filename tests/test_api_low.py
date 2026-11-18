import numpy as np
import pytest
import scipy.io
import skeliner as sk
from scipy.spatial import KDTree

from pywarper.surface import (
    CholeskySolver,
    build_mapping,
    conformal_map_indep_fixed_diagonals,
    fit_sac_surface,
    map_distortion,
)
from pywarper.utils import build_surface_correspondences, read_sumbul_et_al_chat_bands
from pywarper.warpers import (
    get_coverage,
    get_xy_profile,
    get_z_profile,
    poly_basis_2d,
    warp_nodes,
    warp_skeleton,
    z_slice_volumes,
)


def register_node_by_node(
    nodes, top_in, bot_in, top_out, bot_out, window=5.0, max_order=2
):
    """The local least-squares registration fitted afresh around every node:
    what the registration by lattice block has to reproduce."""
    points_in = np.vstack((top_in, bot_in))
    points_out = np.vstack((top_out, bot_out))
    tree = KDTree(points_in[:, :2])
    low, high = points_in[:, :2].min(axis=0), points_in[:, :2].max(axis=0)
    moved = np.array(nodes, dtype=float)
    for k, (x, y, z) in enumerate(np.asarray(nodes, dtype=float)):
        near = np.array(tree.query_ball_point([x, y], r=window * np.sqrt(2.0)), int)
        if near.size:
            dx, dy = points_in[near, 0] - x, points_in[near, 1] - y
            near = near[(np.abs(dx) <= window) & (np.abs(dy) <= window)]
        if near.size < 12:
            continue
        # beyond the points, a fit that cannot tell the polynomial's terms
        # apart would be extrapolated
        beyond = np.any([x, y] < low) or np.any([x, y] > high)
        sides = [len(np.unique(points_in[near, axis])) for axis in (0, 1)]
        if beyond and min(sides) <= max_order:
            continue
        shift = points_in[near, :2].mean(axis=0)
        base = poly_basis_2d(
            points_in[near, 0] - shift[0], points_in[near, 1] - shift[1], max_order
        )
        design = np.hstack((base, base * points_in[near, 2:3]))
        target = np.column_stack((points_out[near, :2] - shift, points_out[near, 2]))
        transform = np.linalg.lstsq(design, target, rcond=None)[0]
        at_node = poly_basis_2d(
            np.array([x - shift[0]]), np.array([y - shift[1]]), max_order
        ).ravel()
        moved[k] = np.concatenate((at_node, z * at_node)) @ transform
        moved[k, :2] += shift
    return moved


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

    # and they are what a fit around every single node gives
    by_node = register_node_by_node(
        skel.nodes,
        *build_surface_correspondences(
            surface_mapping, conformal_jump=2, backward_compatible=True
        )[:4],
    )
    assert np.abs(warped_nodes / voxel_resolution - by_node).max() < 1e-9
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
    from skeliner import Skeleton, Soma

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


def test_warp_mesh_takes_trimesh_and_skeliner_meshes():
    """A mesh's vertices warp as nodes do, whichever kind of mesh holds them;
    the faces stay and the metadata records the warp."""
    import trimesh

    from pywarper.warpers import normalize_nodes, warp_mesh, warp_nodes

    n = 40
    x, y = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    on = 10.0 + 2.0 * np.sin(x / 9.0)
    off = on + 10.0 + np.cos(y / 11.0)
    mapping = build_mapping(
        on, off, [0, n - 1, 0, n - 1], conformal_jump=1, volume_z=(2.0, 30.0)
    )
    rng = np.random.default_rng(3)
    verts = np.column_stack(
        [rng.uniform(8, 30, 30), rng.uniform(8, 30, 30), rng.uniform(5, 25, 30)]
    )
    faces = np.arange(30).reshape(10, 3)
    expected, med_z_on, med_z_off = warp_nodes(verts, mapping)
    expected = normalize_nodes(expected, med_z_on=med_z_on, med_z_off=med_z_off)

    scale = 1e-3  # the mesh in nm, the map in um
    meshes = {
        "trimesh": trimesh.Trimesh(verts / scale, faces, process=False),
        "skeliner": sk.Mesh(verts / scale, faces),
    }
    meshes["trimesh"].metadata["source"] = "somewhere"
    for kind, mesh in meshes.items():
        warped = warp_mesh(mesh, mapping, mesh_vertices_scale=scale)
        assert isinstance(warped, trimesh.Trimesh)
        assert np.allclose(warped.vertices * scale, expected, atol=1e-9), kind
        assert np.array_equal(warped.faces, faces), kind
        assert warped.metadata["med_z_on"] == med_z_on
        assert warped.metadata["coverage"]["depth"].shape == (30, 2)
    assert (
        warp_mesh(meshes["trimesh"], mapping, mesh_vertices_scale=scale).metadata[
            "source"
        ]
        == "somewhere"
    )


def test_volume_profiles_partition_skeliners_volume():
    """The z layers and the xy bins of the volume profiles add up to the volume
    skeliner measures, and a fixed window holds what skeliner finds in its box."""
    skel = _two_dendrites()
    total = sk.dx.volume(skel, include_soma=False)
    # three straight runs of 20 at radius 1, less their overlaps
    assert 0.9 * np.pi * 60 < total < 1.1 * np.pi * 60

    z = get_z_profile(skel, measure="volume")
    assert np.isclose(z["histogram"].sum(), total, rtol=1e-12)
    assert z["extent"] == [14.0, 16.0]

    xy = get_xy_profile(skel, bin_size=5.0, measure="volume")
    assert np.isclose(xy["histogram"].sum(), total, rtol=1e-12)
    assert np.isclose(xy["distribution"].sum(), total, rtol=1e-12)
    assert xy["extents"] == pytest.approx([5.0, 35.0, 5.0, 35.0])
    # between the dendrites only the run that joins them, at y = 30
    between = (xy["x"] > 15) & (xy["x"] < 25)
    assert not xy["histogram"][between][:, xy["y"] < 25].any()
    assert xy["histogram"][between][:, xy["y"] > 25].all()

    window = [5.0, 15.0, 5.0, 35.0]
    part = get_xy_profile(skel, extents=window, bin_size=5.0, measure="volume")
    in_box = sk.dx.volume(skel, bbox=[*window, 13.0, 17.0], include_soma=False)
    assert np.isclose(part["histogram"].sum(), in_box, rtol=0.02)
    assert 0.3 * total < in_box < 0.6 * total

    # one bin over a morphology whose grid skeliner has to coarsen
    thin = _two_dendrites()
    for radii in thin.radii.values():
        radii[:] = 0.05
    one_bin = get_xy_profile(thin, bin_size=1000.0, measure="volume")
    assert one_bin["histogram"].shape == (1, 1)
    assert np.isclose(
        one_bin["histogram"].sum(), sk.dx.volume(thin, include_soma=False), rtol=1e-12
    )


def test_volume_profiles_hold_skeliners_volume_box_by_box():
    """Each voxel layer along z and each xy bin of the volume profiles holds what
    `sk.dx.volume` finds in the box of the whole grid's voxels that the layer or
    the bin takes, a bin taking the columns `np.histogram2d` puts in it."""
    skel = _two_dendrites()
    skel.nodes[21:, 2] += 1.5  # one dendrite deeper, so the layers differ
    _, grid = sk.dx.volume(skel, include_soma=False, return_details=True)
    h, shape = grid["voxel_size"], np.asarray(grid["grid_shape"])
    lo = np.asarray(grid["bbox_lo"], dtype=float)

    def in_box(start, count):
        start = lo + np.asarray(start) * h
        # the box ends inside its last voxels, so that rounding cannot add one
        end = start + (np.asarray(count) - 0.25) * h
        return sk.dx.volume(skel, bbox=(start, end), voxel_size=h, include_soma=False)

    per_layer = np.array([in_box((0, 0, k), (*shape[:2], 1)) for k in range(shape[2])])
    held = np.flatnonzero(per_layer > 0)
    volumes, z = z_slice_volumes(skel)
    assert np.array_equal(volumes, per_layer[held])
    assert np.allclose(z, lo[2] + (held + 0.5) * h, rtol=0, atol=1e-12)
    assert held.size > 1 and not np.array_equal(volumes, volumes[::-1])

    xy = get_xy_profile(skel, bin_size=5.0, measure="volume")
    blocks = []
    for axis, n_bins in enumerate(xy["n_bins"]):
        first_edge, last_edge = xy["extents"][2 * axis : 2 * axis + 2]
        edges = np.append(first_edge + np.arange(n_bins) * 5.0, last_edge)
        centres = lo[axis] + (np.arange(shape[axis]) + 0.5) * h
        inside = (centres >= edges[:-1, None]) & (centres < edges[1:, None])
        inside[-1] |= centres == edges[-1]
        blocks.append((inside.argmax(axis=1), inside.sum(axis=1)))
    (x_first, x_count), (y_first, y_count) = blocks
    expected = np.zeros_like(xy["histogram"])
    for i, j in np.ndindex(expected.shape):
        if x_count[i] and y_count[j]:
            expected[i, j] = in_box(
                (x_first[i], y_first[j], 0), (x_count[i], y_count[j], shape[2])
            )
    assert np.array_equal(xy["histogram"], expected)
    assert np.isclose(expected.sum(), grid["filled_voxels"] * h**3, rtol=1e-12)


def test_profiles_of_nothing_are_zero():
    """A window that holds none of the cell, and a skeleton without radii, give
    profiles of zeros."""
    skel = _two_dendrites()
    for measure in ("length", "volume"):
        away = get_xy_profile(
            skel, extents=[100, 140, 100, 140], bin_size=20, measure=measure
        )
        assert np.array_equal(away["distribution"], np.zeros((2, 2)))

    for radii in skel.radii.values():
        radii[:] = 0.0
    z = get_z_profile(skel, measure="volume")
    assert not z["distribution"].any() and z["hdr"] == []
    xy = get_xy_profile(skel, bin_size=20, measure="volume")
    assert not xy["distribution"].any()
    assert np.isfinite(
        warp_skeleton(
            skel,
            build_mapping(
                np.full((40, 40), 10.0),
                np.full((40, 40), 20.0),
                [0, 39, 0, 39],
                conformal_jump=1,
            ),
        ).nodes
    ).all()


def _bowl_mapping(jump: int, n: int = 60) -> dict:
    """Two curved bands over an n x n grid, the OFF band's gap varying."""
    x, y = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    on = 20.0 + 6.0 * ((x - 30) ** 2 + (y - 30) ** 2) / 900.0
    off = on + 12.0 + 1.5 * np.sin(x / 15.0)
    return build_mapping(on, off, [0, n - 1, 0, n - 1], conformal_jump=jump)


@pytest.mark.parametrize("jump", [1, 2, 3])
@pytest.mark.parametrize("backward_compatible", [False, True])
def test_registration_by_block_equals_node_by_node(jump, backward_compatible, capsys):
    """Nodes that share a block of the map's lattice share one fit, and every
    node lands where a fit of its own would put it: inside the map, on lattice
    lines, along the border, and left alone beyond it."""
    mapping = _bowl_mapping(jump)
    rng = np.random.default_rng(jump)
    inside = np.column_stack(
        [rng.uniform(6, 53, 400), rng.uniform(6, 53, 400), rng.uniform(5, 60, 400)]
    )
    on_lines = np.column_stack(
        [rng.integers(6, 53, 100), rng.integers(6, 53, 100), rng.uniform(5, 60, 100)]
    ).astype(float)
    border = np.column_stack(
        [rng.uniform(-9, 69, 600), rng.uniform(-9, 69, 600), rng.uniform(5, 60, 600)]
    )
    border = border[
        (np.abs(border[:, 0] - 29.5) > 24.5) | (np.abs(border[:, 1] - 29.5) > 24.5)
    ]
    nodes = np.vstack([inside, on_lines, border, [[200.0, 200.0, 30.0]]])
    if backward_compatible:
        nodes[:, :2] += 1.0

    reference = register_node_by_node(
        nodes,
        *build_surface_correspondences(
            mapping, conformal_jump=jump, backward_compatible=backward_compatible
        )[:4],
    )
    warped, med_z_on, med_z_off = warp_nodes(
        nodes, mapping, backward_compatible=backward_compatible
    )
    said = capsys.readouterr().out

    left_alone = np.all(reference == nodes, axis=1)
    assert 5 < left_alone.sum() < len(border)
    assert np.array_equal(warped[left_alone], nodes[left_alone])
    assert np.abs(warped - reference).max() < 1e-9
    # the fits move the nodes, and none is thrown far: not beyond the border,
    # where the lattice ends inside a node's window, either
    moved = np.linalg.norm((reference - nodes)[~left_alone, :2], axis=1)
    assert 0.3 < moved.max() < 5.0
    lattice = np.arange(0, 60, jump) + backward_compatible
    # how far inside the lattice a node is; negative beyond it
    depth = np.minimum(nodes[:, :2] - lattice[0], lattice[-1] - nodes[:, :2]).min(
        axis=1
    )
    if jump < 3:
        # just beyond the border a window still holds enough rows and columns
        assert ((depth < 0) & ~left_alone).sum() > 20
    else:
        # at this spacing it holds two: no fit beyond the border, and inside it
        # the fit of the last two rows or columns
        assert left_alone[depth < 0].all()
        assert ((depth >= 0) & (depth < 1) & ~left_alone).sum() > 5
    assert med_z_on < med_z_off

    assert said.count("no neighbours") + said.count("not enough neighbours") == (
        left_alone.sum()
    )
    last = len(nodes) - 1
    assert f"no neighbours for node {last} at (" in said


def test_two_bands_share_one_analysis_of_their_system():
    """Two surfaces on one grid, mapped with one solver, are analyzed once and
    come out as they do with a solver each; another grid is analyzed anew."""
    n = 40
    idx = np.arange(n)
    x, y = np.meshgrid(idx, idx, indexing="ij")
    on = 10.0 + 2.0 * np.sin(x / 9.0)
    off = on + 10.0 + np.cos(y / 11.0)

    def mapped(surface, solver=None, grid=idx):
        return conformal_map_indep_fixed_diagonals(
            60.0, 60.0, grid, grid, surface[np.ix_(grid, grid)], solver=solver
        )

    shared = CholeskySolver()
    together = [mapped(on, shared), mapped(off, shared)]
    assert shared.analyses == 1
    assert np.array_equal(together[0], mapped(on))
    assert np.array_equal(together[1], mapped(off))
    assert not np.allclose(together[0], together[1])

    mapped(on, shared, grid=idx[::2])
    assert shared.analyses == 2
