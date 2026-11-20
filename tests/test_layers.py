import numpy as np
import pytest
import skeliner as sk
from scipy.spatial import KDTree
from skeliner import Skeleton, Soma

from pywarper import layers
from pywarper.layers import _column_stats, _lattice, _observed
from pywarper.warpers import node_coverage, normalize_nodes, warp_nodes

SIZE = 240.0
BOUNDS = (0.0, SIZE, 0.0, SIZE)
RADIUS = 110.0  # the tissue is a disc of this radius around the map's centre
GX, GY = np.meshgrid(np.arange(SIZE), np.arange(SIZE), indexing="ij")
R = np.hypot(GX - 120.0, GY - 120.0)
TOP = 50.0  # a top face that removes the OFF band beyond r of about 85


def on_height(x, y):
    return 22.0 + 28.0 * ((x - 120.0) ** 2 + (y - 120.0) ** 2) / 120.0**2


def off_height(x, y):
    return on_height(x, y) + 12.0 + 0.01 * (x - 120.0)


def cells_on(height, n_nodes, seed, *, top=None, n_cells=40) -> layers.Cells:
    """Nodes scattered about `height` over the tissue, 3% of them dipping below it;
    nodes above `top` do not exist."""
    rng = np.random.default_rng(seed)
    r = RADIUS * np.sqrt(rng.random(n_nodes))
    a = rng.uniform(0, 2 * np.pi, n_nodes)
    x, y = 120.0 + r * np.cos(a), 120.0 + r * np.sin(a)
    z = height(x, y) + rng.normal(0.0, 1.0, n_nodes)
    dip = rng.random(n_nodes) < 0.03
    z[dip] -= rng.uniform(3.0, 9.0, dip.sum())
    xyz, cell = np.column_stack([x, y, z]), rng.integers(0, n_cells, n_nodes)
    if top is not None:
        xyz, cell = xyz[z <= top], cell[z <= top]
    return layers.Cells(
        ids=np.array([f"{seed}-{i}" for i in range(n_cells)]),
        nodes=xyz,
        cell=cell,
        soma=np.full((n_cells, 3), np.nan),
        soma_extent=np.full(n_cells, np.nan),
    )


def rms(values):
    return float(np.sqrt(np.mean(np.square(values))))


def column_stats_by_search(xyz, xg, yg, radius):
    """The statistics of every column from a search for its nodes: what the
    statistics computed for all columns at once have to equal."""
    shape = (len(xg), len(yg))
    xx, yy = np.meshgrid(xg, yg, indexing="ij")
    centres = np.column_stack([xx.ravel(), yy.ravel()])
    quantiles = np.full((len(centres), 3), np.nan)
    count = np.zeros(len(centres), dtype=int)
    hits = KDTree(xyz[:, :2]).query_ball_point(centres, r=radius, workers=-1)
    for i, members in enumerate(hits):
        if members:
            quantiles[i] = np.percentile(xyz[members, 2], [50, 10, 90])
            count[i] = len(members)
    return {
        "median": quantiles[:, 0].reshape(shape),
        "low": quantiles[:, 1].reshape(shape),
        "high": quantiles[:, 2].reshape(shape),
        "count": count.reshape(shape),
    }


@pytest.fixture(scope="module")
def cut():
    on = cells_on(on_height, 120_000, 1, top=TOP)
    off = cells_on(off_height, 120_000, 2, top=TOP)
    bands = layers.fit_bands(on, off, bounds=BOUNDS, volume_z=(0.0, TOP))
    return bands, on, off


def test_bands_follow_the_cells_and_fill_what_a_face_removes(cut):
    bands, _, _ = cut
    inside, hidden = R < 70, (R > 97) & (R < 106)
    assert rms((bands.on.height - on_height(GX, GY))[inside]) < 0.15
    assert rms((bands.off.height - off_height(GX, GY))[inside]) < 0.15

    assert bands.on.observed[R < 100].all()
    assert bands.off.observed[inside].all()
    assert not bands.off.observed[hidden].any()
    assert bands.off.summary["columns_cut"] > 300
    assert bands.tissue[R < 100].all() and not bands.tissue[R > 120].any()

    filled = (bands.off.height - off_height(GX, GY))[hidden]
    assert abs(filled.mean()) < 0.4 and rms(filled) < 0.6

    # the true gap runs from 10.8 to 13.2; nowhere on the map do the bands approach
    gap = bands.off.height - bands.on.height
    assert 10.5 < gap.min() and gap.max() < 13.5


def test_columns_left_under_a_face_do_not_count(cut):
    bands, _, off = cut
    xg, yg = _lattice(BOUNDS, bands.spacing)
    xx, yy = np.meshgrid(xg, yg, indexing="ij")
    stats = _column_stats(off.nodes, xg, yg, bands.spacing)
    # leftovers under the face: a median well clear of the face, yet far off the band
    stray = (
        (stats["count"] > 0)
        & (stats["median"] <= TOP - 3.0)
        & (stats["median"] < off_height(xx, yy) - 3.0)
    )
    assert stray.sum() > 20
    assert not (stray & _observed(stats, (0.0, TOP), 1.5, 0.1)[0]).any()

    # so the band filled in under the face matches the one seen without a face
    whole = layers.fit_bands(
        cells_on(on_height, 120_000, 1),
        cells_on(off_height, 120_000, 2),
        bounds=BOUNDS,
        volume_z=(0.0, 90.0),
    )
    hidden = (R > 97) & (R < 106)
    assert whole.off.observed[hidden].all()
    assert abs((bands.off.height - whole.off.height)[hidden].mean()) < 0.3


def test_column_statistics_equal_a_search_per_column():
    """Every column's count and percentiles are those of the nodes within the
    spacing of it: for nodes anywhere, on lattice lines, on lattice points and
    beyond the lattice, in columns of one node and of none."""
    spacing = 5.0
    xg, yg = _lattice((0.0, 60.0, 0.0, 45.0), spacing)
    rng = np.random.default_rng(7)
    anywhere = np.column_stack([rng.uniform(-8, 68, 6000), rng.uniform(-8, 53, 6000)])
    on_lines = np.column_stack(
        [spacing * rng.integers(0, 13, 300), rng.uniform(0, 45, 300)]
    )
    on_points = spacing * np.column_stack(
        [rng.integers(0, 13, 200), rng.integers(0, 10, 200)]
    )
    xy = np.vstack([anywhere, on_lines, on_lines[:, ::-1] * [1.0, 0.7], on_points])
    # a strip without nodes, but for one
    xy = np.vstack([xy[(xy[:, 0] < 22) | (xy[:, 0] > 43)], [[32.6, 21.0]]])
    xyz = np.column_stack([xy, rng.normal(30.0, 4.0, len(xy))])

    stats = _column_stats(xyz, xg, yg, spacing)
    expected = column_stats_by_search(xyz, xg, yg, spacing)
    assert np.array_equal(stats["count"], expected["count"])
    for name in ("median", "low", "high"):
        assert np.array_equal(stats[name], expected[name], equal_nan=True)
    assert (stats["count"] == 0).sum() > 5 and (stats["count"] == 1).sum() >= 1
    # a node on a lattice point is in the four columns one step from it too
    assert stats["count"].sum() > len(xyz) * 2

    empty = _column_stats(xyz[:0], xg, yg, spacing)
    assert not empty["count"].any() and np.isnan(empty["median"]).all()


def test_bands_refuse_cells_filed_twice_and_bands_that_cross(cut):
    _, on, off = cut
    with pytest.raises(ValueError, match="both the ON and the OFF"):
        layers.fit_bands(on, on, bounds=BOUNDS)
    with pytest.raises(ValueError, match="start at 0"):
        layers.fit_bands(on, off, bounds=(10.0, SIZE, 0.0, SIZE))


def drifting(x, y):
    """A layer 2.6 to 7.4 um under the ON band, drifting across the tissue."""
    return on_height(x, y) - (5.0 + 2.2 * (x - 120.0) / 100.0)


def test_stratum_layer_follows_a_drifting_layer(cut):
    bands, _, _ = cut
    deep = cells_on(drifting, 60_000, 3, top=TOP)
    truth = drifting(GX, GY)
    kept = {}
    for passes in (1, 2):
        layer = layers.stratum_layer("deep", deep, bands, passes=passes)
        where = bands.tissue & layer.observed & (R < 100)
        kept[passes] = np.polyfit(
            (truth - bands.on.height)[where], (layer.height - bands.on.height)[where], 1
        )[0]
    assert kept[2] > kept[1] and kept[2] > 0.97
    assert rms((layer.height - truth)[where]) < 0.15
    assert layer.observed[R < 100].all()
    assert layer.depth == pytest.approx(-5.0, abs=0.2)
    assert layer.summary["kind"] == "stratum" and layer.summary["cells"] == 40
    assert layer.summary["profile_peak"] == pytest.approx(-5.0, abs=0.3)

    with pytest.raises(ValueError, match="a face of the volume cuts it"):
        layers.stratum_layer("cut", deep, bands, clip_margin=TOP)


def flat_stratum(x, y):
    """A layer at flattened depth -5 everywhere."""
    return on_height(x, y) - 5.0 * (off_height(x, y) - on_height(x, y)) / 12.0


def test_stratum_layer_finds_its_stratum_in_the_cells_depth_profile(cut):
    bands, _, _ = cut
    flat = cells_on(flat_stratum, 60_000, 7, top=TOP)
    layer = layers.stratum_layer("flat", flat, bands)
    scatter = np.hypot(1.0, layers._PROFILE_SMOOTH)  # the nodes' 1 um, smoothed
    hwhm = np.sqrt(2.0 * np.log(2.0)) * scatter
    assert layer.summary["profile_peak"] == pytest.approx(-5.0, abs=0.1)
    assert layer.summary["profile_half_width"] == pytest.approx(hwhm, abs=0.1)
    assert layer.depth == pytest.approx(-5.0, abs=0.1)
    with pytest.raises(ValueError, match="no nodes"):
        layers.stratum_layer("empty", flat.subset(np.zeros(len(flat), bool)), bands)


def outer(x, y):
    return off_height(x, y) + 6.0


def test_stratum_cut_by_a_face_is_pinned_where_it_is_not_seen(cut):
    bands, _, _ = cut
    layer = layers.stratum_layer("outer", cells_on(outer, 60_000, 4, top=TOP), bands)
    away = layer.height - outer(GX, GY)
    assert layer.observed[R < 50].all()
    assert not layer.observed[(R > 75) & bands.tissue].any()
    assert layer.summary["columns_cut"] > 100
    assert rms(away[layer.observed]) < 0.15
    unseen = (R > 75) & (R < 105)
    assert abs(away[unseen].mean()) < 0.3 and np.abs(away[unseen]).max() < 1.2
    assert layer.depth == pytest.approx(18.0, abs=0.3)


def somata_under(top, seed=5, n=1500):
    """Somata 8 +- 1.5 um above the OFF band, 3 um tall; one cut by the face is lost."""
    rng = np.random.default_rng(seed)
    r = RADIUS * np.sqrt(rng.random(n))
    a = rng.uniform(0, 2 * np.pi, n)
    x, y = 120.0 + r * np.cos(a), 120.0 + r * np.sin(a)
    offset = rng.normal(8.0, 1.5, n)
    xyz = np.column_stack([x, y, off_height(x, y) + offset])
    whole = xyz[:, 2] + 3.0 <= top
    cells = layers.Cells(
        ids=np.array([f"s{i}" for i in range(n)]),
        nodes=xyz.copy(),
        cell=np.arange(n),
        soma=np.where(whole[:, None], xyz, np.nan),
        soma_extent=np.where(whole, 3.0, np.nan),
    )
    return cells, r, offset, whole


def test_soma_layer_is_sampled_by_place():
    top = 60.0
    bands = layers.fit_bands(
        cells_on(on_height, 120_000, 1, top=top),
        cells_on(off_height, 120_000, 2, top=top),
        bounds=BOUNDS,
        volume_z=(0.0, top),
    )
    cells, r, offset, whole = somata_under(top)
    # one soma on the wrong side of the bands is not this layer
    cells.soma[0] = [120.0, 120.0, on_height(120.0, 120.0) - 15.0]
    cells.soma_extent[0] = 3.0
    layer = layers.soma_layer("somata", cells, bands)

    assert layer.summary["samples_off_level"] == 1
    assert layer.summary["samples_without_room"] > 100
    assert layer.depth == pytest.approx(12.0 * (1.0 + 8.0 / 12.0), abs=0.4)
    assert layer.observed[R < 65].all() and not layer.observed[R > 85].any()

    # where the face leaves room for only some somata, the survivors are the low
    # ones; the layer keeps to what it saw where every soma fits
    short = (r > 82) & (r < 97)
    assert np.median(offset[short & whole]) < 7.6
    above = (layer.height - bands.off.height)[(R > 82) & (R < 97)]
    assert above.mean() == pytest.approx(8.0, abs=0.5)

    scored = layers.cross_validate("soma", cells, bands)
    assert scored["layer"]["n"] == scored["bands"]["n"] > 100
    assert scored["layer"]["depth_sd"] <= scored["bands"]["depth_sd"]
    assert abs(scored["layer"]["per_100um_x"]) < abs(scored["bands"]["per_100um_x"])

    with pytest.raises(ValueError, match="three cells with a soma"):
        layers.soma_layer("none", cells_on(on_height, 1000, 9), bands)


def test_cross_validate_scores_a_stratum_on_held_out_cells(cut):
    bands, _, _ = cut
    deep = cells_on(drifting, 60_000, 3, top=TOP)
    scored = layers.cross_validate("stratum", deep, bands)
    assert scored["level"] == pytest.approx(-5.0, abs=0.2)
    assert scored["bands"]["depth_sd"] > 1.0  # the drift, in flattened depth
    assert scored["layer"]["depth_sd"] < 0.3
    assert (
        abs(scored["layer"]["per_100um_x"]) < 0.1 < abs(scored["bands"]["per_100um_x"])
    )
    with pytest.raises(ValueError, match="'soma' or 'stratum'"):
        layers.cross_validate("band", deep, bands)


def test_check_cells_marks_the_cell_that_is_off_its_layer(cut):
    bands, on, _ = cut
    assert not layers.check_cells(on, bands.on)["off_layer"].any()
    nodes = on.nodes.copy()
    nodes[on.cell == 7, 2] += 12.0
    moved = layers.Cells(on.ids, nodes, on.cell, on.soma, on.soma_extent)
    table = layers.check_cells(moved, bands.on)
    assert table["ids"][table["off_layer"]].tolist() == [on.ids[7]]
    assert table["share"][7] < 0.1 < table["share"][8]


def test_map_carries_the_layers_through_a_warp_and_a_file(cut, tmp_path):
    bands, _, _ = cut
    deep = layers.stratum_layer("deep", cells_on(drifting, 60_000, 3, top=TOP), bands)
    far = layers.stratum_layer("outer", cells_on(outer, 60_000, 4, top=TOP), bands)
    mapping = layers.build_map(bands, [deep, far], conformal_jump=2)
    assert mapping["provenance"]["out_of_order_share_of_tissue"] == {
        "deep": 0.0,
        "outer": 0.0,
    }
    assert mapping["provenance"]["layers"]["deep"]["kind"] == "stratum"

    # a node on a layer lands on the layer's depth
    x = np.array([70.0, 120.0, 150.0, 185.0])
    y = np.array([120.0, 100.0, 140.0, 120.0])
    for layer in (deep, far):
        z = layer.height[x.astype(int), y.astype(int)]
        warped, med_on, med_off = warp_nodes(np.column_stack([x, y, z]), mapping)
        depth = normalize_nodes(warped, med_on, med_off)[:, 2]
        assert np.allclose(depth, layer.depth, atol=1e-6)

    # x = 210 lies where the face hides both the OFF band and the outer stratum
    coverage = node_coverage(
        np.array([[120.0, 120.0, 30.0], [210.0, 120.0, 40.0]]), mapping
    )
    assert coverage["extra_names"].tolist() == ["deep", "outer"]
    assert coverage["extra_support"].tolist() == [[True, True], [True, False]]
    assert coverage["off_support"].tolist() == [True, False]

    path = tmp_path / "global_map_j2_test.npz"
    layers.save_map(mapping, path)
    with np.load(path, allow_pickle=False) as stored:
        assert "extra_surface_supports" in stored.files
    again = layers.load_map(path)
    assert again["provenance"] == mapping["provenance"]
    assert again["conformal_jump"] == 2 and again["meta"]["pywarper_version"]
    nodes = np.column_stack([x, y, on_height(x, y) + 3.0])
    assert np.array_equal(warp_nodes(nodes, again)[0], warp_nodes(nodes, mapping)[0])

    layers.save_layers(bands, [deep, far], tmp_path / "layers.npz")
    bands_again, extra = layers.load_layers(tmp_path / "layers.npz")
    assert [layer.name for layer in extra] == ["deep", "outer"]
    assert np.array_equal(bands_again.off.height, bands.off.height)
    assert np.array_equal(extra[1].observed, far.observed)
    assert bands_again.volume_z == bands.volume_z and extra[0].depth == deep.depth

    text = layers.report(bands, [deep, far], mapping)
    assert "outer: depth +18" in text and "length inside the tissue" in text

    with pytest.raises(ValueError, match="share a name"):
        layers.build_map(bands, [deep, deep])


# ------------------------------------------------------------------ loading ---


def skeleton(x, *, unit="nm", scale=1000.0, soma=True, **meta) -> Skeleton:
    """A short dendrite in the EW2 volume at x (um), stored in `unit`."""
    nodes = np.column_stack(
        [np.full(12, x), np.linspace(500.0, 511.0, 12), np.full(12, 40.0)]
    )
    nodes = nodes * scale
    radii = np.full(len(nodes), 0.5 * scale)
    ntype = np.zeros(len(nodes), dtype=int)
    if soma:
        ntype[0] = 1
    if unit is not None:
        meta["unit"] = unit
    return Skeleton(
        soma=Soma(
            center=nodes[0].copy(),
            axes=np.array([4.0, 3.0, 2.0]) * scale,
            R=np.eye(3),
            verts=None,
        ),
        nodes=nodes,
        radii={"median": radii, "mean": radii.copy(), "trim": radii.copy()},
        edges=np.column_stack([np.arange(len(nodes) - 1), np.arange(1, len(nodes))]),
        ntype=ntype,
        node2verts=[np.array([i]) for i in range(len(nodes))],
        vert2node={i: i for i in range(len(nodes))},
        meta=meta,
    )


def test_load_cells_reads_whatever_layout_the_pattern_names(tmp_path):
    flat, nested = tmp_path / "flat", tmp_path / "nested"
    flat.mkdir()
    for i, cell in enumerate(("101", "102", "103")):
        sk.io.save_skeleton(
            skeleton(300.0 + 10 * i, soma=cell != "103"), flat / f"{cell}.npz"
        )
        (nested / cell).mkdir(parents=True)
        sk.io.save_skeleton(skeleton(300.0 + 10 * i), nested / cell / "skeleton.npz")
    (flat / "notes.txt").write_text("not a skeleton")

    cells = layers.load_cells(flat, "{id}.npz")
    assert cells.ids.tolist() == ["101", "102", "103"]
    assert np.allclose(cells.nodes[cells.cell == 1, 0], 310.0)  # nm on file, um here
    assert np.allclose(cells.soma[0], [300.0, 500.0, 40.0])
    assert cells.soma_extent[0] == pytest.approx(2.0)  # the ellipsoid's reach along z
    assert np.isnan(cells.soma[2]).all() and np.isnan(cells.soma_extent[2])
    assert cells.summary["without_soma"] == 1 and cells.summary["units"] == {"nm": 3}

    same = layers.load_cells(nested, "{id}/skeleton.npz", ids=[103, "101"])
    assert same.ids.tolist() == ["101", "103"] and same.summary["not_listed"] == 1
    with pytest.raises(FileNotFoundError, match="1 listed cells have no file.*999"):
        layers.load_cells(nested, "{id}/skeleton.npz", ids=["101", "999"])
    with pytest.raises(FileNotFoundError, match="no file matches"):
        layers.load_cells(nested, "{id}.npz")
    with pytest.raises(ValueError, match="exactly once"):
        layers.load_cells(nested, "skeleton.npz")

    in_memory = layers.Cells.from_skeletons(
        {c: sk.io.load_skeleton(flat / f"{c}.npz") for c in ("101", "102", "103")}
    )
    assert np.array_equal(in_memory.nodes, cells.nodes)
    assert np.array_equal(in_memory.cell, cells.cell)
    assert np.array_equal(in_memory.soma, cells.soma, equal_nan=True)
    assert np.array_equal(in_memory.soma_extent, cells.soma_extent, equal_nan=True)

    with pytest.raises(ValueError, match="more than one"):
        layers.Cells.merge(cells, same)
    pooled = layers.Cells.merge(cells.subset(np.array([False, True, False])), same)
    assert pooled.ids.tolist() == ["102", "101", "103"]
    assert np.allclose(pooled.nodes[pooled.cell == 2, 0], 320.0)


def test_cells_are_checked_before_they_are_used():
    um = {"scale": 1.0}
    # both spellings of the micro sign are micrometres
    for unit in ("um", "µm", "μm"):
        cells = layers.Cells.from_skeletons({"1": skeleton(300.0, unit=unit, **um)})
        assert np.allclose(cells.nodes[:, 0], 300.0)

    with pytest.raises(ValueError, match="carries no unit"):
        layers.Cells.from_skeletons({"1": skeleton(300.0, unit=None, **um)})
    assert len(
        layers.Cells.from_skeletons({"1": skeleton(300.0, unit=None, **um)}, unit="um")
    )
    with pytest.raises(ValueError, match="is in nm, not in um"):
        layers.Cells.from_skeletons({"1": skeleton(300.0)}, unit="um")
    with pytest.raises(ValueError, match="neither nm nor um"):
        layers.Cells.from_skeletons({"1": skeleton(300.0, unit="voxel")})

    with pytest.raises(ValueError, match="flattened skeleton"):
        layers.Cells.from_skeletons(
            {"1": skeleton(300.0, warped_at="2026-03-23 15:23:46")}
        )
    # nanometres labelled as micrometres put the cell far outside the volume
    with pytest.raises(ValueError, match="outside the EW2 volume"):
        layers.Cells.from_skeletons({"1": skeleton(300.0, unit="um")})
    with pytest.raises(ValueError, match="filed as cell 1 but names cell 2"):
        layers.Cells.from_skeletons({"1": skeleton(300.0, id=2)})
    assert len(layers.Cells.from_skeletons({"2": skeleton(300.0, id=2)})) == 1
