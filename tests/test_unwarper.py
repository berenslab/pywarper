import numpy as np
import pytest
from skeliner.dataclass import Skeleton, Soma

from pywarper.unwarper import denormalize_nodes, unwarp_nodes, unwarp_skeleton
from pywarper.warpers import normalize_nodes, warp_nodes, warp_skeleton


def _identity_surface_mapping(nx: int = 25, ny: int = 25) -> dict:
    on_sac_surface = np.zeros((nx, ny), dtype=float)
    off_sac_surface = np.full((nx, ny), 10.0, dtype=float)

    sampled_x_idx = np.arange(nx, dtype=int)
    sampled_y_idx = np.arange(ny, dtype=int)
    xmesh, ymesh = np.meshgrid(sampled_x_idx, sampled_y_idx, indexing="ij")

    mapped_xy = np.column_stack([xmesh.ravel(order="F"), ymesh.ravel(order="F")])
    return {
        "mapped_on": mapped_xy.copy(),
        "mapped_off": mapped_xy.copy(),
        "on_sac_surface": on_sac_surface,
        "off_sac_surface": off_sac_surface,
        "sampled_x_idx": sampled_x_idx,
        "sampled_y_idx": sampled_y_idx,
        "conformal_jump": 1,
    }


def _nontrivial_surface_mapping(nx: int = 35, ny: int = 35) -> dict:
    on_sac_surface = np.zeros((nx, ny), dtype=float)
    off_sac_surface = np.full((nx, ny), 10.0, dtype=float)

    sampled_x_idx = np.arange(nx, dtype=int)
    sampled_y_idx = np.arange(ny, dtype=int)
    xmesh, ymesh = np.meshgrid(sampled_x_idx, sampled_y_idx, indexing="ij")

    x = xmesh.ravel(order="F").astype(float)
    y = ymesh.ravel(order="F").astype(float)

    mapped_on = np.column_stack(
        [
            x + 0.15 * np.sin(y / 7.0),
            y + 0.15 * np.sin(x / 8.0),
        ]
    )
    mapped_off = np.column_stack(
        [
            x + 0.35 * np.sin(y / 7.0 + 0.1),
            y + 0.35 * np.sin(x / 8.0 - 0.1),
        ]
    )
    return {
        "mapped_on": mapped_on,
        "mapped_off": mapped_off,
        "on_sac_surface": on_sac_surface,
        "off_sac_surface": off_sac_surface,
        "sampled_x_idx": sampled_x_idx,
        "sampled_y_idx": sampled_y_idx,
        "conformal_jump": 1,
    }


def _test_skeleton(n_nodes: int = 60, seed: int = 11) -> Skeleton:
    rng = np.random.default_rng(seed)
    nodes = np.empty((n_nodes, 3), dtype=float)
    nodes[:, 0] = rng.uniform(6.0, 18.0, size=n_nodes)
    nodes[:, 1] = rng.uniform(6.0, 18.0, size=n_nodes)
    nodes[:, 2] = rng.uniform(0.0, 10.0, size=n_nodes)

    edges = np.column_stack(
        [np.arange(0, n_nodes - 1, dtype=int), np.arange(1, n_nodes, dtype=int)]
    )
    radii_values = rng.uniform(0.1, 1.2, size=n_nodes).astype(float)
    radii = {
        "median": radii_values.copy(),
        "mean": radii_values.copy(),
        "trim": radii_values.copy(),
    }
    ntype = np.ones(n_nodes, dtype=int)
    soma = Soma(
        center=nodes[0].copy(),
        axes=np.array([1.0, 1.0, 1.0], dtype=float),
        R=np.eye(3, dtype=float),
        verts=None,
    )
    node2verts = [np.array([i], dtype=int) for i in range(n_nodes)]
    vert2node = {int(i): int(i) for i in range(n_nodes)}

    return Skeleton(
        soma=soma,
        nodes=nodes,
        radii=radii,
        edges=edges,
        ntype=ntype,
        node2verts=node2verts,
        vert2node=vert2node,
        meta={"source": "toy_skeleton"},
        extra={"tag": "toy"},
    )


def test_denormalize_nodes_inverts_normalize_nodes():
    rng = np.random.default_rng(0)
    nodes = rng.uniform(-10.0, 10.0, size=(100, 3))
    median_depths = {"on_sac": -2.5, "off_sac": 21.0}
    on_sac_pos, off_sac_pos = 0.0, 12.0

    normalized = normalize_nodes(
        nodes,
        median_depths=median_depths,
        anchor_pos=(on_sac_pos, off_sac_pos),
    )
    restored = denormalize_nodes(
        normalized,
        median_depths=median_depths,
        anchor_pos=(on_sac_pos, off_sac_pos),
    )

    assert np.allclose(restored, nodes, rtol=1e-12, atol=1e-12)


def test_unwarp_nodes_roundtrip_with_identity_surface_mapping():
    rng = np.random.default_rng(1)
    mapping = _identity_surface_mapping()

    nodes = np.empty((200, 3), dtype=float)
    nodes[:, 0] = rng.uniform(6.0, 18.0, size=200)
    nodes[:, 1] = rng.uniform(6.0, 18.0, size=200)
    nodes[:, 2] = rng.uniform(0.0, 10.0, size=200)

    warped, median_depths = warp_nodes(nodes, mapping)
    normalized = normalize_nodes(
        warped,
        median_depths=median_depths,
        anchor_pos=(0.0, 12.0),
    )

    recovered = unwarp_nodes(
        normalized,
        mapping,
    )

    assert np.allclose(recovered, nodes, rtol=1e-6, atol=1e-6)


def test_unwarp_nodes_can_infer_median_depths_from_mapping():
    rng = np.random.default_rng(3)
    mapping = _identity_surface_mapping()

    nodes = np.empty((80, 3), dtype=float)
    nodes[:, 0] = rng.uniform(6.0, 18.0, size=80)
    nodes[:, 1] = rng.uniform(6.0, 18.0, size=80)
    nodes[:, 2] = rng.uniform(0.0, 10.0, size=80)

    warped, median_depths = warp_nodes(nodes, mapping)
    normalized = normalize_nodes(
        warped,
        median_depths=median_depths,
        anchor_pos=(0.0, 12.0),
    )

    # med_z inferred from surface_mapping.
    recovered = unwarp_nodes(
        normalized,
        mapping,
    )

    assert np.allclose(recovered, nodes, rtol=1e-6, atol=1e-6)


def test_unwarp_nodes_supports_custom_sac_positions():
    rng = np.random.default_rng(4)
    mapping = _identity_surface_mapping()

    nodes = np.empty((120, 3), dtype=float)
    nodes[:, 0] = rng.uniform(6.0, 18.0, size=120)
    nodes[:, 1] = rng.uniform(6.0, 18.0, size=120)
    nodes[:, 2] = rng.uniform(0.0, 10.0, size=120)

    warped, median_depths = warp_nodes(nodes, mapping)
    on_sac_pos, off_sac_pos = -3.0, 21.5
    normalized = normalize_nodes(
        warped,
        median_depths=median_depths,
        anchor_pos=(on_sac_pos, off_sac_pos),
    )

    recovered_local = unwarp_nodes(
        normalized,
        mapping,
        on_sac_pos=on_sac_pos,
        off_sac_pos=off_sac_pos,
    )
    recovered_opt = unwarp_nodes(
        normalized,
        mapping,
        method="optimize",
        on_sac_pos=on_sac_pos,
        off_sac_pos=off_sac_pos,
        max_evals_per_point=60,
        convergence_tol=1e-9,
    )

    assert np.allclose(recovered_local, nodes, rtol=1e-6, atol=1e-6)
    assert np.allclose(recovered_opt, nodes, rtol=1e-6, atol=1e-6)


def test_unwarp_nodes_optimize_refines_nontrivial_mapping():
    rng = np.random.default_rng(42)
    mapping = _nontrivial_surface_mapping()

    nodes = np.empty((80, 3), dtype=float)
    nodes[:, 0] = rng.uniform(8.0, 27.0, size=80)
    nodes[:, 1] = rng.uniform(8.0, 27.0, size=80)
    # allow points far beyond ON/OFF layer z to match annotation use-cases
    nodes[:, 2] = rng.uniform(-120.0, 180.0, size=80)

    warped, median_depths = warp_nodes(nodes, mapping)
    normalized = normalize_nodes(
        warped,
        median_depths=median_depths,
        anchor_pos=(0.0, 12.0),
    )

    recovered_local = unwarp_nodes(
        normalized,
        mapping,
    )
    recovered_opt = unwarp_nodes(
        normalized,
        mapping,
        method="optimize",
        max_evals_per_point=80,
        convergence_tol=1e-9,
        bound_xy_to_map=True,
    )

    local_err = np.linalg.norm(recovered_local - nodes, axis=1)
    opt_err = np.linalg.norm(recovered_opt - nodes, axis=1)

    assert opt_err.mean() < local_err.mean() * 1e-3
    assert np.quantile(opt_err, 0.95) < 1e-6

    rewarped_opt, _ = warp_nodes(recovered_opt, mapping)
    renorm_opt = normalize_nodes(
        rewarped_opt,
        median_depths=median_depths,
        anchor_pos=(0.0, 12.0),
    )
    assert np.allclose(renorm_opt, normalized, rtol=1e-8, atol=1e-8)


def test_unwarp_nodes_optimize_invalid_params_raise():
    mapping = _identity_surface_mapping()
    nodes = np.array([[10.0, 10.0, 5.0]], dtype=float)
    warped, median_depths = warp_nodes(nodes, mapping)
    normalized = normalize_nodes(
        warped,
        median_depths=median_depths,
        anchor_pos=(0.0, 12.0),
    )

    with pytest.raises(ValueError, match="max_evals_per_point"):
        unwarp_nodes(
            normalized,
            mapping,
            method="optimize",
            max_evals_per_point=0,
        )

    with pytest.raises(ValueError, match="convergence_tol"):
        unwarp_nodes(
            normalized,
            mapping,
            method="optimize",
            convergence_tol=0.0,
        )


def test_unwarp_skeleton_roundtrip_with_identity_surface_mapping():
    mapping = _identity_surface_mapping()
    skel = _test_skeleton(seed=7)
    voxel_resolution = np.array([0.4, 0.4, 0.5], dtype=float)

    warped = warp_skeleton(
        skel,
        mapping,
        voxel_resolution=voxel_resolution,
        on_sac_pos=0.0,
        off_sac_pos=12.0,
    )
    recovered = unwarp_skeleton(
        warped,
        mapping,
        voxel_resolution=voxel_resolution,
        on_sac_pos=0.0,
        off_sac_pos=12.0,
    )

    assert np.allclose(recovered.nodes, skel.nodes, rtol=1e-6, atol=1e-6)
    assert np.allclose(recovered.soma.center, skel.nodes[0], rtol=1e-6, atol=1e-6)
    assert np.array_equal(recovered.edges, skel.edges)
    assert np.array_equal(recovered.ntype, skel.ntype)
    assert recovered.vert2node == skel.vert2node
    assert recovered.meta == warped.meta
    assert recovered.extra.keys() == warped.extra.keys()
    assert np.allclose(
        recovered.extra["prenormed_nodes"], warped.extra["prenormed_nodes"]
    )
    assert recovered.extra["med_z_on"] == warped.extra["med_z_on"]
    assert recovered.extra["med_z_off"] == warped.extra["med_z_off"]
    for key in ("median", "mean", "trim"):
        assert np.array_equal(recovered.radii[key], skel.radii[key])
    for i in range(len(skel.node2verts)):
        assert np.array_equal(recovered.node2verts[i], skel.node2verts[i])


def test_unwarp_skeleton_optimize_roundtrip_with_scaling():
    rng = np.random.default_rng(17)
    mapping = _nontrivial_surface_mapping()
    skel = _test_skeleton(n_nodes=50, seed=13)
    skel.nodes[:, 0] = rng.uniform(5.0, 12.0, size=len(skel.nodes))
    skel.nodes[:, 1] = rng.uniform(5.0, 12.0, size=len(skel.nodes))
    skel.nodes[:, 2] = rng.uniform(-120.0, 180.0, size=len(skel.nodes))
    skel.soma.center = skel.nodes[0].copy()

    voxel_resolution = np.array([0.4, 0.4, 0.5], dtype=float)
    on_sac_pos, off_sac_pos = -3.0, 21.5
    skeleton_nodes_scale = 2.0

    warped = warp_skeleton(
        skel,
        mapping,
        voxel_resolution=voxel_resolution,
        on_sac_pos=on_sac_pos,
        off_sac_pos=off_sac_pos,
        skeleton_nodes_scale=skeleton_nodes_scale,
    )
    recovered = unwarp_skeleton(
        warped,
        mapping,
        voxel_resolution=voxel_resolution,
        on_sac_pos=on_sac_pos,
        off_sac_pos=off_sac_pos,
        skeleton_nodes_scale=skeleton_nodes_scale,
        method="optimize",
        max_evals_per_point=80,
        convergence_tol=1e-9,
        bound_xy_to_map=True,
    )

    err = np.linalg.norm(recovered.nodes - skel.nodes, axis=1)
    assert np.quantile(err, 0.95) < 1e-6

    rewarped = warp_skeleton(
        recovered,
        mapping,
        voxel_resolution=voxel_resolution,
        on_sac_pos=on_sac_pos,
        off_sac_pos=off_sac_pos,
        skeleton_nodes_scale=skeleton_nodes_scale,
    )
    assert np.allclose(rewarped.nodes, warped.nodes, rtol=1e-8, atol=1e-8)
