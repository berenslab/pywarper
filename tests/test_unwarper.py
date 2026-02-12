import numpy as np
import pytest

from pywarper.unwarper import denormalize_nodes, unwarp_nodes
from pywarper.warpers import normalize_nodes, warp_nodes


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


def test_denormalize_nodes_inverts_normalize_nodes():
    rng = np.random.default_rng(0)
    nodes = rng.uniform(-10.0, 10.0, size=(100, 3))
    med_z_on, med_z_off = -2.5, 21.0
    on_sac_pos, off_sac_pos = 0.0, 12.0

    normalized = normalize_nodes(
        nodes,
        med_z_on=med_z_on,
        med_z_off=med_z_off,
        on_sac_pos=on_sac_pos,
        off_sac_pos=off_sac_pos,
    )
    restored = denormalize_nodes(
        normalized,
        med_z_on=med_z_on,
        med_z_off=med_z_off,
        on_sac_pos=on_sac_pos,
        off_sac_pos=off_sac_pos,
    )

    assert np.allclose(restored, nodes, rtol=1e-12, atol=1e-12)


def test_unwarp_nodes_roundtrip_with_identity_surface_mapping():
    rng = np.random.default_rng(1)
    mapping = _identity_surface_mapping()

    nodes = np.empty((200, 3), dtype=float)
    nodes[:, 0] = rng.uniform(6.0, 18.0, size=200)
    nodes[:, 1] = rng.uniform(6.0, 18.0, size=200)
    nodes[:, 2] = rng.uniform(0.0, 10.0, size=200)

    warped, med_z_on, med_z_off = warp_nodes(nodes, mapping)
    normalized = normalize_nodes(
        warped,
        med_z_on=med_z_on,
        med_z_off=med_z_off,
        on_sac_pos=0.0,
        off_sac_pos=12.0,
    )

    recovered = unwarp_nodes(
        normalized,
        mapping,
        med_z_on=med_z_on,
        med_z_off=med_z_off,
        on_sac_pos=0.0,
        off_sac_pos=12.0,
    )
    recovered_from_prenormed = unwarp_nodes(
        warped,
        mapping,
        med_z_on=med_z_on,
        med_z_off=med_z_off,
        prenormalized=True,
    )

    assert np.allclose(recovered, nodes, rtol=1e-6, atol=1e-6)
    assert np.allclose(recovered_from_prenormed, nodes, rtol=1e-6, atol=1e-6)


def test_unwarp_nodes_optimize_refines_nontrivial_mapping():
    rng = np.random.default_rng(42)
    mapping = _nontrivial_surface_mapping()

    nodes = np.empty((80, 3), dtype=float)
    nodes[:, 0] = rng.uniform(8.0, 27.0, size=80)
    nodes[:, 1] = rng.uniform(8.0, 27.0, size=80)
    # allow points far beyond ON/OFF layer z to match annotation use-cases
    nodes[:, 2] = rng.uniform(-120.0, 180.0, size=80)

    warped, med_z_on, med_z_off = warp_nodes(nodes, mapping)

    recovered_local = unwarp_nodes(
        warped,
        mapping,
        med_z_on=med_z_on,
        med_z_off=med_z_off,
        prenormalized=True,
        method="local_ls",
    )
    recovered_opt = unwarp_nodes(
        warped,
        mapping,
        med_z_on=med_z_on,
        med_z_off=med_z_off,
        prenormalized=True,
        method="optimize",
        optimize_max_evals_per_point=80,
        optimize_convergence_tol=1e-9,
        optimize_bound_xy_to_map=True,
    )

    local_err = np.linalg.norm(recovered_local - nodes, axis=1)
    opt_err = np.linalg.norm(recovered_opt - nodes, axis=1)

    assert opt_err.mean() < local_err.mean() * 1e-3
    assert np.quantile(opt_err, 0.95) < 1e-6

    rewarped_opt, _, _ = warp_nodes(recovered_opt, mapping)
    assert np.allclose(rewarped_opt, warped, rtol=1e-8, atol=1e-8)


def test_unwarp_nodes_invalid_method_raises():
    mapping = _identity_surface_mapping()
    nodes = np.array([[10.0, 10.0, 5.0]], dtype=float)
    warped, med_z_on, med_z_off = warp_nodes(nodes, mapping)

    with pytest.raises(ValueError, match="method"):
        unwarp_nodes(
            warped,
            mapping,
            med_z_on=med_z_on,
            med_z_off=med_z_off,
            prenormalized=True,
            method="unknown",
        )
