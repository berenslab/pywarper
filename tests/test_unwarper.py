import numpy as np

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

