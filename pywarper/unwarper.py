"""Inverse mapping helpers for pywarper."""

from __future__ import annotations

import numpy as np

from .utils import build_surface_correspondences, resolve_conformal_jump
from .warpers import local_ls_registration


def denormalize_nodes(
    nodes: np.ndarray,
    med_z_on: float,
    med_z_off: float,
    on_sac_pos: float = 0.0,
    off_sac_pos: float = 12.0,
) -> np.ndarray:
    """
    Undo `normalize_nodes` and map z back to the pre-normalized warped frame.

    Parameters
    ----------
    nodes : np.ndarray
        (N, 3) normalized [x, y, z] coordinates.
    med_z_on : float
        Median z-value of the ON SAC surface used during warping.
    med_z_off : float
        Median z-value of the OFF SAC surface used during warping.
    on_sac_pos : float, default=0.0
        ON surface position used in normalized space.
    off_sac_pos : float, default=12.0
        OFF surface position used in normalized space.

    Returns
    -------
    np.ndarray
        (N, 3) coordinates in the pre-normalized warped frame.
    """
    nodes = np.asarray(nodes, dtype=float)
    if nodes.ndim != 2 or nodes.shape[1] != 3:
        raise ValueError("nodes must be an (N, 3) array.")
    if np.isclose(off_sac_pos, on_sac_pos):
        raise ValueError("off_sac_pos and on_sac_pos must be different values.")

    denormalized_nodes = nodes.copy()
    rel_depth = (nodes[:, 2] - on_sac_pos) / (off_sac_pos - on_sac_pos)
    denormalized_nodes[:, 2] = med_z_on + rel_depth * (med_z_off - med_z_on)
    return denormalized_nodes


def unwarp_nodes(
    nodes: np.ndarray,
    surface_mapping: dict,
    med_z_on: float,
    med_z_off: float,
    *,
    on_sac_pos: float = 0.0,
    off_sac_pos: float = 12.0,
    conformal_jump: int | None = None,
    prenormalized: bool = False,
    backward_compatible: bool = False,
) -> np.ndarray:
    """
    Approximate inverse of `warp_nodes` for point coordinates.

    The inverse is computed with the same local least-squares model used by
    forward warping, but with input/output correspondences swapped.
    """
    points = np.asarray(nodes, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("nodes must be an (N, 3) array.")

    if prenormalized:
        prenormed_nodes = points
    else:
        prenormed_nodes = denormalize_nodes(
            points,
            med_z_on=med_z_on,
            med_z_off=med_z_off,
            on_sac_pos=on_sac_pos,
            off_sac_pos=off_sac_pos,
        )

    resolved_jump = resolve_conformal_jump(surface_mapping, conformal_jump)
    on_input_pts, off_input_pts, on_output_pts, off_output_pts, _, _ = (
        build_surface_correspondences(
            surface_mapping,
            conformal_jump=resolved_jump,
            backward_compatible=backward_compatible,
        )
    )

    # Inverse pass: swap forward correspondences (flattened -> curved frame).
    return local_ls_registration(
        prenormed_nodes,
        on_output_pts,
        off_output_pts,
        on_input_pts,
        off_input_pts,
    )
