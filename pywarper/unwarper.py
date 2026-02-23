"""Inverse mapping helpers for pywarper."""

from __future__ import annotations

from copy import deepcopy

import numpy as np
from scipy.optimize import least_squares
from skeliner.dataclass import Skeleton

from .utils import build_surface_correspondences, resolve_conformal_jump
from .warpers import (
    _apply_local_ls_state,
    _build_local_ls_state,
    local_ls_registration,
)


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


def _prepare_unwarp_inputs(
    nodes: np.ndarray,
    surface_mapping: dict,
    *,
    on_sac_pos: float,
    off_sac_pos: float,
    conformal_jump: int | None,
    backward_compatible: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    points = np.asarray(nodes, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("nodes must be an (N, 3) array.")

    resolved_jump = resolve_conformal_jump(surface_mapping, conformal_jump)
    on_input_pts, off_input_pts, on_output_pts, off_output_pts, map_med_z_on, map_med_z_off = (
        build_surface_correspondences(
            surface_mapping,
            conformal_jump=resolved_jump,
            backward_compatible=backward_compatible,
        )
    )

    prenormed_nodes = denormalize_nodes(
        points,
        med_z_on=map_med_z_on,
        med_z_off=map_med_z_off,
        on_sac_pos=on_sac_pos,
        off_sac_pos=off_sac_pos,
    )

    return prenormed_nodes, on_input_pts, off_input_pts, on_output_pts, off_output_pts


def unwarp_nodes(
    nodes: np.ndarray,
    surface_mapping: dict,
    *,
    on_sac_pos: float = 0.0,
    off_sac_pos: float = 12.0,
    conformal_jump: int | None = None,
    backward_compatible: bool = False,
    method: str = "local_ls",
    max_evals_per_point: int = 80,
    convergence_tol: float = 1e-9,
    bound_xy_to_map: bool = True,
) -> np.ndarray:
    """
    Inverse of `warp_nodes` for point coordinates.

    `method="local_ls"` mirrors the forward local least-squares model with
    swapped correspondences (flattened -> curved frame).
    `method="optimize"` refines per-point inverse coordinates by minimizing
    forward residuals (`warp_nodes(x) ~= target`).
    Input nodes are assumed to be normalized warped coordinates and are
    denormalized using the provided ON/OFF SAC reference positions.
    """
    prenormed_nodes, on_input_pts, off_input_pts, on_output_pts, off_output_pts = (
        _prepare_unwarp_inputs(
            nodes,
            surface_mapping,
            on_sac_pos=on_sac_pos,
            off_sac_pos=off_sac_pos,
            conformal_jump=conformal_jump,
            backward_compatible=backward_compatible,
        )
    )

    if method == "local_ls":
        # Inverse pass: swap forward correspondences (flattened -> curved frame).
        return local_ls_registration(
            prenormed_nodes,
            on_output_pts,
            off_output_pts,
            on_input_pts,
            off_input_pts,
        )

    if method != "optimize":
        raise ValueError("method must be one of {'local_ls', 'optimize'}")

    if max_evals_per_point <= 0:
        raise ValueError("max_evals_per_point must be a positive integer.")
    if convergence_tol <= 0:
        raise ValueError("convergence_tol must be a positive float.")

    # Start from the fast approximate inverse and refine against the forward model.
    inverse_state = _build_local_ls_state(
        on_output_pts,
        off_output_pts,
        on_input_pts,
        off_input_pts,
        window=5.0,
        max_order=2,
    )
    initial = _apply_local_ls_state(prenormed_nodes, inverse_state, warn=False)

    forward_state = _build_local_ls_state(
        on_input_pts,
        off_input_pts,
        on_output_pts,
        off_output_pts,
        window=5.0,
        max_order=2,
    )

    if bound_xy_to_map:
        x_min = float(min(on_input_pts[:, 0].min(), off_input_pts[:, 0].min()))
        x_max = float(max(on_input_pts[:, 0].max(), off_input_pts[:, 0].max()))
        y_min = float(min(on_input_pts[:, 1].min(), off_input_pts[:, 1].min()))
        y_max = float(max(on_input_pts[:, 1].max(), off_input_pts[:, 1].max()))
        lower_bounds = np.array([x_min, y_min, -np.inf], dtype=float)
        upper_bounds = np.array([x_max, y_max, np.inf], dtype=float)
    else:
        lower_bounds = np.array([-np.inf, -np.inf, -np.inf], dtype=float)
        upper_bounds = np.array([np.inf, np.inf, np.inf], dtype=float)

    recovered = np.empty_like(prenormed_nodes)
    for i, target in enumerate(prenormed_nodes):
        x0 = initial[i].astype(float, copy=True)
        if bound_xy_to_map:
            x0[:2] = np.clip(x0[:2], lower_bounds[:2], upper_bounds[:2])

        def _residual(x: np.ndarray) -> np.ndarray:
            warped = _apply_local_ls_state(x[None, :], forward_state, warn=False)[0]
            return warped - target

        sol = least_squares(
            _residual,
            x0=x0,
            bounds=(lower_bounds, upper_bounds),
            method="trf",
            max_nfev=int(max_evals_per_point),
            ftol=float(convergence_tol),
            xtol=float(convergence_tol),
            gtol=float(convergence_tol),
        )
        recovered[i] = sol.x

    return recovered


def _coerce_voxel_resolution(
    voxel_resolution: float
    | list[float | int]
    | tuple[float | int, float | int, float | int]
) -> np.ndarray:
    voxel_res = np.asarray(voxel_resolution, dtype=float)
    if voxel_res.ndim == 0:
        voxel_res = np.repeat(voxel_res, 3)
    if voxel_res.shape != (3,):
        raise ValueError("voxel_resolution must be a scalar or a length-3 sequence.")
    if np.any(np.isclose(voxel_res, 0.0)):
        raise ValueError("voxel_resolution entries must be non-zero.")
    return voxel_res


def unwarp_skeleton(
    skel: Skeleton,
    surface_mapping: dict,
    *,
    voxel_resolution: float
    | list[float | int]
    | tuple[float | int, float | int, float | int] = (1.0, 1.0, 1.0),
    on_sac_pos: float = 0.0,
    off_sac_pos: float = 12.0,
    skeleton_nodes_scale: float = 1.0,
    conformal_jump: int | None = None,
    backward_compatible: bool = False,
    method: str = "local_ls",
    max_evals_per_point: int = 80,
    convergence_tol: float = 1e-9,
    bound_xy_to_map: bool = True,
) -> Skeleton:
    """
    Inverse of `warp_skeleton` for Skeleton objects.

    Parameters
    ----------
    skel : Skeleton
        Warped skeleton, typically produced by `warp_skeleton`.
    surface_mapping : dict
        Surface mapping used for the forward warp.
    voxel_resolution : float or length-3 sequence, default=(1.0, 1.0, 1.0)
        Resolution that was used in `warp_skeleton` to convert warped nodes
        to physical units. It is undone before node-level inversion.
    on_sac_pos, off_sac_pos : float
        SAC reference positions used during normalization in the forward pass.
    skeleton_nodes_scale : float, default=1.0
        Scale factor that was used in `warp_skeleton` before warping.
    conformal_jump, backward_compatible
        Mapping options forwarded to `unwarp_nodes`.
    method, max_evals_per_point, convergence_tol, bound_xy_to_map
        Inversion options forwarded to `unwarp_nodes`.

    Returns
    -------
    Skeleton
        Skeleton with recovered nodes in the original (pre-warp) node units.
    """
    scale = float(skeleton_nodes_scale)
    if np.isclose(scale, 0.0):
        raise ValueError("skeleton_nodes_scale must be non-zero.")

    voxel_res = _coerce_voxel_resolution(voxel_resolution)

    # `warp_skeleton` stores nodes in physical units, so undo that first.
    normalized_nodes = np.asarray(skel.nodes, dtype=float) / voxel_res
    # `warp_skeleton` divides by this scale before returning the skeleton.
    normalized_nodes *= scale

    recovered_nodes = unwarp_nodes(
        normalized_nodes,
        surface_mapping,
        on_sac_pos=on_sac_pos,
        off_sac_pos=off_sac_pos,
        conformal_jump=conformal_jump,
        backward_compatible=backward_compatible,
        method=method,
        max_evals_per_point=max_evals_per_point,
        convergence_tol=convergence_tol,
        bound_xy_to_map=bound_xy_to_map,
    )
    recovered_nodes /= scale

    recovered_soma = deepcopy(skel.soma)
    recovered_soma.center = recovered_nodes[0].copy()

    return Skeleton(
        soma=recovered_soma,
        nodes=recovered_nodes,
        edges=skel.edges,
        radii=skel.radii,
        ntype=skel.ntype,
        node2verts=skel.node2verts,
        vert2node=skel.vert2node,
        meta=skel.meta.copy(),
        extra=skel.extra.copy(),
    )
