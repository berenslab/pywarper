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
    median_depths: dict[str, float] | None = None,
    anchors: tuple[str, str] = ("on_sac", "off_sac"),
    anchor_pos: tuple[float, float] = (0.0, 12.0),
    *,
    med_z_on: float | None = None,
    med_z_off: float | None = None,
    on_sac_pos: float | None = None,
    off_sac_pos: float | None = None,
) -> np.ndarray:
    """
    Undo `normalize_nodes` and map z back to the pre-normalized warped frame.

    Parameters
    ----------
    nodes : np.ndarray
        (N, 3) normalized [x, y, z] coordinates.
    median_depths : dict[str, float] or None
        Mapping of surface tag -> median z depth.
        When None, *med_z_on* and *med_z_off* must be supplied instead.
    anchors : tuple[str, str]
        Tags of the two anchor surfaces.
    anchor_pos : tuple[float, float]
        Normalized positions for the two anchors.
    med_z_on, med_z_off : float | None
        Legacy keywords for the two-surface case. Used when *median_depths*
        is None.
    on_sac_pos, off_sac_pos : float | None
        Legacy keywords that override *anchor_pos*.

    Returns
    -------
    np.ndarray
        (N, 3) coordinates in the pre-normalized warped frame.
    """
    nodes = np.asarray(nodes, dtype=float)
    if nodes.ndim != 2 or nodes.shape[1] != 3:
        raise ValueError("nodes must be an (N, 3) array.")

    # ---- resolve legacy call convention ------------------------------------
    if median_depths is None:
        if med_z_on is None or med_z_off is None:
            raise ValueError(
                "Either median_depths dict or both med_z_on and med_z_off must be provided."
            )
        median_depths_dict: dict[str, float] = {
            "on_sac": float(med_z_on),
            "off_sac": float(med_z_off),
        }
    elif isinstance(median_depths, (int, float)):
        # Positional scalar: treat as med_z_on for backward compat
        if med_z_off is None:
            raise ValueError("med_z_off must be provided when median_depths is a scalar (legacy API).")
        median_depths_dict = {"on_sac": float(median_depths), "off_sac": float(med_z_off)}
    else:
        median_depths_dict = median_depths

    if on_sac_pos is not None:
        anchor_pos = (on_sac_pos, anchor_pos[1] if off_sac_pos is None else off_sac_pos)
    if off_sac_pos is not None and on_sac_pos is None:
        anchor_pos = (anchor_pos[0], off_sac_pos)

    if np.isclose(anchor_pos[1], anchor_pos[0]):
        raise ValueError("anchor positions must be different values.")

    z_a = median_depths_dict[anchors[0]]
    z_b = median_depths_dict[anchors[1]]

    denormalized_nodes = nodes.copy()
    rel_depth = (nodes[:, 2] - anchor_pos[0]) / (anchor_pos[1] - anchor_pos[0])
    denormalized_nodes[:, 2] = z_a + rel_depth * (z_b - z_a)
    return denormalized_nodes


def _prepare_unwarp_inputs(
    nodes: np.ndarray,
    surface_mapping: dict,
    *,
    on_sac_pos: float,
    off_sac_pos: float,
    conformal_jump: int | None,
    backward_compatible: bool,
) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray]]:
    points = np.asarray(nodes, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("nodes must be an (N, 3) array.")

    resolved_jump = resolve_conformal_jump(surface_mapping, conformal_jump)
    input_pts_list, output_pts_list, median_depths = build_surface_correspondences(
        surface_mapping,
        conformal_jump=resolved_jump,
        backward_compatible=backward_compatible,
    )

    prenormed_nodes = denormalize_nodes(
        points,
        median_depths=median_depths,
        anchor_pos=(on_sac_pos, off_sac_pos),
    )

    return prenormed_nodes, input_pts_list, output_pts_list


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
    prenormed_nodes, input_pts_list, output_pts_list = _prepare_unwarp_inputs(
        nodes,
        surface_mapping,
        on_sac_pos=on_sac_pos,
        off_sac_pos=off_sac_pos,
        conformal_jump=conformal_jump,
        backward_compatible=backward_compatible,
    )

    if method == "local_ls":
        # Inverse pass: swap forward correspondences (flattened -> curved frame).
        return local_ls_registration(
            prenormed_nodes,
            output_pts_list,
            input_pts_list,
        )

    if method != "optimize":
        raise ValueError("method must be one of {'local_ls', 'optimize'}")

    if max_evals_per_point <= 0:
        raise ValueError("max_evals_per_point must be a positive integer.")
    if convergence_tol <= 0:
        raise ValueError("convergence_tol must be a positive float.")

    # Start from the fast approximate inverse and refine against the forward model.
    inverse_state = _build_local_ls_state(
        output_pts_list,
        input_pts_list,
        window=5.0,
        max_order=2,
    )
    initial = _apply_local_ls_state(prenormed_nodes, inverse_state, warn=False)

    forward_state = _build_local_ls_state(
        input_pts_list,
        output_pts_list,
        window=5.0,
        max_order=2,
    )

    if bound_xy_to_map:
        all_input = np.vstack(input_pts_list)
        x_min = float(all_input[:, 0].min())
        x_max = float(all_input[:, 0].max())
        y_min = float(all_input[:, 1].min())
        y_max = float(all_input[:, 1].max())
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
