"""Inverse mapping helpers for pywarper."""

from __future__ import annotations

from copy import deepcopy

import numpy as np
from skeliner import Skeleton

from .warpers import _DepthModel, _LocalWarp


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

    Input nodes are normalized warped coordinates; they are denormalized with
    the given ON/OFF SAC reference positions and the mapping's band medians.
    Each point is then found by Newton steps on the forward warp, so that it
    warps back onto the input: at most `max_evals_per_point` steps, until a
    step is shorter than `convergence_tol`. The forward warp jumps slightly
    where two of its local fits meet; an input that falls in such a jump gets
    the point that warps closest to it. A mapping with extra surfaces is
    inverted through its piecewise depth. With `bound_xy_to_map`, x and y of
    the result are brought back inside the mapping's lattice.

    `method` takes ``"local_ls"`` or ``"optimize"``; both run this inverse.
    """
    if method not in {"local_ls", "optimize"}:
        raise ValueError("method must be one of {'local_ls', 'optimize'}")
    if max_evals_per_point <= 0:
        raise ValueError("max_evals_per_point must be a positive integer.")
    if convergence_tol <= 0:
        raise ValueError("convergence_tol must be a positive float.")

    registration = _LocalWarp(surface_mapping, conformal_jump, backward_compatible)
    prenormed_nodes = denormalize_nodes(
        nodes,
        med_z_on=registration.med_z_on,
        med_z_off=registration.med_z_off,
        on_sac_pos=on_sac_pos,
        off_sac_pos=off_sac_pos,
    )
    layers = None
    if "extra_surfaces" in surface_mapping:
        layers = _DepthModel(surface_mapping, backward_compatible)
    return registration.unwarp(
        prenormed_nodes,
        layers=layers,
        max_steps=max_evals_per_point,
        tolerance=convergence_tol,
        bound=bound_xy_to_map,
    )


def _coerce_voxel_resolution(
    voxel_resolution: float
    | list[float | int]
    | tuple[float | int, float | int, float | int],
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
