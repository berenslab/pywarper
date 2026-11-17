"""
pywarper.warpers
==============
Spatial warping and profiling utilities for **neuronal skeleton reconstructions**.

This module takes a neuronal tree (nodes+edges) and the previously-computed
ON/OFF Starburst Amacrine Cell (SAC) surface mapping in order to

1. **Warp the skeleton into the flattened SAC coordinate frame** (`warp_nodes`).
   Each node is locally re-registered with a polynomial least-squares fit that
   references both SAC layers so that depth is preserved relative to the
   curved retina.
2. **Compute depth (z) profiles** (`get_z_profile`).  Edge lengths are first binned
   directly (histogram) and then re-estimated with a Kaiser–Bessel gridding
   kernel to obtain a smooth 1-D density across the inner plexiform layer.
3. **Compute planar (xy) density maps** (`get_xy_profile`).  Dendritic length is
   accumulated on a user-defined 2-D grid and optionally Gaussian-smoothed for
   visualisation or group statistics.

Key algorithms
--------------
* **Polynomial local registration** – Around every node a 2-D polynomial
  basis (up to a configurable `max_order`) is fitted to the positions of the
  neighbouring SAC-band sample points, three least-squares systems at once.
  The sample points lie on a regular lattice, so nodes whose neighbourhood is
  the same block of it share one fit.
* **Kaiser–Bessel gridding** – The 1-D `gridder1d` function emulates the
  non-uniform FFT gridding scheme used by older MATLAB code, yielding the exact
  same numerical output but in fully vectorised NumPy.
"""

import time
from copy import deepcopy
from importlib import metadata as _metadata
from pathlib import Path

import numpy as np
import skeliner as sk
import trimesh
from numpy.linalg import lstsq
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import gaussian_filter
from scipy.spatial import KDTree
from scipy.special import i0
from skeliner import Skeleton
from skeliner._core import _bfs_parents

from .surface import build_mapping, fit_sac_surface
from .utils import resolve_conformal_jump, surface_lattice

_PYWARPER_VERSION = _metadata.version("pywarper")

# points handled at a time by the registration
_CHUNK = 100_000
# step of the finite difference that reads a layer's tilt, in surface units
_NUDGE = 1e-3


def poly_basis_2d(x: np.ndarray, y: np.ndarray, max_order: int) -> np.ndarray:
    """
    Return the full 2-D polynomial basis up to total order *max_order*
    for coordinates (x, y).  Shape:  (len(x), n_terms)

    Order layout matches the original code:
        [1,
         x, y,
         y²,  x·y,  x²,      # order 2
         y³,  x y², x²y, x³, …]
    """
    cols = [np.ones_like(x), x, y]  # constant + linear
    for order in range(2, max_order + 1):
        for ox in range(order + 1):
            oy = order - ox
            cols.append(x**ox * y**oy)
    return np.stack(cols, axis=1)  # (N, n_terms)


def _poly_gradient_2d(
    x: np.ndarray, y: np.ndarray, max_order: int
) -> tuple[np.ndarray, np.ndarray]:
    """Derivatives of `poly_basis_2d`'s terms along x and along y, in its layout."""
    zero, one = np.zeros_like(x), np.ones_like(x)
    along_x, along_y = [zero, one, zero], [zero, zero, one]
    for order in range(2, max_order + 1):
        for ox in range(order + 1):
            oy = order - ox
            along_x.append(ox * x ** (ox - 1) * y**oy if ox else zero)
            along_y.append(oy * x**ox * y ** (oy - 1) if oy else zero)
    return np.stack(along_x, axis=1), np.stack(along_y, axis=1)


class _LocalWarp:
    """
    The local least-squares registration of a surface mapping, and its inverse.

    A point is moved by a polynomial in (x, y) up to `max_order`, linear in z,
    fitted to the map's vertices of both bands that lie within `window` of the
    point in x and in y: from where the vertices are to where the mapping
    flattened them. The vertices form one regular lattice, so every point whose
    window covers the same block of it has the same fit; a block is fitted once,
    when a point first needs it.

    A point is left where it is when its window holds fewer than 12 vertices,
    or when it lies beyond the lattice and its block has too few rows or
    columns to determine the polynomial, which the fit would then extrapolate.
    """

    def __init__(
        self,
        surface_mapping: dict,
        conformal_jump: int | None = None,
        backward_compatible: bool = False,
        *,
        window: float = 5.0,
        max_order: int = 2,
    ):
        xs, ys, z_on, z_off, flat_on, flat_off, self.med_z_on, self.med_z_off = (
            surface_lattice(
                surface_mapping,
                conformal_jump=resolve_conformal_jump(surface_mapping, conformal_jump),
                backward_compatible=backward_compatible,
            )
        )
        self.xs, self.ys = xs.astype(float), ys.astype(float)
        shape = (len(xs), len(ys))
        # per band and lattice vertex: height, flattened x and y
        self.heights = np.stack([z_on, z_off])
        self.flat = np.stack(
            [
                flat.reshape((shape[1], shape[0], 2)).transpose(1, 0, 2)
                for flat in (flat_on, flat_off)
            ]
        )
        self.window, self.max_order = float(window), int(max_order)
        self.n_terms = (self.max_order + 1) * (self.max_order + 2) // 2
        self._rows: dict[int, int] = {}
        self._transforms = np.zeros((0, 2 * self.n_terms, 3))
        self._shifts = np.zeros((0, 2))
        self._fitted = np.zeros(0, dtype=bool)
        self._coarse = None

    # ------------------------------------------------------------- the fits
    def _spans(self, points: np.ndarray) -> np.ndarray:
        """Per point, the lattice block inside its window: rows [i0, i1) and
        columns [j0, j1)."""
        lo, hi = points[:, :2] - self.window, points[:, :2] + self.window
        return np.column_stack(
            [
                np.searchsorted(self.xs, lo[:, 0], side="left"),
                np.searchsorted(self.xs, hi[:, 0], side="right"),
                np.searchsorted(self.ys, lo[:, 1], side="left"),
                np.searchsorted(self.ys, hi[:, 1], side="right"),
            ]
        )

    def _blocks(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Per point, where its block's fit is stored (fitting the blocks that
        are new), and the block itself."""
        spans = self._spans(points)
        nx, ny = len(self.xs) + 1, len(self.ys) + 1
        keys = ((spans[:, 0] * nx + spans[:, 1]) * ny + spans[:, 2]) * ny + spans[:, 3]
        unique, first, inverse = np.unique(keys, return_index=True, return_inverse=True)
        rows = np.array([self._rows.get(key, -1) for key in unique.tolist()], dtype=int)
        new = np.flatnonzero(rows < 0)
        if new.size:
            rows[new] = self._fit(spans[first[new]])
            self._rows.update(zip(unique[new].tolist(), rows[new].tolist()))
        return rows[inverse], spans

    def _fit(self, spans: np.ndarray) -> np.ndarray:
        """Fit the blocks `spans` (C, 4); returns where their fits are stored."""
        n = len(spans)
        transforms = np.zeros((n, 2 * self.n_terms, 3))
        shifts = np.zeros((n, 2))
        fitted = np.zeros(n, dtype=bool)
        shape = spans[:, [1, 3]] - spans[:, [0, 2]]
        for ni, nj in np.unique(shape, axis=0):
            if 2 * ni * nj < 12:
                continue
            same = np.flatnonzero((shape[:, 0] == ni) & (shape[:, 1] == nj))
            for part in np.array_split(same, len(same) // 2000 + 1):
                transforms[part], shifts[part] = self._fit_alike(spans[part], ni, nj)
            fitted[same] = True
        start = len(self._fitted)
        self._transforms = np.concatenate([self._transforms, transforms])
        self._shifts = np.concatenate([self._shifts, shifts])
        self._fitted = np.concatenate([self._fitted, fitted])
        return np.arange(start, start + n)

    def _fit_alike(
        self, spans: np.ndarray, ni: int, nj: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Transforms (C, 2 * n_terms, 3) and xy shifts (C, 2) of blocks that
        all hold `ni` rows and `nj` columns of the lattice."""
        n = len(spans)
        ii = (spans[:, 0, None] + np.arange(ni))[:, :, None]
        jj = (spans[:, 2, None] + np.arange(nj))[:, None, :]
        both = (n, 2 * ni * nj)
        # the block's vertices, the ON band's then the OFF band's
        x = np.broadcast_to(self.xs[ii], (2, n, ni, nj)).transpose(1, 0, 2, 3)
        y = np.broadcast_to(self.ys[jj], (2, n, ni, nj)).transpose(1, 0, 2, 3)
        x, y = x.reshape(both), y.reshape(both)
        z = self.heights[:, ii, jj].transpose(1, 0, 2, 3).reshape(both)
        flat = self.flat[:, ii, jj].transpose(1, 0, 2, 3, 4).reshape(both + (2,))
        flat_z = np.repeat([self.med_z_on, self.med_z_off], ni * nj)

        shift = np.column_stack([x.mean(axis=1), y.mean(axis=1)])
        base = poly_basis_2d(
            (x - shift[:, :1]).ravel(), (y - shift[:, 1:]).ravel(), self.max_order
        ).reshape(both + (self.n_terms,))
        design = np.concatenate([base, base * z[:, :, None]], axis=2)
        target = np.concatenate(
            [
                flat - shift[:, None, :],
                np.broadcast_to(flat_z[None, :, None], both + (1,)),
            ],
            axis=2,
        )
        if min(ni, nj) > self.max_order:
            cutoff = max(design.shape[1:]) * np.finfo(float).eps
            return np.linalg.pinv(design, rtol=cutoff) @ target, shift
        # too few rows or columns for the polynomial: the smallest solution
        transforms = np.stack(
            [lstsq(design[c], target[c], rcond=None)[0] for c in range(n)]
        )
        return transforms, shift

    # ------------------------------------------------------- forward, inverse
    def _evaluate(self, points: np.ndarray, jacobian: bool = False):
        """Warped `points`, which of them had a fit, their blocks and, on
        request, the derivative of the warp at each point (N, 3, 3)."""
        rows, spans = self._blocks(points)
        transforms, shift = self._transforms[rows], self._shifts[rows]
        thin = (
            np.minimum(spans[:, 1] - spans[:, 0], spans[:, 3] - spans[:, 2])
            <= self.max_order
        )
        beyond = (
            (points[:, 0] < self.xs[0])
            | (points[:, 0] > self.xs[-1])
            | (points[:, 1] < self.ys[0])
            | (points[:, 1] > self.ys[-1])
        )
        fitted = self._fitted[rows] & ~(thin & beyond)
        x, y, z = points[:, 0] - shift[:, 0], points[:, 1] - shift[:, 1], points[:, 2]
        base = poly_basis_2d(x, y, self.max_order)

        def moved(terms: np.ndarray, with_z: np.ndarray) -> np.ndarray:
            return np.einsum(
                "nk,nkd->nd", np.concatenate([terms, with_z], axis=1), transforms
            )

        warped = moved(base, base * z[:, None])
        warped[:, :2] += shift
        warped[~fitted] = points[~fitted]
        if not jacobian:
            return warped, fitted, spans
        along_x, along_y = _poly_gradient_2d(x, y, self.max_order)
        slope = np.stack(
            [
                moved(along_x, along_x * z[:, None]),
                moved(along_y, along_y * z[:, None]),
                moved(np.zeros_like(base), base),
            ],
            axis=2,
        )
        slope[~fitted] = np.eye(3)
        return warped, fitted, spans, slope

    def warp(self, points: np.ndarray, *, warn: bool = True) -> np.ndarray:
        """`points` (N, 3) in the frame the mapping flattens to, before their
        depth is normalized."""
        points = np.asarray(points, dtype=float)
        warped = np.empty_like(points)
        for start in range(0, len(points), _CHUNK):
            part = points[start : start + _CHUNK]
            warped[start : start + _CHUNK], fitted, spans = self._evaluate(part)
            if not warn:
                continue
            for k in np.flatnonzero(~fitted):
                vertices = (spans[k, 1] - spans[k, 0]) * (spans[k, 3] - spans[k, 2])
                what = "not enough neighbours" if vertices else "no neighbours"
                x, y, z = part[k]
                print(
                    f"[pywarper] Warning: {what} for node {start + k} "
                    f"at ({x:.2f}, {y:.2f}, {z:.2f})"
                )
        return warped

    def unwarp(
        self,
        targets: np.ndarray,
        *,
        layers: "_DepthModel | None" = None,
        max_steps: int = 80,
        tolerance: float = 1e-9,
        bound: bool = True,
    ) -> np.ndarray:
        """
        The points that `warp` takes to `targets` (N, 3), by Newton steps.

        With `layers`, depth is the mapping's piecewise one: z then follows from
        the layers in each point's column and the steps are taken in x and y.
        A point stops when a step is shorter than `tolerance`; one that has not
        stopped after `max_steps` gets the position that came closest. With
        `bound`, x and y of the result are brought back inside the lattice.
        """
        targets = np.asarray(targets, dtype=float)
        points = np.empty_like(targets)
        for start in range(0, len(targets), _CHUNK):
            part = slice(start, start + _CHUNK)
            points[part] = self._unwarp(
                targets[part], layers, int(max_steps), float(tolerance), bound
            )
        return points

    def _nearby(self, targets: np.ndarray, depth: np.ndarray) -> np.ndarray:
        """A point near each target's preimage: the lattice vertex that is
        flattened nearest to it, among a sparse set of them."""
        if self._coarse is None:
            spacing = self.xs[1] - self.xs[0] if len(self.xs) > 1 else 1.0
            every = max(1, int(round(2.0 * self.window / spacing)))
            ii, jj = np.meshgrid(
                np.arange(0, len(self.xs), every),
                np.arange(0, len(self.ys), every),
                indexing="ij",
            )
            ii, jj = ii.ravel(), jj.ravel()
            self._coarse = (KDTree(self.flat[:, ii, jj].mean(axis=0)), ii, jj)
        tree, ii, jj = self._coarse
        nearest = tree.query(targets[:, :2])[1]
        i, j = ii[nearest], jj[nearest]
        z_on, z_off = self.heights[0, i, j], self.heights[1, i, j]
        return np.column_stack([self.xs[i], self.ys[j], z_on + depth * (z_off - z_on)])

    def _unwarp(self, targets, layers, max_steps, tolerance, bound) -> np.ndarray:
        depth = (targets[:, 2] - self.med_z_on) / (self.med_z_off - self.med_z_on)
        points = self._nearby(targets, depth)
        low = np.array([self.xs[0], self.ys[0]])
        high = np.array([self.xs[-1], self.ys[-1]])

        def miss_and_step(p: np.ndarray, which: np.ndarray):
            """How far `p` warps from its target, and the Newton step from it."""
            if layers is None:
                warped, _, _, slope = self._evaluate(p, jacobian=True)
                miss = warped - targets[which]
                return miss, _solve(slope, miss)
            p[:, 2] = layers.height(p[:, :2], depth[which])
            warped, _, _, slope = self._evaluate(p, jacobian=True)
            miss = warped[:, :2] - targets[which, :2]
            # z moves with x and y along the surface of this depth
            tilt = np.empty((len(p), 2))
            for axis in (0, 1):
                ahead, behind = p[:, :2].copy(), p[:, :2].copy()
                ahead[:, axis] += _NUDGE
                behind[:, axis] -= _NUDGE
                tilt[:, axis] = (
                    layers.height(ahead, depth[which])
                    - layers.height(behind, depth[which])
                ) / (2.0 * _NUDGE)
            along = slope[:, :2, :2] + slope[:, :2, 2:] * tilt[:, None, :]
            step = np.zeros_like(p)
            step[:, :2] = _solve(along, miss)
            return miss, step

        closest = points.copy()
        closest_miss = np.full(len(points), np.inf)
        moving = np.arange(len(points))
        for _ in range(max_steps + 1):
            p = points[moving]
            miss, step = miss_and_step(p, moving)
            size = np.linalg.norm(miss, axis=1)
            closer = size < closest_miss[moving]
            closest[moving[closer]] = p[closer]
            closest_miss[moving[closer]] = size[closer]
            points[moving] = p - step
            moving = moving[np.abs(step).max(axis=1) >= tolerance]
            if not moving.size:
                break
        # a point that never settled is on a border between two fits
        points[moving] = closest[moving]
        if bound:
            np.clip(points[:, :2], low, high, out=points[:, :2])
        if layers is not None:
            points[:, 2] = layers.height(points[:, :2], depth)
        return points


def _solve(matrices: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Solve each small system of a stack; least squares where one is singular."""
    try:
        return np.linalg.solve(matrices, right[:, :, None])[:, :, 0]
    except np.linalg.LinAlgError:
        return (np.linalg.pinv(matrices) @ right[:, :, None])[:, :, 0]


def warp_nodes(
    nodes: np.ndarray,
    surface_mapping: dict,
    conformal_jump: int | None = None,
    backward_compatible: bool = False,
) -> tuple[np.ndarray, float, float]:
    """Warp `nodes` (surface units) into the flattened frame, before normalization."""
    registration = _LocalWarp(surface_mapping, conformal_jump, backward_compatible)
    med_z_on, med_z_off = registration.med_z_on, registration.med_z_off
    warped = registration.warp(nodes)
    if "extra_surfaces" in surface_mapping:
        rel = relative_depth(
            nodes, surface_mapping, backward_compatible=backward_compatible
        )
        warped[:, 2] = med_z_on + rel * (med_z_off - med_z_on)

    return warped, med_z_on, med_z_off


def normalize_nodes(
    nodes: np.ndarray,
    med_z_on: float,
    med_z_off: float,
    on_sac_pos: float = 0.0,
    off_sac_pos: float = 12.0,
) -> np.ndarray:
    """
    Normalize the z-coordinates of nodes based on the median z-values
    of the ON and OFF SAC surfaces.
    This function rescales the z-coordinates of the nodes to a normalized
    space where the ON SAC surface is at `on_sac_pos` and the OFF SAC
    surface is at `off_sac_pos`. The z-coordinates are adjusted based on
    the provided median z-values of the ON and OFF SAC surfaces.

    Parameters
    ----------
    nodes : np.ndarray
        (N, 3) array of [x, y, z] coordinates for the nodes to be normalized.
    med_z_on : float
        Median z-value of the ON SAC surface.
    med_z_off : float
        Median z-value of the OFF SAC surface.
    on_sac_pos : float, default=0.0
        Desired position of the ON SAC surface in the normalized space (µm).
    off_sac_pos : float, default=12.0
        Desired position of the OFF SAC surface in the normalized space (µm).
    z_res : float, default=1.0
        Spatial resolution along z (µm / voxel) after warping.
    Returns
    -------
    np.ndarray
        (N, 3) array of [x, y, z] coordinates with normalized z-coordinates.
    """
    normalized_nodes = nodes.copy().astype(float)

    # Compute the relative depth of each node
    rel_depth = (nodes[:, 2] - med_z_on) / (med_z_off - med_z_on)  # 0->ON, 1->OFF

    # Rescale the z-coordinates to the normalized space
    z_phys = on_sac_pos + rel_depth * (off_sac_pos - on_sac_pos)  # µm in global frame
    normalized_nodes[:, 2] = z_phys  # update the z-coordinate to the flattened space

    return normalized_nodes


def _bracket(
    values: np.ndarray, valid: np.ndarray, query: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Per row, the two valid columns of `values` (increasing) that bracket `query`."""
    n_cols = values.shape[1]
    idx = np.arange(n_cols)
    at_or_below = valid & (values <= query[:, None])
    lo = np.where(at_or_below, idx, -1).max(axis=1)
    hi = np.where(valid & ~at_or_below, idx, n_cols).min(axis=1)
    first = np.argmax(valid, axis=1)
    second = np.argmax(valid & (idx > first[:, None]), axis=1)
    last = n_cols - 1 - np.argmax(valid[:, ::-1], axis=1)
    second_last = (
        n_cols - 1 - np.argmax((valid & (idx < last[:, None]))[:, ::-1], axis=1)
    )
    # outside every valid layer the outermost interval is extended
    under, over = lo < 0, hi >= n_cols
    lo = np.where(under, first, np.where(over, second_last, lo))
    hi = np.where(under, second, np.where(over, last, hi))
    return lo, hi


class _DepthModel:
    """Column-wise depth through a mapping's layers: relative depth and its inverse."""

    def __init__(self, surface_mapping: dict, backward_compatible: bool = False):
        on = np.asarray(surface_mapping["on_sac_surface"], dtype=float)
        off = np.asarray(surface_mapping["off_sac_surface"], dtype=float)
        self.first = 1.0 if backward_compatible else 0.0
        self.shape = on.shape
        self.gx = np.arange(on.shape[0]) + self.first
        self.gy = np.arange(on.shape[1]) + self.first
        heights, depths = [on, off], [0.0, 1.0]
        if "extra_surfaces" in surface_mapping:
            heights += [
                np.asarray(h, dtype=float) for h in surface_mapping["extra_surfaces"]
            ]
            depths += [float(d) for d in surface_mapping["extra_surface_depths"]]
        self.depths = np.asarray(depths)
        self.fields = [RegularGridInterpolator((self.gx, self.gy), h) for h in heights]
        # extra layers join outward from the bands, the nearest first
        outside = (self.depths < 0) | (self.depths > 1)
        distance = np.where(
            outside,
            np.maximum(-self.depths, self.depths - 1.0),
            np.minimum(self.depths, 1.0 - self.depths),
        )
        self.join = [int(k) + 2 for k in np.argsort(distance[2:], kind="stable")]
        self.order = np.argsort(self.depths, kind="stable")

    def xy(self, points: np.ndarray) -> np.ndarray:
        """x, y of `points`, clipped to the surfaces' grid."""
        return np.column_stack(
            [
                np.clip(points[:, 0], self.gx[0], self.gx[-1]),
                np.clip(points[:, 1], self.gy[0], self.gy[-1]),
            ]
        )

    def pixel(self, xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Nearest grid indices of clipped x, y."""
        ix = np.clip(np.rint(xy[:, 0] - self.first).astype(int), 0, self.shape[0] - 1)
        iy = np.clip(np.rint(xy[:, 1] - self.first).astype(int), 0, self.shape[1] - 1)
        return ix, iy

    def stack(
        self, xy: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Layer heights at xy in depth order, oriented to grow with depth, the
        depths, which layers keep that order in each column, and the orientation."""
        z = np.column_stack([f(xy) for f in self.fields])
        sign = np.sign(z[:, 1] - z[:, 0])
        z = z * sign[:, None]
        valid = np.ones(z.shape, dtype=bool)
        kept = [0, 1]
        for k in self.join:
            below = [j for j in kept if self.depths[j] < self.depths[k]]
            above = [j for j in kept if self.depths[j] > self.depths[k]]
            lo = hi = None
            if below:
                lo = np.where(valid[:, below], z[:, below], -np.inf).max(axis=1)
            if above:
                hi = np.where(valid[:, above], z[:, above], np.inf).min(axis=1)
            ok = np.ones(len(z), dtype=bool)
            if lo is not None:
                ok &= z[:, k] > lo
            if hi is not None:
                ok &= z[:, k] < hi
            valid[:, k] = ok
            kept.append(k)
        return z[:, self.order], self.depths[self.order], valid[:, self.order], sign

    def relative(self, points: np.ndarray) -> np.ndarray:
        """Relative depth of (N, 3) points."""
        points = np.asarray(points, dtype=float)
        xy = self.xy(points)
        if len(self.fields) == 2:
            z_on, z_off = self.fields[0](xy), self.fields[1](xy)
            with np.errstate(divide="ignore", invalid="ignore"):
                return (points[:, 2] - z_on) / (z_off - z_on)
        z, depths, valid, sign = self.stack(xy)
        p = points[:, 2] * sign
        lo, hi = _bracket(z, valid, p)
        rows = np.arange(len(p))
        with np.errstate(divide="ignore", invalid="ignore"):
            return depths[lo] + (p - z[rows, lo]) / (z[rows, hi] - z[rows, lo]) * (
                depths[hi] - depths[lo]
            )

    def height(self, xy: np.ndarray, depth: np.ndarray) -> np.ndarray:
        """z at which a point over (N, 2) xy has the given relative depth."""
        xy = self.xy(np.asarray(xy, dtype=float))
        depth = np.asarray(depth, dtype=float)
        z, depths, valid, sign = self.stack(xy)
        lo, hi = _bracket(np.broadcast_to(depths, z.shape), valid, depth)
        rows = np.arange(len(depth))
        oriented = z[rows, lo] + (depth - depths[lo]) / (depths[hi] - depths[lo]) * (
            z[rows, hi] - z[rows, lo]
        )
        return oriented * sign


def relative_depth(
    points: np.ndarray,
    surface_mapping: dict,
    *,
    backward_compatible: bool = False,
) -> np.ndarray:
    """
    Relative depth of points (0 at the ON band, 1 at the OFF band), per column.

    Without extra surfaces in the mapping this is the fraction of the way from
    the ON to the OFF band in the point's column, extended linearly outside
    them. With extra surfaces (see `build_mapping`) the depth is interpolated
    piecewise between the two layers that bracket the point and extended
    linearly with the outermost interval. An extra layer that is out of order
    in a column, given the layers nearer the bands, is left out there.

    Parameters
    ----------
    points : np.ndarray
        (N, 3) positions before warping, in the surfaces' units.
    surface_mapping : dict
        Output of `build_mapping`.
    backward_compatible : bool, default=False
        The surfaces' grid starts at 1 instead of 0.

    Returns
    -------
    np.ndarray
        (N,) relative depth.
    """
    return _DepthModel(surface_mapping, backward_compatible).relative(points)


def height_at_depth(
    xy: np.ndarray,
    depth: np.ndarray,
    surface_mapping: dict,
    *,
    backward_compatible: bool = False,
) -> np.ndarray:
    """
    Inverse of `relative_depth`: the z at which each (x, y) has relative depth `depth`.

    Parameters
    ----------
    xy : np.ndarray
        (N, 2) positions before warping, in the surfaces' units.
    depth : np.ndarray
        (N,) relative depth (0 at the ON band, 1 at the OFF band).
    surface_mapping : dict
        Output of `build_mapping`.
    backward_compatible : bool, default=False
        The surfaces' grid starts at 1 instead of 0.

    Returns
    -------
    np.ndarray
        (N,) z in the surfaces' units.
    """
    return _DepthModel(surface_mapping, backward_compatible).height(xy, depth)


def node_coverage(
    nodes: np.ndarray,
    surface_mapping: dict,
    *,
    backward_compatible: bool = False,
) -> dict | None:
    """
    Where the imaged volume and the layer observations end, per node.

    Parameters
    ----------
    nodes : np.ndarray
        (N, 3) node positions before warping, in the surfaces' units.
    surface_mapping : dict
        Output of `build_mapping`. Besides the surfaces, only its ``volume_z``,
        ``on_support``, ``off_support`` and ``extra_surface_supports`` entries
        are read.
    backward_compatible : bool, default=False
        The surfaces' grid starts at 1 instead of 0.

    Returns
    -------
    dict or None
        None when the mapping has none of those entries. Otherwise, each only
        when the mapping provides what it needs:

        ``depth_rel`` : (N, 2)
            The depth range the volume covers in each node's column, lowest
            first, in relative depth (0 at the ON band, 1 at the OFF band).
            ±inf on a side whose face is unknown.
        ``on_support``, ``off_support`` : (N,) bool
            Whether that band was observed in the node's column.
        ``extra_support`` : (N, K) bool, ``extra_names`` : (K,)
            Whether each extra layer was observed in the node's column, and the
            layers' names.
    """
    has_faces = "volume_z" in surface_mapping
    bands = [b for b in ("on", "off") if f"{b}_support" in surface_mapping]
    has_extra = "extra_surface_supports" in surface_mapping
    if not has_faces and not bands and not has_extra:
        return None

    nodes = np.asarray(nodes, dtype=float)
    model = _DepthModel(surface_mapping, backward_compatible)
    xy = model.xy(nodes)

    coverage: dict = {}
    if has_faces:
        # z grows with depth (+1) or against it (-1) in each column
        sign = np.sign(model.fields[1](xy) - model.fields[0](xy))
        faces = np.asarray(surface_mapping["volume_z"], dtype=float)
        rel = np.empty((len(nodes), 2))
        for k, face in enumerate(faces):
            if np.isnan(face):
                # an unknown face does not bound the column on its side
                rel[:, k] = (-np.inf if k == 0 else np.inf) * sign
            else:
                at_face = np.column_stack([xy, np.full(len(nodes), face)])
                rel[:, k] = model.relative(at_face)
        coverage["depth_rel"] = np.sort(rel, axis=1)

    ix, iy = model.pixel(xy)
    for band in bands:
        support = np.asarray(surface_mapping[f"{band}_support"], dtype=bool)
        coverage[f"{band}_support"] = support[ix, iy]
    if has_extra:
        supports = np.asarray(surface_mapping["extra_surface_supports"], dtype=bool)
        coverage["extra_support"] = supports[:, ix, iy].T
        coverage["extra_names"] = np.asarray(surface_mapping["extra_surface_names"])
    return coverage


def _covered_depth(
    depth_rel: np.ndarray,
    on_sac_pos: float,
    off_sac_pos: float,
    scale: float = 1.0,
    z_resolution: float = 1.0,
) -> np.ndarray:
    """`node_coverage`'s relative depth range in the warped output's z units."""
    with np.errstate(invalid="ignore"):
        depth = on_sac_pos + np.asarray(depth_rel) * (off_sac_pos - on_sac_pos)
    return np.sort(depth / scale * z_resolution, axis=1)


def warp_skeleton(
    skel: Skeleton,
    surface_mapping: dict,
    voxel_resolution: float | list[float | int] = [1.0, 1.0, 1.0],
    on_sac_pos: float = 0.0,
    off_sac_pos: float = 12.0,
    z_profile_extent: list[float | int] | None = None,  # [z_min, z_max]
    z_profile_bin_size: float | int = 1.0,
    z_profile_hdr_mass: float | int = 0.95,
    z_profile_include_soma: bool = False,
    z_profile_voxel_size: float | None = None,
    xy_profile_extents: list[float | int] | None = None,  # [x_min, x_max, y_min, y_max]
    xy_profile_bin_size: float | int = 20.0,
    xy_profile_smooth: float = 1.0,
    xy_profile_include_soma: bool = False,
    xy_profile_voxel_size: float | None = None,
    radius_metric: str | None = None,
    skeleton_nodes_scale: float = 1.0,
    conformal_jump: int | None = None,
    backward_compatible: bool = False,
    verbose: bool = False,
) -> Skeleton:
    """
    Applies a local surface flattening (warp) to a neuronal skeleton using the results
    of previously computed surface mappings.

    Parameters
    ----------
    nodes : np.ndarray
        (N, 3) array of [x, y, z] coordinates for the skeleton to be warped.
    edges : np.ndarray
        (E, 2) array of indices defining connectivity between nodes.
    radii : np.ndarray
        (N,) array of radii corresponding to each node.
    surface_mapping : dict
        Dictionary containing keys:
          - "mapped_min_positions" : np.ndarray
              (X*Y, 2) mapped coordinates for one surface band (e.g., "min" band).
          - "mapped_max_positions" : np.ndarray
              (X*Y, 2) mapped coordinates for the other surface band (e.g., "max" band).
          - "thisVZminmesh" : np.ndarray
              (X, Y) mesh representing the first surface (“min” band) in 3D space.
          - "thisVZmaxmesh" : np.ndarray
              (X, Y) mesh representing the second surface (“max” band) in 3D space.
          - "thisx" : np.ndarray
              1D array of x-indices (possibly downsampled) used during mapping.
          - "thisy" : np.ndarray
              1D array of y-indices (possibly downsampled) used during mapping.
    conformal_jump : int, default=1
        Step size used in the conformal mapping (downsampling factor).
    verbose : bool, default=False
        If True, prints timing and progress information.

    Returns
    -------
    dict
        Dictionary containing:
          - "nodes": np.ndarray
              (N, 3) warped [x, y, z] coordinates after applying local registration.
          - "edges": np.ndarray
              (E, 2) connectivity array (passed through unchanged).
          - "radii": np.ndarray
              (N,) radii array (passed through unchanged).
          - "medVZmin": float
              Median z-value of the “min” surface mesh within the region of interest.
          - "medVZmax": float
              Median z-value of the “max” surface mesh within the region of interest.

    Notes
    -----
    1. The function extracts a subregion of the surfaces according to thisx/thisy and
       conformal_jump, matching the flattening step used in the mapping.
    2. Each node in `nodes` is then warped via local least-squares registration,
       referencing top (min) and bottom (max) surfaces.
    3. The median z-values (medVZmin, medVZmax) are recorded, which often serve as
       reference planes in further analyses.
    4. When the mapping carries the volume's z-range or band support masks,
       ``extra["coverage"]`` holds `node_coverage`'s record for every node, plus
       ``depth``: the depth range the volume covers, in the output's z units.
       `get_coverage` summarises it per skeleton.
    """

    nodes = (
        skel.nodes.astype(float) * skeleton_nodes_scale
    )  # scale to the surface unit, which is often µm

    if verbose:
        print("[pywarper] Warping skeleton...")
        start_time = time.time()
    warped_nodes, med_z_on, med_z_off = warp_nodes(
        nodes,
        surface_mapping,
        conformal_jump=conformal_jump,
        backward_compatible=backward_compatible,
    )

    normalized_nodes = normalize_nodes(
        warped_nodes,
        med_z_on=med_z_on,
        med_z_off=med_z_off,
        on_sac_pos=on_sac_pos,
        off_sac_pos=off_sac_pos,
    )

    normalized_nodes /= skeleton_nodes_scale

    if verbose:
        print(f"    done in {time.time() - start_time:.2f} seconds.")

    normalized_soma = deepcopy(skel.soma)
    normalized_soma.center = (
        normalized_nodes[0] * voxel_resolution
    )  # soma is at the first node

    skel_norm = Skeleton(
        soma=normalized_soma,
        nodes=normalized_nodes * voxel_resolution,
        edges=skel.edges,
        radii=skel.radii,
        ntype=skel.ntype,
        node2verts=skel.node2verts,
        vert2node=skel.vert2node,
        meta=skel.meta.copy(),
    )

    z_profiles = {
        measure: get_z_profile(
            skel_norm,
            extent=z_profile_extent,
            bin_size=z_profile_bin_size,
            hdr_mass=z_profile_hdr_mass,
            measure=measure,
            include_soma=z_profile_include_soma,
            voxel_size=z_profile_voxel_size,
            radius_metric=radius_metric,
        )
        for measure in ["length", "volume"]
    }
    xy_profiles = {
        measure: get_xy_profile(
            skel_norm,
            extents=xy_profile_extents,
            bin_size=xy_profile_bin_size,
            smooth=xy_profile_smooth,
            measure=measure,
            include_soma=xy_profile_include_soma,
            voxel_size=xy_profile_voxel_size,
            radius_metric=radius_metric,
        )
        for measure in ["length", "volume"]
    }

    skel_norm.extra = {
        "prenormed_nodes": warped_nodes
        * voxel_resolution,  # keep the pre-normed warped nodes for future use
        "med_z_on": float(med_z_on),
        "med_z_off": float(med_z_off),
        # what `renormalize` needs to put renormalized nodes in these same units
        "voxel_resolution": np.broadcast_to(
            np.asarray(voxel_resolution, dtype=float), (3,)
        ).copy(),
        "skeleton_nodes_scale": float(skeleton_nodes_scale),
        "z_profiles": z_profiles,
        "xy_profiles": xy_profiles,
    }
    coverage = node_coverage(
        nodes, surface_mapping, backward_compatible=backward_compatible
    )
    if coverage is not None:
        if "depth_rel" in coverage:
            coverage["depth"] = _covered_depth(
                coverage["depth_rel"],
                on_sac_pos,
                off_sac_pos,
                skeleton_nodes_scale,
                skel_norm.extra["voxel_resolution"][2],
            )
        skel_norm.extra["coverage"] = coverage
    skel_norm.meta.update(
        {
            "pywarper_version": _PYWARPER_VERSION,
            "warped_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
    )

    return skel_norm


def warp_mesh(
    mesh: "trimesh.Trimesh | sk.Mesh",  # mostly nm
    surface_mapping: dict,  # mostly µm
    conformal_jump: int | None = None,
    on_sac_pos: float = 0.0,  # µm
    off_sac_pos: float = 12.0,  # µm
    mesh_vertices_scale: float = 1.0,  # scale factor for mesh vertices, e.g., 1e-3 for nm to µm
    backward_compatible: bool = False,
    verbose: bool = False,
) -> trimesh.Trimesh:
    """
    Applies a local surface flattening (warp) to a 3D mesh using the results
    of previously computed surface mappings.

    `mesh` is a trimesh mesh or a skeliner one (what `sk.io.load_mesh` returns).
    The result is a trimesh mesh with the same faces; its ``metadata`` holds the
    input's, when it has any, and the record of the warp.
    """

    vertices = (
        np.asarray(mesh.vertices, dtype=float) * mesh_vertices_scale
    )  # scale to the surface unit, which is often µm

    if verbose:
        print("[pywarper] Warping mesh...")
        start_time = time.time()
    warped_vertices, med_z_on, med_z_off = warp_nodes(
        vertices,
        surface_mapping,
        conformal_jump=conformal_jump,
        backward_compatible=backward_compatible,
    )

    normalized_vertices = normalize_nodes(
        warped_vertices,
        med_z_on=med_z_on,
        med_z_off=med_z_off,
        on_sac_pos=on_sac_pos,
        off_sac_pos=off_sac_pos,
    )

    if verbose:
        print(f"    done in {time.time() - start_time:.2f} seconds.")

    # Create a new mesh with the warped vertices
    warped_mesh = trimesh.Trimesh(
        vertices=normalized_vertices
        / mesh_vertices_scale,  # rescale back to original units
        faces=mesh.faces,
        process=False,  # no processing
    )
    warped_mesh.metadata = dict(getattr(mesh, "metadata", None) or {})
    warped_mesh.metadata["med_z_on"] = float(med_z_on)
    warped_mesh.metadata["med_z_off"] = float(med_z_off)
    warped_mesh.metadata["conformal_jump"] = conformal_jump
    warped_mesh.metadata["surface_mapping"] = surface_mapping
    warped_mesh.metadata["on_sac_pos"] = on_sac_pos
    warped_mesh.metadata["off_sac_pos"] = off_sac_pos
    coverage = node_coverage(
        vertices, surface_mapping, backward_compatible=backward_compatible
    )
    if coverage is not None:
        if "depth_rel" in coverage:
            coverage["depth"] = _covered_depth(
                coverage["depth_rel"], on_sac_pos, off_sac_pos, mesh_vertices_scale
            )
        warped_mesh.metadata["coverage"] = coverage
    warped_mesh.metadata["pywarper_meta"] = {
        "version": _PYWARPER_VERSION,
        "warped_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    return warped_mesh


# =====================================================================
# helpers for get_z_profile() and get_xy_profile()
# =====================================================================


def segment_lengths(skel: Skeleton) -> tuple[np.ndarray, np.ndarray]:
    """
    Per-edge cable length at every non-root node & its mid-point.

    Returns
    -------
    lengths : (N,)  nonzero only at child nodes (each entry is edge length)
    mid     : (N,3) midpoints (same convention as before)
    """
    parent = np.asarray(
        _bfs_parents(skel.edges, len(skel.nodes), root=0), dtype=np.int64
    )
    child = np.where(parent != -1)[0]  # nodes with a parent

    a = skel.nodes[child]  # child coords
    b = skel.nodes[parent[child]]  # parent coords
    vec = b - a
    L = np.linalg.norm(vec, axis=1)  # edge length

    # midpoints in a full (N,3) array
    mid = skel.nodes.copy()
    mid[child] += 0.5 * vec

    lengths = np.zeros(len(skel.nodes), dtype=float)
    lengths[child] = L
    return lengths, mid


def _volume_grid(
    skel: Skeleton,
    *,
    voxel_size: float | None,
    include_soma: bool,
    radius_metric: str | None,
) -> tuple[dict, dict]:
    """
    skeliner's voxel grid over the whole morphology, and the arguments that
    give the same union of edges and soma on a box of that grid.

    The grid is `sk.dx.volume`'s record of one call: ``voxel_size``,
    ``bbox_lo``, ``grid_shape`` and ``filled_voxels``.
    """
    union = {
        "radius_metric": radius_metric or skel.recommend_radius()[0],
        "include_soma": include_soma,
    }
    _, grid = sk.dx.volume(skel, voxel_size=voxel_size, return_details=True, **union)
    return grid, union


def _filled_voxels(skel: Skeleton, grid: dict, union: dict, start, count) -> int:
    """Filled voxels in the block of `grid` that starts at voxel `start` and
    holds `count` voxels along each axis."""
    start, count = np.asarray(start), np.asarray(count)
    h = grid["voxel_size"]
    lo = np.asarray(grid["bbox_lo"], dtype=float) + start * h
    # the box ends inside its last voxels, past their centres, so that
    # rounding cannot give it one voxel more
    hi = lo + (count - 0.25) * h
    _, block = sk.dx.volume(
        skel, bbox=(lo, hi), voxel_size=h, return_details=True, **union
    )
    if block["voxel_size"] == h and tuple(block["grid_shape"]) == tuple(count):
        return int(block["filled_voxels"])
    # skeliner coarsens a box that holds more voxels than it takes in one grid
    axis = int(np.argmax(count))
    if count[axis] < 2:
        raise RuntimeError("skeliner did not voxelize a box on the grid it was given.")
    near, far = count.copy(), count.copy()
    near[axis] = count[axis] // 2
    far[axis] = count[axis] - near[axis]
    beyond = start.copy()
    beyond[axis] += near[axis]
    return _filled_voxels(skel, grid, union, start, near) + _filled_voxels(
        skel, grid, union, beyond, far
    )


def z_slice_volumes(
    skel: Skeleton,
    *,
    voxel_size: float | None = None,
    include_soma: bool = False,
    radius_metric: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Union volume of the morphology in each layer of voxels along z.

    Returns
    -------
    volumes : (K,) float
        Volume in each layer that holds any (in unit³).
    z : (K,) float
        The layers' centres.
    """
    grid, union = _volume_grid(
        skel,
        voxel_size=voxel_size,
        include_soma=include_soma,
        radius_metric=radius_metric,
    )
    h = grid["voxel_size"]
    nx, ny, nz = grid["grid_shape"]
    filled = np.array(
        [_filled_voxels(skel, grid, union, (0, 0, k), (nx, ny, 1)) for k in range(nz)]
    )
    z = grid["bbox_lo"][2] + (np.arange(nz) + 0.5) * h
    keep = filled > 0
    return filled[keep].astype(np.float64) * (h**3), z[keep]


def _xy_bin_volumes(
    skel: Skeleton, grid: dict, union: dict, x_edges: np.ndarray, y_edges: np.ndarray
) -> np.ndarray:
    """Union volume in each bin of an xy histogram: a column of voxels belongs
    to the bin `np.histogram2d` would put its centre in."""
    h = grid["voxel_size"]
    nz = grid["grid_shape"][2]
    spans = []
    for axis, edges in enumerate((x_edges, y_edges)):
        centres = (
            grid["bbox_lo"][axis] + (np.arange(grid["grid_shape"][axis]) + 0.5) * h
        )
        first = np.searchsorted(centres, edges[:-1], side="left")
        last = np.append(first[1:], np.searchsorted(centres, edges[-1], side="right"))
        spans.append((first, np.maximum(last, first)))
    (x0, x1), (y0, y1) = spans
    filled = np.zeros((len(x0), len(y0)))
    for i in np.flatnonzero(x1 > x0):
        for j in np.flatnonzero(y1 > y0):
            filled[i, j] = _filled_voxels(
                skel,
                grid,
                union,
                (x0[i], y0[j], 0),
                (x1[i] - x0[i], y1[j] - y0[j], nz),
            )
    return filled * (h**3)


def _xy_volume_histogram(
    skel: Skeleton,
    extents: list[float | int] | None,
    bin_size: float,
    *,
    voxel_size: float | None,
    include_soma: bool,
    radius_metric: str | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Union volume per xy bin, the bins' edges along x and y, and the volume
    of the whole morphology. Without `extents` the window is the whole bins
    around the nodes and the filled voxel columns."""
    grid, union = _volume_grid(
        skel,
        voxel_size=voxel_size,
        include_soma=include_soma,
        radius_metric=radius_metric,
    )
    h = grid["voxel_size"]
    total = float(grid["filled_voxels"]) * (h**3)

    def histogram(lo, hi):
        x_edges = _edges_from_bin_size(lo[0], hi[0], bin_size)
        y_edges = _edges_from_bin_size(lo[1], hi[1], bin_size)
        return _xy_bin_volumes(skel, grid, union, x_edges, y_edges), x_edges, y_edges

    if extents is not None:
        xmin, xmax, ymin, ymax = extents
        return *histogram((xmin, ymin), (xmax, ymax)), total

    # whole bins around every node and every voxel column of the grid ...
    nodes_lo = np.floor(skel.nodes[:, :2].min(axis=0) / bin_size)
    nodes_hi = np.ceil(skel.nodes[:, :2].max(axis=0) / bin_size)
    shape = np.asarray(grid["grid_shape"][:2])
    centre = np.asarray(grid["bbox_lo"][:2], dtype=float) + 0.5 * h
    low = np.minimum(np.floor(centre / bin_size), nodes_lo)
    high = np.maximum(np.ceil((centre + (shape - 1) * h) / bin_size), nodes_hi)
    hist, x_edges, y_edges = histogram(low * bin_size, high * bin_size)
    # ... then without the outer bins that hold neither a node nor volume
    tight_low, tight_high = nodes_lo.copy(), nodes_hi.copy()
    for axis in (0, 1):
        held = np.flatnonzero(hist.sum(axis=1 - axis) > 0)
        if held.size:
            tight_low[axis] = min(tight_low[axis], low[axis] + held[0])
            tight_high[axis] = max(tight_high[axis], low[axis] + held[-1] + 1)
    if np.any(tight_low != low) or np.any(tight_high != high):
        hist, x_edges, y_edges = histogram(tight_low * bin_size, tight_high * bin_size)
    return hist, x_edges, y_edges, total


def gridder1d(
    z_samples: np.ndarray,
    density: np.ndarray,
    n: int,
) -> np.ndarray:
    """
    Kaiser–Bessel gridding kernel in 1-D   (alpha=2, W=5)

    Vectorised patch-accumulation: identical output, ~2× faster.
    """
    if z_samples.shape != density.shape:
        raise ValueError("z_samples and density must have the same shape")

    # ------------------------------------------------------------------
    # Constants and lookup table (unchanged)
    # ------------------------------------------------------------------
    alpha, W, err = 2, 5, 1e-3
    S = int(np.ceil(0.91 / err / alpha))
    beta = np.pi * np.sqrt((W / alpha * (alpha - 0.5)) ** 2 - 0.8)

    s = np.linspace(-1, 1, 2 * S * W + 1)
    F_kbZ = i0(beta * np.sqrt(1 - s**2))
    F_kbZ /= F_kbZ.max()

    # ------------------------------------------------------------------
    # Fourier transform of the 1-D kernel (unchanged)
    # ------------------------------------------------------------------
    Gz = alpha * n
    z = np.arange(-Gz // 2, Gz // 2)
    arg = (np.pi * W * z / Gz) ** 2 - beta**2

    kbZ = np.empty_like(arg, dtype=float)
    pos, neg = arg > 1e-12, arg < -1e-12
    kbZ[pos] = np.sin(np.sqrt(arg[pos])) / np.sqrt(arg[pos])
    kbZ[neg] = np.sinh(np.sqrt(-arg[neg])) / np.sqrt(-arg[neg])
    kbZ[~(pos | neg)] = 1.0
    kbZ *= np.sqrt(Gz)

    # ------------------------------------------------------------------
    # Oversampled grid and *vectorised* accumulation
    # ------------------------------------------------------------------
    n_os = Gz
    out = np.zeros(n_os, dtype=float)

    centre = n_os / 2 + 1  # 1-based like MATLAB
    nz = centre + n_os * z_samples  # (N,)

    half_w = (W - 1) // 2
    lz_offsets = np.arange(-half_w, half_w + 1)  # (W,)

    # shape manipulations so that the first index is lz (to keep
    # addition order identical to the original loop)
    nz_mat = nz[None, :] + lz_offsets[:, None]  # (W, N)
    nzt = np.round(nz_mat).astype(int)  # (W, N)
    zpos_mat = S * ((nz[None, :] - nzt) + W / 2)  # (W, N)
    kw_mat = F_kbZ[np.round(zpos_mat).astype(int)]  # (W, N)

    nzt_clipped = np.clip(nzt, 0, n_os - 1)  # (W, N)
    np.add.at(
        out,
        nzt_clipped.ravel(order="C"),  # lz-major order
        (density[None, :] * kw_mat).ravel(order="C"),
    )

    out[0] = out[-1] = 0.0  # edge artefacts

    # ------------------------------------------------------------------
    # myifft  ->  de-apodise  ->  abs(myfft3)  (unchanged)
    # ------------------------------------------------------------------
    u = n
    f = np.fft.ifftshift(np.fft.ifft(np.fft.ifftshift(out))) * np.sqrt(u)
    f = f[int(np.ceil((f.size - u) / 2)) : int(np.ceil((f.size + u) / 2))]
    f /= kbZ[u // 2 : 3 * u // 2]

    F = np.fft.fftshift(np.fft.fftn(np.fft.fftshift(f))) / np.sqrt(f.size)
    return np.abs(F)


# =====================================================================
# helpers for get_z_profile() END
# =====================================================================


def _split_path(
    path: list[int], xyz: np.ndarray, split_length: float
) -> list[list[int]]:
    """
    Divide an ordered node path into n roughly equal sub-paths.

    n = max(1, round(total_length / split_length)), so a 100 µm branch with
    split_length=30 becomes 3 sub-paths of ~33 µm rather than 30-30-40.
    Adjacent sub-paths share their boundary node.
    """
    coords = xyz[path]
    edge_lens = np.linalg.norm(np.diff(coords, axis=0), axis=1)
    L = float(edge_lens.sum())
    n = max(1, round(L / split_length))
    if n == 1:
        return [path]

    cum = np.concatenate([[0.0], np.cumsum(edge_lens)])
    targets = np.linspace(0.0, L, n + 1)

    # node index in path for each interior cut (skip 0 and L)
    cut_idx = [0]
    for t in targets[1:-1]:
        idx = int(np.argmin(np.abs(cum - t)))
        if idx > cut_idx[-1]:  # avoid duplicate split points
            cut_idx.append(idx)
    cut_idx.append(len(path) - 1)

    return [
        path[cut_idx[i] : cut_idx[i + 1] + 1]
        for i in range(len(cut_idx) - 1)
        if cut_idx[i + 1] > cut_idx[i]
    ]


def _orient_classify(
    skel: Skeleton,
    branch_split_length: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Branch-level orientation classifier.

    For each node i returns:
    - vert[i] : |dz| / L for the (sub-)segment containing edge (i -> parent[i]).
                NaN for the root (no parent).
    - tip[i]  : True if node i is in a terminal branch (ends at a leaf).

    A **branch** is a maximal chain of nodes where every interior node has
    exactly one child.  If branch_split_length is given, branches longer than
    that are split into n = round(L / branch_split_length) equal sub-segments
    so that orientation is evaluated at a consistent spatial scale.  Tip
    detection always uses the full branch length, not the sub-segment length.
    """
    raw_parent = _bfs_parents(skel.edges, len(skel.nodes), root=0)
    parent = np.asarray(raw_parent, dtype=np.int64)
    xyz = np.asarray(skel.nodes, dtype=np.float64)
    N = len(xyz)

    n_ch = np.zeros(N, dtype=int)
    child_nodes = np.where(parent != -1)[0]
    if len(child_nodes) == 0:
        return np.full(N, np.nan), np.zeros(N, dtype=bool)
    np.add.at(n_ch, parent[child_nodes], 1)

    ch: list[list[int]] = [[] for _ in range(N)]
    for c in map(int, child_nodes):
        ch[int(parent[c])].append(c)

    vert = np.full(N, np.nan)
    tip = np.zeros(N, dtype=bool)

    # DFS over branches.  Each stack entry is (first_node, branch_top_node).
    # first_node is a direct child of branch_top and is the entry to a new branch.
    stack: list[tuple[int, int]] = [(c, 0) for c in ch[0]]

    while stack:
        first, bstart = stack.pop()

        # Walk the single-child chain to the branch bottom (junction node).
        current = first
        while n_ch[current] == 1:
            current = ch[current][0]
        bend = current

        # Reconstruct ordered branch path by tracing back from bend to bstart.
        path: list[int] = [bend]
        cur = int(bend)
        while cur != bstart:
            cur = int(parent[cur])
            path.append(cur)
        path.reverse()  # [bstart, ..., bend]

        if len(path) >= 2:
            # All terminal branches are tips regardless of length or orientation.
            is_term = bool(n_ch[bend] == 0)
            qualifies_as_tip = is_term

            # Split into sub-segments (or keep as one if no split_length given).
            subs = (
                _split_path(path, xyz, branch_split_length)
                if branch_split_length is not None
                else [path]
            )

            for sub in subs:
                coords = xyz[sub]
                L_sub = float(np.linalg.norm(np.diff(coords, axis=0), axis=1).sum())
                dz_sub = abs(float(coords[-1, 2] - coords[0, 2]))
                bvert = dz_sub / L_sub if L_sub > 0.0 else 0.0

                for node in sub[1:]:  # child nodes in this sub-segment
                    vert[node] = bvert
                    if qualifies_as_tip:
                        tip[node] = True

        # Continue DFS into branches below branch points.
        if n_ch[bend] >= 2:
            for c in ch[bend]:
                stack.append((c, bend))

    return vert, tip


def get_z_profile(
    skel: Skeleton,
    extent: list[float | int] | None = None,
    bin_size: float = 1,  # µm
    hdr_mass: float = 0.95,
    *,
    measure: str = "length",  # ["length", "volume"]
    radius_metric: str | None = None,
    voxel_size: float | None = None,  # only used for volume (union)
    include_soma: bool = False,
    orientation_threshold: float | None = None,
    branch_split_length: float | None = None,  # µm
) -> dict:
    """
    Compute a 1-D depth profile.

    measure:
        "length" – cable length per bin (histogram + KB-smoothed).
        "volume" – **union-correct** morphology volume per bin (voxel union).

    orientation_threshold:
        If set (float in [0, 1]), also compute split profiles for
        measure='length':
          vlength – edges where branch |dz|/L > threshold (vertical /
                    layer-bridging).
          hlength – all remaining edges + tip segments (horizontal /
                    synapse-making).
        The threshold is compared against the (sub-)segment |dz|/L.
        All terminal branches (leaf nodes) always go into hlength regardless
        of their orientation.
        branch_split_length: if set, each branch is divided into
        n = round(L / branch_split_length) equal sub-segments so that
        orientation is evaluated at a consistent spatial scale rather than
        over the full (possibly very long) branch.

    Notes
    -----
    * volume is skeliner's voxel union (`sk.dx.volume`): branch junctions and
      soma contacts are **not counted twice**.
    * 'include_soma' defaults to False to match the 'length' convention
      (edges only). Set True if you want soma membrane/volume included.
    * For stable plots, keep bin_size >= voxel_size (if you set voxel_size).
    * orientation_threshold is only supported for measure='length'.
    """

    if measure == "length":
        density, mid = segment_lengths(skel)
        z_vals = mid[:, 2]
    elif measure == "volume":
        density, z_vals = z_slice_volumes(
            skel,
            voxel_size=voxel_size,
            include_soma=include_soma,
            radius_metric=radius_metric,
        )
    else:
        raise ValueError("measure must be one of {'length','volume'}")

    # window
    if extent is None:
        # a morphology without volume still gets the window of its nodes
        spread = z_vals if z_vals.size else skel.nodes[:, 2]
        z_min, z_max = np.floor(spread.min()), np.ceil(spread.max())
        # a skeleton lying exactly on one integer depth still gets one bin
        z_max = max(z_max, z_min + bin_size)
    else:
        z_min, z_max = extent

    # histogram bins
    n_bins = max(1, int(np.ceil((z_max - z_min) / bin_size)))
    edges = z_min + np.arange(n_bins + 1) * bin_size
    edges[-1] = z_max
    x_um = 0.5 * (edges[1:] + edges[:-1])

    # shared KB-smoothing parameters
    centre = (z_min + z_max) / 2.0
    halfspan = (z_max - z_min) / 2.0
    z_samples = (z_vals - centre) / max(halfspan, np.finfo(float).eps)

    def _profile(w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Histogram + KB-smoothed profile for weight array w."""
        tot = float(w.sum())
        if tot == 0.0:
            return np.zeros(n_bins), np.zeros(n_bins)
        hist, _ = np.histogram(z_vals, bins=edges, weights=w)
        if hist.sum() > 0:
            hist = hist * (tot / hist.sum())
        dist = gridder1d(z_samples / 2.0, w, n_bins)
        if dist.sum() > 0:
            dist = dist * (tot / dist.sum())
        return hist, dist

    z_hist, z_dist = _profile(density)
    intervals = hdr(x_um, z_dist, mass=hdr_mass)
    unit = skel.meta.get("unit", "µm")

    result = {
        "x": x_um,
        "distribution": z_dist,
        "histogram": z_hist,
        "extent": [z_min, z_max],
        "n_bins": n_bins,
        "bin_size": bin_size,
        "hdr": intervals,
        "hdr_mass": hdr_mass,
        "measure": measure,
        "y_units": unit if measure == "length" else f"{unit}³",
        "include_soma": include_soma,
        "voxel_size": voxel_size,
        "radius_metric": radius_metric,
    }

    # Orientation split (length only)
    if orientation_threshold is not None:
        if measure != "length":
            raise ValueError("orientation_threshold requires measure='length'")

        vert_vals, tip_mask = _orient_classify(skel, branch_split_length)
        # vertical: not a tip AND branch |dz|/L > threshold
        mask_v = (
            np.isfinite(vert_vals) & (vert_vals > orientation_threshold) & ~tip_mask
        )

        v_hist, v_dist = _profile(density * mask_v)
        h_hist, h_dist = _profile(density * ~mask_v)

        result.update(
            {
                "vlength_distribution": v_dist,
                "vlength_histogram": v_hist,
                "hlength_distribution": h_dist,
                "hlength_histogram": h_hist,
                "orientation_threshold": orientation_threshold,
                "branch_split_length": branch_split_length,
            }
        )

    return result


def _edges_from_bin_size(lo: float, hi: float, bin_size: float) -> np.ndarray:
    """Generate edges >= bin_size wide, last bin clipped to *hi*."""
    n = int(np.ceil((hi - lo) / bin_size))
    edges = lo + np.arange(n + 1) * bin_size
    edges[-1] = hi  # ensure inclusion
    eps = np.finfo(float).eps
    edges[-1] += eps * max(1.0, abs(edges[-1]))  # numeric cushion
    return edges


def get_xy_profile(
    skel: Skeleton,
    extents: list[float | int] | None = None,
    bin_size: float | int = 2.0,
    smooth: float | int = 1.0,
    *,
    measure: str = "length",  # {"length","volume"}
    radius_metric: str | None = None,
    voxel_size: float | None = None,
    include_soma: bool = False,
) -> dict:
    """
    Planar (x-y) dendritic-length density on a **square** grid.

    Parameters
    ----------
    extents
        Fixed window ``[xmin, xmax, ymin, ymax]`` (µm).  *None* -> tight box.
    bin_size
        Side length of each square bin (µm).  `edges = lo + k·bin_size`.
        The window is *expanded* to the next multiple so that every bin is
        exactly `bin_size` wide.  (This guarantees comparability.)
    smooth
        sigma of the Gaussian kernel (bins) applied to the histogram.
    measure:
        "length" – dendritic cable length per bin (histogram -> Gaussian smooth).
        "volume" – **union-correct** morphology volume per bin (voxel union),
                   integrated along z, then binned in x-y. Every bin that
                   holds voxels is one call into skeliner: many small bins
                   take long.

    Returns
    -------
    dict  with keys ``x`` / ``y`` (centres), ``distribution`` (smoothed),
           ``histogram`` (raw), ``extents``, ``bin_size``, ``nbins`` …
    """

    if measure == "length":
        density, mid = segment_lengths(skel)
        units = skel.meta.get("unit", "µm")
        total = float(density.sum())

        if extents is None:
            # tight box around the edges' midpoints and the nodes, in whole bins
            x_all = np.concatenate((mid[:, 0], skel.nodes[:, 0]))
            y_all = np.concatenate((mid[:, 1], skel.nodes[:, 1]))
            xmin = np.floor(x_all.min() / bin_size) * bin_size
            xmax = np.ceil(x_all.max() / bin_size) * bin_size
            ymin = np.floor(y_all.min() / bin_size) * bin_size
            ymax = np.ceil(y_all.max() / bin_size) * bin_size
        else:
            xmin, xmax, ymin, ymax = extents

        # every cell is square (bin_size × bin_size)
        x_edges = _edges_from_bin_size(xmin, xmax, bin_size)
        y_edges = _edges_from_bin_size(ymin, ymax, bin_size)
        xy_hist, _, _ = np.histogram2d(
            mid[:, 0], mid[:, 1], bins=[x_edges, y_edges], weights=density
        )
    elif measure == "volume":
        units = f"{skel.meta.get('unit', 'µm')}³"
        xy_hist, x_edges, y_edges, total = _xy_volume_histogram(
            skel,
            extents,
            bin_size,
            voxel_size=voxel_size,
            include_soma=include_soma,
            radius_metric=radius_metric,
        )
    else:
        raise ValueError("measure must be one of {'length','volume'}")

    xy_dist = gaussian_filter(xy_hist, sigma=smooth, mode="nearest")
    if xy_dist.sum() > 0:
        xy_dist *= total / xy_dist.sum()

    x = 0.5 * (x_edges[:-1] + x_edges[1:])
    y = 0.5 * (y_edges[:-1] + y_edges[1:])

    return {
        "x": x,
        "y": y,
        "distribution": xy_dist,
        "histogram": xy_hist,
        "extents": [x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]],
        "n_bins": (len(x), len(y)),  # may differ if dx != dy
        "bin_size": bin_size,
        "smooth": smooth,
        "measure": measure,
        "units": units,
        "include_soma": include_soma,
        "voxel_size": voxel_size,
        "radius_metric": radius_metric,
    }


def get_coverage(skel: Skeleton, extent: list[float] | tuple[float, float]) -> dict:
    """
    How much of a warped skeleton's cable lies where the data do not cover it.

    Parameters
    ----------
    skel : Skeleton
        Output of `warp_skeleton` or `Warper.renormalize`.
    extent : (z_min, z_max)
        Depth window in the warped skeleton's z units, e.g. its z-profile extent.

    Returns
    -------
    dict
        Shares of the cable length, each NaN when the skeleton has no record of it
        (see `node_coverage`):

        ``cut_below`` / ``cut_above``
            The imaged volume ends inside the window, below z_min / above z_max,
            in that cable's column.
        ``on_unsupported`` / ``off_unsupported``
            That band was not observed in the cable's column.
        ``<name>_unsupported``
            Same for each extra layer the skeleton has a record of.
    """
    out = {
        k: np.nan
        for k in ("cut_below", "cut_above", "on_unsupported", "off_unsupported")
    }
    lengths, _ = segment_lengths(skel)
    total = lengths.sum()
    coverage = (skel.extra or {}).get("coverage") or {}
    if total <= 0:
        return out
    z_min, z_max = extent
    if "depth" in coverage:
        depth = np.asarray(coverage["depth"], dtype=float)
        out["cut_below"] = float(lengths[depth[:, 0] > z_min].sum() / total)
        out["cut_above"] = float(lengths[depth[:, 1] < z_max].sum() / total)
    for band in ("on", "off"):
        if f"{band}_support" in coverage:
            support = np.asarray(coverage[f"{band}_support"], dtype=bool)
            out[f"{band}_unsupported"] = float(lengths[~support].sum() / total)
    if "extra_support" in coverage:
        support = np.asarray(coverage["extra_support"], dtype=bool)
        for k, name in enumerate(coverage["extra_names"]):
            out[f"{name}_unsupported"] = float(lengths[~support[:, k]].sum() / total)
    return out


def hdr(z_centres, z_density, mass=0.95):
    """
    High Density Region (HDR) estimator.

    Parameters
    ----------
    z_centres : (N,) bin centres (µm)
    z_density : (N,) density per bin (any units)
    mass      : float, 0 < mass <= 1 (e.g. 0.95)

    Example
    -------
    >>> hdr(z, d, 0.95)
    [array([ -2.1,  1.7]),   # ON sheet
     array([ 10.3, 13.9])]   # OFF sheet

    A profile without mass has no high-density region: the result is empty.
    """
    total = z_density.sum()
    if not total > 0:
        return []
    p = z_density / total  # normalise -> probability
    order = np.argsort(p)[::-1]  # bins from high to low density

    selected = []
    cum = 0.0
    for idx in order:
        selected.append(idx)
        cum += p[idx]
        if cum >= mass:
            break

    sel = np.sort(selected)  # ascending bin indices
    # split where gaps > 1 bin
    gaps = np.where(np.diff(sel) > 1)[0]
    groups = [g for g in np.split(sel, gaps + 1)]

    intervals = [
        np.array([z_centres[g[0]], z_centres[g[-1]]]) for g in groups if len(g) > 0
    ]
    return intervals


class Warper:
    """High-level interface around *pywarper* for IPL flattening."""

    def __init__(
        self,
        off_sac_points: dict[str, np.ndarray]
        | tuple[np.ndarray, np.ndarray, np.ndarray]
        | None = None,
        on_sac_points: dict[str, np.ndarray]
        | tuple[np.ndarray, np.ndarray, np.ndarray]
        | None = None,
        swc_path: str | None = None,
        *,
        voxel_resolution: list[float] = [1.0, 1.0, 1.0],
        verbose: bool = False,
    ) -> None:
        self.voxel_resolution = voxel_resolution
        self.verbose = verbose
        self.swc_path = swc_path
        # what fit_surfaces() settled on, per band; folded into the mapping meta
        self.surface_fit: dict[str, dict] = {}

        if off_sac_points is not None:
            self.off_sac_points = self._as_xyz(off_sac_points)
        if on_sac_points is not None:
            self.on_sac_points = self._as_xyz(on_sac_points)

        if swc_path is not None:
            self.swc_path = swc_path
            self.load_swc(swc_path)  # raw SWC -> self.nodes / edges / radii
        else:
            self.swc_path = None

    # ---------------------------- IO -------------------------------------
    def load_swc(self, swc_path: str | None = None) -> "Warper":
        """Load the skeleton from *swc_path*."""

        if self.verbose:
            print(f"[pywarper] Loading skeleton -> {self.swc_path}")

        if swc_path is None:
            swc_path = self.swc_path

        if swc_path is not None:
            self.skeleton = sk.io.load_skeleton_swc(swc_path)
        else:
            raise ValueError("SWC path must be provided to load the skeleton.")

        return self

    @staticmethod
    def _as_xyz(data) -> tuple[np.ndarray, np.ndarray, np.ndarray]:  # for load_sac()
        """Accept *dict* or tuple and return *(x, y, z)* numpy arrays."""
        if isinstance(data, dict):
            return np.asarray(data["x"]), np.asarray(data["y"]), np.asarray(data["z"])
        if isinstance(data, (tuple, list)) and len(data) == 3:
            return map(np.asarray, data)  # type: ignore[arg-type]
        raise TypeError(
            "SAC data must be a mapping with keys x/y/z or a 3-tuple of arrays."
        )

    def load_sac(self, off_sac_points, on_sac_points) -> "Warper":
        """Load the SAC meshes from *off_sac_points* and *on_sac_points*."""
        if self.verbose:
            print("[pywarper] Loading SAC meshes...")
        self.off_sac_points = self._as_xyz(off_sac_points)
        self.on_sac_points = self._as_xyz(on_sac_points)
        return self

    def load_warped_skeleton(
        self,
        filepath: str,
        med_z_on: float | None = None,
        med_z_off: float | None = None,
    ) -> None:
        """Load a warped skeleton from *swc_path*."""
        path = Path(filepath)

        if path.suffix.lower() == ".swc":
            self.warped_skeleton = sk.io.load_skeleton_swc(path)

            if (med_z_on is not None) and (med_z_off is not None):
                self.warped_skeleton.extra["med_z_on"] = float(med_z_on)
                self.warped_skeleton.extra["med_z_off"] = float(med_z_off)
            else:
                self.warped_skeleton.extra["med_z_on"] = None
                self.warped_skeleton.extra["med_z_off"] = None
        elif path.suffix.lower() == ".npz":
            self.warped_skeleton = sk.io.load_skeleton_npz(path)

        if self.verbose:
            print(f"[pywarper] Loaded warped skeleton -> {path}")

    # ---------------------------- Core -----------------------------------

    def fit_surfaces(
        self,
        xmax: int | float | None = None,
        ymax: int | float | None = None,
        method: str = "gridfit",
        stride: int | None = None,
        smoothness: int | float | str | None = None,
        gamma: float | None = None,
        k: int | tuple[int, int] | None = None,
        bs: str | None = None,
        backward_compatible: bool = False,
    ) -> "Warper":
        """Fit the ON / OFF SAC meshes.

        Parameters
        ----------
        method : {"gridfit", "gam"}, default="gridfit"
            Which fitter to use, see :func:`pywarper.surface.fit_sac_surface`.
        stride, smoothness : optional
            ``"gridfit"`` settings; default to 3 and 15 here. Pass
            ``smoothness="auto"`` to select it from the data instead -- what each
            band settled on lands in ``self.surface_fit[band]``, since the two are
            fitted separately and need not agree. Rejected under ``"gam"``.
        gamma : float, optional
            ``"gridfit"`` under ``smoothness="auto"`` only; the length scale the
            selection is optimal for. See :func:`pywarper.surface.fit_sac_surface`.
        k, bs : optional
            ``"gam"`` settings; default to 100 and "tp". Rejected under
            ``"gridfit"``.
        """
        if self.verbose:
            print("[pywarper] Fitting SAC surfaces...")

        if backward_compatible is False and (xmax is None or ymax is None):
            # use the bounding box of the skeleton
            xmax = max(self.off_sac_points[0].max(), self.on_sac_points[0].max())
            ymax = max(self.off_sac_points[1].max(), self.on_sac_points[1].max())

        if method == "gridfit":
            stride = 3 if stride is None else stride
            smoothness = 15 if smoothness is None else smoothness

        self.surface_fit = {}
        for band, points, label in (
            ("off", self.off_sac_points, "OFF (max)"),
            ("on", self.on_sac_points, "ON (min)"),
        ):
            _t0 = time.time()
            surface = fit_sac_surface(
                x=points[0],
                y=points[1],
                z=points[2],
                xmax=xmax,
                ymax=ymax,
                method=method,
                stride=stride,
                smoothness=smoothness,
                gamma=gamma,
                k=k,
                bs=bs,
                backward_compatible=backward_compatible,
            )
            setattr(self, f"{band}_sac_surface", surface.zmesh)
            setattr(self, f"{band}_sac_surface_se", surface.se)
            self.surface_fit[band] = surface.summary
            if self.verbose:
                print(
                    f"-> fitting {label} surface\n    done in {time.time() - _t0:.2f} seconds."
                )
        return self

    def build_mapping(
        self,
        bounds: np.ndarray | tuple | str | None = "local",
        conformal_jump: int = 2,
        n_anchors: int = 16,
        backward_compatible: bool = False,
        *,
        volume_z: tuple[float | None, float | None] | None = None,
        on_support: np.ndarray | None = None,
        off_support: np.ndarray | None = None,
        extra_surfaces: dict | None = None,
    ) -> "Warper":
        """Create the quasi-conformal surface mapping.

        `volume_z`, `on_support`, `off_support` and `extra_surfaces` are passed
        on to :func:`pywarper.surface.build_mapping`; warped skeletons then
        record their coverage and take their depth from every layer.
        """
        if self.off_sac_surface is None or self.on_sac_surface is None:
            raise RuntimeError("Surfaces not fitted. Call fit_surfaces() first.")

        if bounds is None or bounds == "local":
            # skeleton-derived box (rounded to int so it plays nicely with
            # backward-compatible 1-based code paths)
            xmin, xmax = (
                self.skeleton.nodes[:, 0].min(),
                self.skeleton.nodes[:, 0].max(),
            )
            ymin, ymax = (
                self.skeleton.nodes[:, 1].min(),
                self.skeleton.nodes[:, 1].max(),
            )
            bounds = np.array([xmin, xmax, ymin, ymax], dtype=float)
        elif bounds == "global":
            # use whichever SAC fit is larger in each axis
            nx = max(self.on_sac_surface.shape[0], self.off_sac_surface.shape[0])
            ny = max(self.on_sac_surface.shape[1], self.off_sac_surface.shape[1])
            bounds = np.array([0, nx, 0, ny], dtype=float)
        else:
            bounds = np.asarray(bounds, dtype=float)
            if bounds.shape != (4,):
                raise ValueError(
                    "Bounds must be a 4-element array or tuple (x_min, x_max, y_min, y_max)."
                )

        if self.verbose:
            print("[pywarper] Building mapping...")
        self.mapping: dict = build_mapping(
            self.on_sac_surface,
            self.off_sac_surface,
            bounds,
            conformal_jump=conformal_jump,
            n_anchors=n_anchors,
            volume_z=volume_z,
            on_support=on_support,
            off_support=off_support,
            extra_surfaces=extra_surfaces,
            backward_compatible=backward_compatible,
            verbose=self.verbose,
        )
        if self.surface_fit:
            # so a warped skeleton records how its surfaces were fitted
            self.mapping["meta"]["surface_fit"] = self.surface_fit
        return self

    def warp_skeleton(
        self,
        on_sac_pos: float = 0.0,
        off_sac_pos: float = 12.0,
        z_profile_extent: list[float | int] | None = None,
        z_profile_bin_size: float | int = 1,  # um
        z_profile_hdr_mass: float | int = 0.95,
        z_profile_include_soma: bool = False,
        z_profile_voxel_size: float | None = None,
        xy_profile_extents: list[float | int] | None = None,
        xy_profile_bin_size: float | int = 20,  # um
        xy_profile_smooth: float | int = 1.0,
        xy_profile_include_soma: bool = False,
        xy_profile_voxel_size: float | None = None,
        radius_metric: str | None = None,
        skeleton_nodes_scale: float = 1.0,
        voxel_resolution: list[float | int] | None = None,
        conformal_jump: int | None = None,
        backward_compatible: bool = False,
    ) -> "Warper":
        """Apply the mapping to the skeleton."""
        if self.mapping is None:
            raise RuntimeError("Mapping missing. Call build_mapping() first.")

        if voxel_resolution is None:
            voxel_resolution = self.voxel_resolution

        self.warped_skeleton = warp_skeleton(
            self.skeleton,
            self.mapping,
            on_sac_pos=on_sac_pos,
            off_sac_pos=off_sac_pos,
            voxel_resolution=voxel_resolution,
            conformal_jump=conformal_jump,
            z_profile_extent=z_profile_extent,
            z_profile_bin_size=z_profile_bin_size,
            z_profile_hdr_mass=z_profile_hdr_mass,
            z_profile_include_soma=z_profile_include_soma,
            z_profile_voxel_size=z_profile_voxel_size,
            xy_profile_extents=xy_profile_extents,
            xy_profile_bin_size=xy_profile_bin_size,
            xy_profile_smooth=xy_profile_smooth,
            xy_profile_include_soma=xy_profile_include_soma,
            xy_profile_voxel_size=xy_profile_voxel_size,
            radius_metric=radius_metric,
            backward_compatible=backward_compatible,
            skeleton_nodes_scale=skeleton_nodes_scale,
            verbose=self.verbose,
        )
        return self

    def renormalize(
        self,
        on_sac_pos: float = 0.0,
        off_sac_pos: float = 12.0,
        z_profile_extent: list[float | int] | None = None,  # [z_min, z_max]
        z_profile_bin_size: float | int | None = None,
        z_profile_hdr_mass: float | int | None = None,
        z_profile_include_soma: bool | None = None,
        z_profile_voxel_size: float | None = None,
        xy_profile_extents: list[float | int]
        | None = None,  # [x_min, x_max, y_min, y_max]
        xy_profile_bin_size: float | int | None = None,
        xy_profile_smooth: float | int | None = None,
        xy_profile_include_soma: bool | None = None,
        xy_profile_voxel_size: float | None = None,
        radius_metric: str | None = None,
    ) -> Skeleton:
        """Renormalize the warped skeleton to the desired ON/OFF SAC positions.

        The result is in the same units as the output of `warp_skeleton`: the
        voxel resolution and node scale that the warp recorded are undone before
        renormalizing and applied again after. Skeletons warped before those were
        recorded fall back to this Warper's `voxel_resolution` and a scale of 1.
        """
        if self.warped_skeleton is None:
            raise RuntimeError("Warped skeleton missing. Call warp_skeleton() first.")
        extra = self.warped_skeleton.extra
        voxel_resolution = np.broadcast_to(
            np.asarray(
                extra.get("voxel_resolution", self.voxel_resolution), dtype=float
            ),
            (3,),
        )
        scale = float(extra.get("skeleton_nodes_scale", 1.0))
        renormed_nodes = (
            normalize_nodes(
                np.asarray(extra["prenormed_nodes"], dtype=float) / voxel_resolution,
                med_z_on=extra["med_z_on"],
                med_z_off=extra["med_z_off"],
                on_sac_pos=on_sac_pos,
                off_sac_pos=off_sac_pos,
            )
            / scale
            * voxel_resolution
        )

        soma_renormed = deepcopy(self.warped_skeleton.soma)
        soma_renormed.center = renormed_nodes[0]

        skel_renormed = Skeleton(
            soma=soma_renormed,
            nodes=renormed_nodes,
            edges=self.warped_skeleton.edges,
            radii=self.warped_skeleton.radii,
            ntype=self.warped_skeleton.ntype,
            meta=self.warped_skeleton.meta.copy(),  # copy metadata
        )

        z_profiles = {
            measure: get_z_profile(
                skel_renormed,
                extent=z_profile_extent
                if z_profile_extent is not None
                else self.warped_skeleton.extra["z_profiles"][measure]["extent"],
                bin_size=z_profile_bin_size
                if z_profile_bin_size is not None
                else self.warped_skeleton.extra["z_profiles"][measure]["bin_size"],
                hdr_mass=z_profile_hdr_mass
                if z_profile_hdr_mass is not None
                else self.warped_skeleton.extra["z_profiles"][measure]["hdr_mass"],
                measure=measure,
                include_soma=z_profile_include_soma
                if z_profile_include_soma is not None
                else self.warped_skeleton.extra["z_profiles"][measure]["include_soma"],
                voxel_size=z_profile_voxel_size
                if z_profile_voxel_size is not None
                else self.warped_skeleton.extra["z_profiles"][measure]["voxel_size"],
                radius_metric=radius_metric
                if radius_metric is not None
                else self.warped_skeleton.extra["z_profiles"][measure]["radius_metric"],
            )
            for measure in ["length", "volume"]
        }
        xy_profiles = {
            measure: get_xy_profile(
                skel_renormed,
                extents=xy_profile_extents
                if xy_profile_extents is not None
                else self.warped_skeleton.extra["xy_profiles"][measure]["extents"],
                bin_size=xy_profile_bin_size
                if xy_profile_bin_size is not None
                else self.warped_skeleton.extra["xy_profiles"][measure]["bin_size"],
                smooth=xy_profile_smooth
                if xy_profile_smooth is not None
                else self.warped_skeleton.extra["xy_profiles"][measure]["smooth"],
                measure=measure,
                include_soma=xy_profile_include_soma
                if xy_profile_include_soma is not None
                else self.warped_skeleton.extra["xy_profiles"][measure]["include_soma"],
                voxel_size=xy_profile_voxel_size
                if xy_profile_voxel_size is not None
                else self.warped_skeleton.extra["xy_profiles"][measure]["voxel_size"],
                radius_metric=radius_metric
                if radius_metric is not None
                else self.warped_skeleton.extra["xy_profiles"][measure][
                    "radius_metric"
                ],
            )
            for measure in ["length", "volume"]
        }

        skel_renormed.extra = {
            "prenormed_nodes": self.warped_skeleton.extra[
                "prenormed_nodes"
            ],  # keep the pre-normed warped nodes for future use
            "med_z_on": float(self.warped_skeleton.extra["med_z_on"]),
            "med_z_off": float(self.warped_skeleton.extra["med_z_off"]),
            "voxel_resolution": voxel_resolution.copy(),
            "skeleton_nodes_scale": scale,
            "z_profiles": z_profiles,
            "xy_profiles": xy_profiles,
        }
        if "coverage" in extra:
            coverage = dict(extra["coverage"])
            if "depth_rel" in coverage:
                coverage["depth"] = _covered_depth(
                    coverage["depth_rel"],
                    on_sac_pos,
                    off_sac_pos,
                    scale,
                    voxel_resolution[2],
                )
            skel_renormed.extra["coverage"] = coverage
        skel_renormed.meta.update(
            {
                "pywarper_version": _PYWARPER_VERSION,
                "renormed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
        )

        return skel_renormed
