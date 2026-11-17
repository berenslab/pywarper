"""Reference layers from skeletons: the SAC bands, extra depth layers, the global map."""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from importlib import metadata as _metadata
from pathlib import Path
from typing import Iterable, Iterator, Mapping

import numpy as np
import skeliner as sk
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import binary_fill_holes
from scipy.spatial import KDTree

from .surface import build_mapping, fit_sac_surface, map_distortion
from .warpers import _DepthModel, relative_depth

__all__ = [
    "BOUNDS",
    "VOLUME_Z",
    "Bands",
    "Cells",
    "Layer",
    "build_map",
    "check_cells",
    "cross_validate",
    "fit_bands",
    "load_cells",
    "load_layers",
    "load_map",
    "report",
    "save_layers",
    "save_map",
    "soma_layer",
    "stratum_layer",
]

#: The EW2 volume in x and y, um.
BOUNDS = (0.0, 1311.0, 0.0, 1311.0)
#: The EW2 volume's bottom and top face, um.
VOLUME_Z = (0.042, 82.551)
#: Flattened depth of the ON and of the OFF band.
ON_DEPTH, OFF_DEPTH = 0.0, 12.0

_NM = {"nm", "nanometer", "nanometre"}
_UM = {"um", "µm", "micron", "micrometer", "micrometre"}


# ------------------------------------------------------------------ cells ---


@dataclass(frozen=True)
class Cells:
    """One cell type's skeleton nodes in um, with `cell` indexing `ids` per node.
    `soma` (centre) and `soma_extent` (z half-height) are NaN for a cell without one."""

    ids: np.ndarray
    nodes: np.ndarray
    cell: np.ndarray
    soma: np.ndarray
    soma_extent: np.ndarray
    summary: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.ids)

    @classmethod
    def from_skeletons(cls, skeletons: Mapping, *, unit: str | None = None) -> "Cells":
        """Cells from skeliner skeletons keyed by cell ID; `unit` is for
        skeletons that carry none."""
        items = ((key, skel, f"cell {key}") for key, skel in skeletons.items())
        return _gather(items, unit, {})

    @classmethod
    def merge(cls, *cells: "Cells") -> "Cells":
        """Several sets of cells as one, e.g. types that share a stratum."""
        ids = np.concatenate([c.ids for c in cells])
        if len(set(ids.tolist())) != len(ids):
            raise ValueError("a cell is in more than one of the merged sets.")
        offsets = np.cumsum([0] + [len(c) for c in cells[:-1]])
        return cls(
            ids=ids,
            nodes=np.concatenate([c.nodes for c in cells]),
            cell=np.concatenate([c.cell + o for c, o in zip(cells, offsets)]),
            soma=np.concatenate([c.soma for c in cells]),
            soma_extent=np.concatenate([c.soma_extent for c in cells]),
            summary={"merged": [c.summary for c in cells], "cells": len(ids)},
        )

    def subset(self, keep: np.ndarray) -> "Cells":
        """The cells where the boolean mask `keep` (one entry per cell) is True."""
        keep = np.asarray(keep, dtype=bool)
        renumber = np.cumsum(keep) - 1
        nodes_kept = keep[self.cell]
        return Cells(
            ids=self.ids[keep],
            nodes=self.nodes[nodes_kept],
            cell=renumber[self.cell[nodes_kept]],
            soma=self.soma[keep],
            soma_extent=self.soma_extent[keep],
            summary=dict(self.summary, cells=int(keep.sum())),
        )


def load_cells(
    root: str | Path,
    pattern: str,
    *,
    ids: Iterable | None = None,
    unit: str | None = None,
) -> Cells:
    """Read a cell type's skeletons: `pattern` locates one cell's file under `root`,
    ``{id}`` standing for the cell ID (``"{id}.npz"``, ``"{id}/skeleton.npz"``).
    `ids` restricts the read to those cells; `unit` is for files that carry none."""
    root = Path(root)
    paths = _find(root, pattern)
    if not paths:
        raise FileNotFoundError(f"no file matches {pattern!r} under {root}.")
    not_listed = 0
    if ids is not None:
        wanted = sorted({str(i) for i in ids})
        missing = [i for i in wanted if i not in paths]
        if missing:
            shown = ", ".join(missing[:5]) + (", ..." if len(missing) > 5 else "")
            raise FileNotFoundError(
                f"{len(missing)} listed cells have no file under {root}: {shown}"
            )
        not_listed = len(paths) - len(wanted)
        paths = {i: paths[i] for i in wanted}

    def read() -> Iterator[tuple]:
        for key in sorted(paths):
            try:
                skel = sk.io.load_skeleton(paths[key])
            except Exception as exc:
                raise OSError(
                    f"cannot read {paths[key]}: {type(exc).__name__}: {exc}"
                ) from exc
            yield key, skel, str(paths[key])

    source = {"root": str(root), "pattern": pattern, "not_listed": not_listed}
    return _gather(read(), unit, source)


def _find(root: Path, pattern: str) -> dict[str, Path]:
    """Cell ID -> file for every file under `root` that matches `pattern`."""
    if pattern.count("{id}") != 1:
        raise ValueError("pattern must contain {id} exactly once.")
    expression = re.escape(pattern)
    for token, sub in (
        (re.escape("{id}"), "(?P<id>[^/]+)"),
        (re.escape("**/"), "(?:.*/)?"),
        (re.escape("*"), "[^/]*"),
        (re.escape("?"), "[^/]"),
    ):
        expression = expression.replace(token, sub)
    matcher = re.compile(f"^{expression}$")
    found: dict[str, Path] = {}
    for path in sorted(root.glob(pattern.replace("{id}", "*"))):
        match = matcher.match(path.relative_to(root).as_posix())
        if match is None:
            continue
        key = match.group("id")
        if key in found:
            raise ValueError(f"cell {key} has two files: {found[key]} and {path}.")
        found[key] = path
    return found


def _to_um(unit, where: str) -> float | None:
    """Factor from `unit` to um, None when there is no unit."""
    if unit is None:
        return None
    name = str(unit).strip().lower().replace("μ", "µ")
    if name in _NM:
        return 1e-3
    if name in _UM:
        return 1.0
    raise ValueError(f"{where}: unit {unit!r} is neither nm nor um.")


def _read_cell(key: str, skel, unit: str | None, where: str) -> tuple:
    """Nodes in um, soma centre and z half-extent of one skeleton, checked."""
    meta = skel.meta or {}
    if "warped_at" in meta:
        raise ValueError(
            f"{where} is a flattened skeleton (warped {meta['warped_at']}); "
            "layers are fitted from skeletons in the volume's own frame."
        )
    own, given = _to_um(meta.get("unit"), where), _to_um(unit, "unit")
    if own is None and given is None:
        raise ValueError(f"{where} carries no unit; pass unit='nm' or unit='um'.")
    if own is not None and given is not None and own != given:
        raise ValueError(f"{where} is in {meta['unit']}, not in {unit}.")
    scale = own if own is not None else given
    if meta.get("id") is not None and str(meta["id"]) != key:
        raise ValueError(f"{where} is filed as cell {key} but names cell {meta['id']}.")

    xyz = np.asarray(skel.nodes, dtype=float) * scale
    centre = np.median(xyz, axis=0)
    low = np.array([BOUNDS[0], BOUNDS[2], VOLUME_Z[0]]) - 5.0
    high = np.array([BOUNDS[1], BOUNDS[3], VOLUME_Z[1]]) + 5.0
    if not ((centre >= low) & (centre <= high)).all():
        raise ValueError(
            f"{where} lies outside the EW2 volume (median node at "
            f"{np.round(centre, 1).tolist()} um): a wrong unit, or another dataset."
        )

    soma, extent = np.full(3, np.nan), np.nan
    if (np.asarray(skel.ntype) == 1).any():
        soma = np.asarray(skel.soma.center, dtype=float) * scale
        along_z = np.asarray(skel.soma.R, dtype=float)[2]
        axes = np.asarray(skel.soma.axes, dtype=float)
        extent = float(np.sqrt(((along_z * axes) ** 2).sum()) * scale)
    return xyz, soma, extent, str(meta.get("unit", unit))


def _gather(items: Iterable[tuple], unit: str | None, source: dict) -> Cells:
    """Cells from (cell ID, skeleton, label) triples, one skeleton in memory at a time."""
    ids, nodes, cell, somata, extents = [], [], [], [], []
    units: dict[str, int] = {}
    for key, skel, where in items:
        key = str(key)
        xyz, soma, extent, label = _read_cell(key, skel, unit, where)
        cell.append(np.full(len(xyz), len(ids), dtype=np.int64))
        ids.append(key)
        nodes.append(xyz)
        somata.append(soma)
        extents.append(extent)
        units[label] = units.get(label, 0) + 1
    if not ids:
        raise ValueError("no skeletons were given.")
    if len(set(ids)) != len(ids):
        raise ValueError("a cell ID occurs twice.")
    extents = np.asarray(extents, dtype=float)
    summary = dict(
        source,
        cells=len(ids),
        without_soma=int(np.isnan(extents).sum()),
        units=units,
    )
    return Cells(
        ids=np.asarray(ids),
        nodes=np.concatenate(nodes),
        cell=np.concatenate(cell),
        soma=np.asarray(somata, dtype=float).reshape(-1, 3),
        soma_extent=extents,
        summary=summary,
    )


# ----------------------------------------------------------------- layers ---


@dataclass(frozen=True)
class Layer:
    """A reference layer: its height field (um) on the map's unit lattice, the
    flattened depth it lands at (ON band 0, OFF band 12), and where it was observed."""

    name: str
    height: np.ndarray
    depth: float
    observed: np.ndarray
    summary: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Bands:
    """The two SAC bands, the map they span, and where the tissue is."""

    on: Layer
    off: Layer
    tissue: np.ndarray
    bounds: tuple
    volume_z: tuple
    spacing: float
    summary: dict = field(default_factory=dict)


def _lattice(bounds: tuple, spacing: float) -> tuple[np.ndarray, np.ndarray]:
    """Column centres on multiples of the spacing, covering the map."""
    if bounds[0] != 0 or bounds[2] != 0:
        raise ValueError("bounds must start at 0 in x and y.")
    xg = np.arange(0, np.floor(bounds[1] / spacing) + 1) * spacing
    yg = np.arange(0, np.floor(bounds[3] / spacing) + 1) * spacing
    return xg, yg


def _column_stats(
    xyz: np.ndarray, xg: np.ndarray, yg: np.ndarray, spacing: float
) -> dict[str, np.ndarray]:
    """Median, 10th and 90th percentile of z and the node count per column: the
    nodes within `spacing` of each point of the lattice `xg` x `yg`, which lies
    on multiples of `spacing`."""
    shape = (len(xg), len(yg))
    n_columns = shape[0] * shape[1]
    quantiles = np.full((n_columns, 3), np.nan)
    count = np.zeros(n_columns, dtype=int)
    if len(xyz):
        # nodes by rising z, so that a sort of (column, node) pairs sorts each
        # column's members by z
        order = np.argsort(xyz[:, 2])
        x, y, z = xyz[order, 0], xyz[order, 1], xyz[order, 2]
        # the lattice cell of each node: i * spacing <= x < (i + 1) * spacing
        cell = []
        for coordinate in (x, y):
            index = np.floor(coordinate / spacing).astype(np.int64)
            index -= coordinate < index * spacing
            index += coordinate >= (index + 1) * spacing
            cell.append(index)
        on_line = [x == cell[0] * spacing, y == cell[1] * spacing]
        # a node reaches the corners of its cell and, from a lattice line, the
        # lattice points one step behind that line
        bits = int(len(z)).bit_length()
        pairs = []
        for a in (-1, 0, 1):
            for b in (-1, 0, 1):
                if a < 0 and b < 0:
                    continue
                who = None
                if a < 0 or b < 0:
                    who = np.flatnonzero(on_line[0] if a < 0 else on_line[1])
                    if not who.size:
                        continue
                i = (cell[0] if who is None else cell[0][who]) + a
                j = (cell[1] if who is None else cell[1][who]) + b
                dx = (x if who is None else x[who]) - i * spacing
                dy = (y if who is None else y[who]) - j * spacing
                member = (
                    (i >= 0)
                    & (i < shape[0])
                    & (j >= 0)
                    & (j < shape[1])
                    & (dx * dx + dy * dy <= spacing * spacing)
                )
                node = np.flatnonzero(member) if who is None else who[member]
                pairs.append(((i[member] * shape[1] + j[member]) << bits) | node)
        pairs = np.sort(np.concatenate(pairs))
        column = pairs >> bits
        values = z[pairs & ((1 << bits) - 1)]
        count = np.bincount(column, minlength=n_columns)
        held = np.flatnonzero(count)
        n, first = count[held], (np.cumsum(count) - count)[held]
        # the percentiles as numpy interpolates them between order statistics
        for k, q in enumerate((0.5, 0.1, 0.9)):
            position = (n - 1) * q
            below = np.floor(position).astype(np.int64)
            above = np.minimum(below + 1, n - 1)
            weight = position - below
            low, high = values[first + below], values[first + above]
            step = high - low
            quantiles[held, k] = np.where(
                weight >= 0.5, high - step * (1 - weight), low + step * weight
            )
    return {
        "median": quantiles[:, 0].reshape(shape),
        "low": quantiles[:, 1].reshape(shape),
        "high": quantiles[:, 2].reshape(shape),
        "count": count.reshape(shape),
    }


def _observed(
    stats: dict, volume_z: tuple, margin: float, min_fill: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Columns that hold the layer, those a face cuts, those too thin, and the
    typical node count of a column."""
    count = stats["count"]
    typical = float(np.median(count[count > 0])) if (count > 0).any() else 0.0
    cut = np.zeros(count.shape, dtype=bool)
    with np.errstate(invalid="ignore"):
        if volume_z[1] is not None:
            cut |= stats["high"] > volume_z[1] - margin
        if volume_z[0] is not None:
            cut |= stats["low"] < volume_z[0] + margin
    thin = (count > 0) & (count < min_fill * typical) & ~cut
    return (count > 0) & ~cut & ~thin, cut, thin, typical


def _neighbour_median(
    query_xy: np.ndarray, ref_xy: np.ndarray, ref_val: np.ndarray, k: int
) -> np.ndarray:
    """Median of `ref_val` over the k reference points nearest each query."""
    _, nearest = KDTree(ref_xy).query(query_xy, k=min(k, len(ref_xy)), workers=-1)
    if nearest.ndim == 1:
        nearest = nearest[:, None]
    return np.median(ref_val[nearest], axis=1)


def _fit(x, y, v, bounds: tuple, smoothness: float) -> np.ndarray:
    """Smooth field through scattered values, on the map's unit lattice."""
    return fit_sac_surface(
        x, y, v, xmax=bounds[1], ymax=bounds[3], smoothness=smoothness
    ).zmesh


def _bounded_field(
    sample_xy: np.ndarray,
    sample_val: np.ndarray,
    xx: np.ndarray,
    yy: np.ndarray,
    covered: np.ndarray,
    k: int,
    smoothness: float,
    bounds: tuple,
) -> np.ndarray:
    """Smooth field through the samples that keeps to their range away from them:
    lattice columns outside `covered` take the median of the k nearest samples."""
    far = ~covered
    pinned = _neighbour_median(
        np.column_stack([xx[far], yy[far]]), sample_xy, sample_val, k
    )
    return _fit(
        np.r_[sample_xy[:, 0], xx[far]],
        np.r_[sample_xy[:, 1], yy[far]],
        np.r_[sample_val, pinned],
        bounds,
        smoothness,
    )


def _to_unit(
    mask: np.ndarray, xg: np.ndarray, yg: np.ndarray, shape: tuple
) -> np.ndarray:
    """A lattice mask on the unit lattice, each pixel taking its nearest column."""
    spacing = xg[1] - xg[0]
    ix = np.clip(np.rint(np.arange(shape[0]) / spacing).astype(int), 0, len(xg) - 1)
    iy = np.clip(np.rint(np.arange(shape[1]) / spacing).astype(int), 0, len(yg) - 1)
    return mask[np.ix_(ix, iy)]


def _read(surface: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Bilinear read of a unit-lattice field, clipped to the map."""
    gx, gy = np.arange(surface.shape[0]), np.arange(surface.shape[1])
    at = np.column_stack([np.clip(x, 0, gx[-1]), np.clip(y, 0, gy[-1])])
    return RegularGridInterpolator((gx, gy), surface)(at)


def _relative(xyz: np.ndarray, bands: Bands) -> tuple[np.ndarray, np.ndarray]:
    """Two-band relative depth of points (ON band 0, OFF band 1) and the local gap."""
    z_on = _read(bands.on.height, xyz[:, 0], xyz[:, 1])
    gap = _read(bands.off.height, xyz[:, 0], xyz[:, 1]) - z_on
    return (xyz[:, 2] - z_on) / gap, gap


def _flat(relative: np.ndarray | float) -> np.ndarray | float:
    """Relative depth in flattened units."""
    return ON_DEPTH + relative * (OFF_DEPTH - ON_DEPTH)


def fit_bands(
    on: Cells,
    off: Cells,
    *,
    bounds: tuple = BOUNDS,
    volume_z: tuple = VOLUME_Z,
    spacing: float = 5.0,
    smoothness: float = 15.0,
    min_fill: float = 0.1,
    clip_margin: float = 1.5,
    neighbours: int = 64,
) -> Bands:
    """The two SAC bands over the map: ON fitted, OFF = ON plus a gap field.
    A column holds a band with `min_fill` of its typical node count and deciles
    `clip_margin` clear of the faces; missing gaps take the `neighbours` nearest."""
    if set(on.ids.tolist()) & set(off.ids.tolist()):
        raise ValueError("a cell is among both the ON and the OFF cells.")
    xg, yg = _lattice(bounds, spacing)
    xx, yy = np.meshgrid(xg, yg, indexing="ij")
    stats, observed, cut, thin, typical = {}, {}, {}, {}, {}
    for name, cells in (("on", on), ("off", off)):
        stats[name] = _column_stats(cells.nodes, xg, yg, spacing)
        observed[name], cut[name], thin[name], typical[name] = _observed(
            stats[name], volume_z, clip_margin, min_fill
        )
    both = observed["on"] & observed["off"]
    if not both.any():
        raise ValueError("the two bands share no observed column.")
    gap_columns = (stats["off"]["median"] - stats["on"]["median"])[both]
    if not ((gap_columns > 0).all() or (gap_columns < 0).all()):
        raise ValueError("the bands cross where both are observed.")

    gap_lattice = np.empty(xx.shape)
    gap_lattice[both] = gap_columns
    gap_lattice[~both] = _neighbour_median(
        np.column_stack([xx[~both], yy[~both]]),
        np.column_stack([xx[both], yy[both]]),
        gap_columns,
        neighbours,
    )
    gap = _fit(xx.ravel(), yy.ravel(), gap_lattice.ravel(), bounds, smoothness)

    either = observed["on"] | observed["off"]
    on_columns = np.where(
        observed["on"], stats["on"]["median"], stats["off"]["median"] - gap_lattice
    )
    on_height = _fit(xx[either], yy[either], on_columns[either], bounds, smoothness)

    shape = on_height.shape
    settings = {
        "spacing": spacing,
        "smoothness": smoothness,
        "min_fill": min_fill,
        "clip_margin": clip_margin,
        "neighbours": neighbours,
    }
    layers = {}
    for name, height, depth, cells in (
        ("on", on_height, ON_DEPTH, on),
        ("off", on_height + gap, OFF_DEPTH, off),
    ):
        layers[name] = Layer(
            name=name,
            height=height,
            depth=depth,
            observed=_to_unit(
                binary_fill_holes(observed[name]) & ~cut[name], xg, yg, shape
            ),
            summary={
                "kind": "band",
                "cells": len(cells),
                "nodes": int(len(cells.nodes)),
                "columns_observed": int(observed[name].sum()),
                "columns_cut": int(cut[name].sum()),
                "columns_thin": int(thin[name].sum()),
                "typical_nodes": typical[name],
                "source": cells.summary,
                "cell_ids": cells.ids.tolist(),
            },
        )
    summary = {
        "columns": {
            "both": int(both.sum()),
            "on_only": int((observed["on"] & ~observed["off"]).sum()),
            "off_only": int((observed["off"] & ~observed["on"]).sum()),
            "neither": int((~either).sum()),
        },
        "gap_um": {
            "median": float(np.median(gap_columns)),
            "p5": float(np.percentile(gap_columns, 5)),
            "p95": float(np.percentile(gap_columns, 95)),
            "min_over_map": float(gap.min()),
            "max_over_map": float(gap.max()),
        },
        "settings": settings,
    }
    return Bands(
        on=layers["on"],
        off=layers["off"],
        tissue=_to_unit(binary_fill_holes(either), xg, yg, shape),
        bounds=tuple(float(b) for b in bounds),
        volume_z=tuple(volume_z),
        spacing=float(spacing),
        summary=summary,
    )


def _field_summary(field_: np.ndarray, observed: np.ndarray, bands: Bands) -> dict:
    """Range of a relative-depth field inside the tissue, in flattened depth."""
    inside = field_[bands.tissue]
    return {
        "depth_min_in_tissue": float(_flat(inside.min())),
        "depth_max_in_tissue": float(_flat(inside.max())),
        "observed_share_of_tissue": float(
            (observed & bands.tissue).sum() / bands.tissue.sum()
        ),
    }


def stratum_layer(
    name: str,
    cells: Cells,
    bands: Bands,
    depth: float,
    half_width: float,
    *,
    passes: int = 2,
    smoothness: float = 15.0,
    min_fill: float = 0.1,
    clip_margin: float = 1.5,
    neighbours: int = 64,
) -> Layer:
    """A layer through the stratum `cells` form near flattened `depth` (ON 0, OFF 12).
    Pass one takes nodes within twice `half_width` of `depth`, later passes those
    within `half_width` of the layer so far; it lands at its columns' median depth."""
    if passes < 1:
        raise ValueError("passes must be at least 1.")
    scale = OFF_DEPTH - ON_DEPTH
    xg, yg = _lattice(bands.bounds, bands.spacing)
    xx, yy = np.meshgrid(xg, yg, indexing="ij")
    column_on = _read(bands.on.height, xx.ravel(), yy.ravel()).reshape(xx.shape)
    column_gap = (
        _read(bands.off.height, xx.ravel(), yy.ravel()).reshape(xx.shape) - column_on
    )
    relative, _ = _relative(cells.nodes, bands)
    centre = np.full(len(relative), (depth - ON_DEPTH) / scale)
    width = 2.0 * half_width / scale
    for _ in range(passes):
        inside = np.abs(relative - centre) <= width
        stats = _column_stats(cells.nodes[inside], xg, yg, bands.spacing)
        observed, cut, thin, typical = _observed(
            stats, bands.volume_z, clip_margin, min_fill
        )
        if not observed.any():
            raise ValueError(
                f"no column holds layer {name!r}: these cells have no stratum "
                f"near depth {depth}."
            )
        column_depth = (stats["median"] - column_on) / column_gap
        field_ = _bounded_field(
            np.column_stack([xx[observed], yy[observed]]),
            column_depth[observed],
            xx,
            yy,
            observed,
            neighbours,
            smoothness,
            bands.bounds,
        )
        centre = _read(field_, cells.nodes[:, 0], cells.nodes[:, 1])
        width = half_width / scale

    shape = bands.on.height.shape
    seen = _to_unit(binary_fill_holes(observed) & ~cut, xg, yg, shape)
    summary = {
        "kind": "stratum",
        "cells": len(cells),
        "nodes_in_window": int(inside.sum()),
        "columns_observed": int(observed.sum()),
        "columns_cut": int(cut.sum()),
        "columns_thin": int(thin.sum()),
        "typical_nodes": typical,
        "settings": {
            "depth": depth,
            "half_width": half_width,
            "passes": passes,
            "smoothness": smoothness,
            "min_fill": min_fill,
            "clip_margin": clip_margin,
            "neighbours": neighbours,
        },
        "source": cells.summary,
        "cell_ids": cells.ids.tolist(),
        **_field_summary(field_, seen, bands),
    }
    return Layer(
        name=name,
        height=bands.on.height + field_ * (bands.off.height - bands.on.height),
        depth=float(_flat(np.median(column_depth[observed]))),
        observed=seen,
        summary=summary,
    )


def _on_level(relative: np.ndarray) -> np.ndarray:
    """Samples within five robust standard deviations of the population's level;
    one far off it is not on the layer."""
    level = np.median(relative)
    spread = 1.4826 * np.median(np.abs(relative - level))
    if spread == 0:
        return np.ones(len(relative), dtype=bool)
    return np.abs(relative - level) <= 5.0 * spread


def _room(level: float, bands: Bands, x: np.ndarray, y: np.ndarray) -> tuple:
    """Distance from a layer at two-band depth `level` down to the bottom face and
    up to the top face, inf for an unknown face."""
    z_on = _read(bands.on.height, x, y)
    z = z_on + level * (_read(bands.off.height, x, y) - z_on)
    low, high = bands.volume_z
    below = z - low if low is not None else np.full(len(z), np.inf)
    above = high - z if high is not None else np.full(len(z), np.inf)
    return below, above


def soma_layer(
    name: str,
    cells: Cells,
    bands: Bands,
    *,
    smoothness: float = 200.0,
    reach: float = 100.0,
    neighbours: int = 16,
    clip_margin: float = 1.5,
) -> Layer:
    """A layer through the somata of `cells`, one sample per cell.
    Only places where the layer's spread plus a soma's half-height clears both faces
    by `clip_margin` count; the field is pinned beyond `reach` um from those samples."""
    has = np.isfinite(cells.soma).all(axis=1) & np.isfinite(cells.soma_extent)
    if has.sum() < 3:
        raise ValueError(f"layer {name!r} needs at least three cells with a soma.")
    xyz, extent = cells.soma[has], cells.soma_extent[has]
    relative, gap = _relative(xyz, bands)

    use = _on_level(relative)

    def needs(keep: np.ndarray, level: float) -> tuple[float, float]:
        about = (relative[keep] - level) * gap[keep]
        own = float(np.median(extent[keep])) + clip_margin
        return float(np.percentile(about, 90)) + own, own - float(
            np.percentile(about, 10)
        )

    def enough(keep: np.ndarray) -> np.ndarray:
        if keep.sum() < 3:
            raise ValueError(
                f"layer {name!r}: fewer than three somata lie where the whole "
                "layer fits in the volume."
            )
        return keep

    level = float(np.median(relative[use]))
    below, above = _room(level, bands, xyz[:, 0], xyz[:, 1])
    keep = enough(
        use & (below - extent >= clip_margin) & (above - extent >= clip_margin)
    )
    for _ in range(3):
        up, down = needs(keep, level)
        keep = enough(use & (above >= up) & (below >= down))
        level = float(np.median(relative[keep]))
        below, above = _room(level, bands, xyz[:, 0], xyz[:, 1])

    xg, yg = _lattice(bands.bounds, bands.spacing)
    xx, yy = np.meshgrid(xg, yg, indexing="ij")
    column_below, column_above = _room(level, bands, xx.ravel(), yy.ravel())
    distance, _ = KDTree(xyz[keep, :2]).query(
        np.column_stack([xx.ravel(), yy.ravel()]), workers=-1
    )
    covered = (
        (column_above >= up) & (column_below >= down) & (distance <= reach)
    ).reshape(xx.shape)
    field_ = _bounded_field(
        xyz[keep, :2],
        relative[keep],
        xx,
        yy,
        covered,
        neighbours,
        smoothness,
        bands.bounds,
    )
    seen = _to_unit(covered, xg, yg, bands.on.height.shape)
    summary = {
        "kind": "soma",
        "cells": len(cells),
        "somata": int(has.sum()),
        "samples_used": int(keep.sum()),
        "samples_off_level": int((~use).sum()),
        "samples_without_room": int((use & ~keep).sum()),
        "room_needed_above_um": up,
        "room_needed_below_um": down,
        "settings": {
            "smoothness": smoothness,
            "reach": reach,
            "neighbours": neighbours,
            "clip_margin": clip_margin,
        },
        "source": cells.summary,
        "cell_ids": cells.ids[has][keep].tolist(),
        **_field_summary(field_, seen, bands),
    }
    return Layer(
        name=name,
        height=bands.on.height + field_ * (bands.off.height - bands.on.height),
        depth=float(_flat(level)),
        observed=seen,
        summary=summary,
    )


# ----------------------------------------------------------------- checks ---


def check_cells(
    cells: Cells, layer: Layer, *, window: float = 4.0, min_share: float = 0.1
) -> dict[str, np.ndarray]:
    """Per cell: the median distance (um) of its nodes near the layer from it, the
    share of its nodes within `window` um, and `off_layer`, True for a cell with
    less than `min_share` there or a median more than half a window away."""
    away = cells.nodes[:, 2] - _read(layer.height, cells.nodes[:, 0], cells.nodes[:, 1])
    order = np.argsort(cells.cell, kind="stable")
    ends = np.cumsum(np.bincount(cells.cell, minlength=len(cells)))
    offset, share = np.full(len(cells), np.nan), np.zeros(len(cells))
    for i, (start, stop) in enumerate(zip(np.r_[0, ends[:-1]], ends)):
        own = away[order[start:stop]]
        near = own[np.abs(own) <= 3.0 * window]
        if len(near):
            offset[i] = np.median(near)
        if len(own):
            share[i] = np.mean(np.abs(own) <= window)
    off_layer = (share < min_share) | ~(np.abs(offset) <= window / 2.0)
    return {"ids": cells.ids, "offset": offset, "share": share, "off_layer": off_layer}


def _depth_mapping(bands: Bands, layers: Iterable[Layer]) -> dict:
    """What `relative_depth` needs of a map, without building one."""
    mapping = {"on_sac_surface": bands.on.height, "off_sac_surface": bands.off.height}
    layers = list(layers)
    if layers:
        scale = OFF_DEPTH - ON_DEPTH
        mapping["extra_surfaces"] = np.stack([layer.height for layer in layers])
        mapping["extra_surface_depths"] = np.array(
            [(layer.depth - ON_DEPTH) / scale for layer in layers]
        )
    return mapping


def _drift(xyz: np.ndarray, depth: np.ndarray, band_z: np.ndarray) -> dict:
    """Spread of flattened depth and its slopes against raw band z, x and y."""
    if len(depth) < 4:
        return {"n": int(len(depth))}
    centred = (xyz[:, :2] - xyz[:, :2].mean(axis=0)) / 100.0
    plane = np.linalg.lstsq(
        np.column_stack([np.ones(len(depth)), centred]), depth, rcond=None
    )[0]
    return {
        "n": int(len(depth)),
        "depth_median": float(np.median(depth)),
        "depth_sd": float(depth.std()),
        "per_um_raw_z": float(np.polyfit(band_z, depth, 1)[0]),
        "per_100um_x": float(plane[1]),
        "per_100um_y": float(plane[2]),
    }


def cross_validate(
    kind: str, cells: Cells, bands: Bands, *, seed: int = 0, **settings
) -> dict:
    """Fit a layer ("soma" or "stratum", with that function's `settings`) on a random
    half of the cells and score the other half where the layer is observed: depth
    spread and slopes with the bands alone ("bands") and with the layer ("layer")."""
    half = np.random.default_rng(seed).random(len(cells)) < 0.5
    fitted, held = cells.subset(half), cells.subset(~half)
    if kind == "soma":
        layer = soma_layer("held_out", fitted, bands, **settings)
        xyz = held.soma[np.isfinite(held.soma).all(axis=1)]
        xyz = xyz[_on_level(_relative(xyz, bands)[0])]
    elif kind == "stratum":
        layer = stratum_layer("held_out", fitted, bands, **settings)
        relative, gap = _relative(held.nodes, bands)
        at_layer = (
            _read(layer.height, held.nodes[:, 0], held.nodes[:, 1])
            - _read(bands.on.height, held.nodes[:, 0], held.nodes[:, 1])
        ) / gap
        width = settings["half_width"] / (OFF_DEPTH - ON_DEPTH)
        xg, yg = _lattice(bands.bounds, bands.spacing)
        xx, yy = np.meshgrid(xg, yg, indexing="ij")
        stats = _column_stats(
            held.nodes[np.abs(relative - at_layer) <= width], xg, yg, bands.spacing
        )
        observed = _observed(
            stats,
            bands.volume_z,
            settings.get("clip_margin", 1.5),
            settings.get("min_fill", 0.1),
        )[0]
        xyz = np.column_stack([xx[observed], yy[observed], stats["median"][observed]])
    else:
        raise ValueError("kind must be 'soma' or 'stratum'.")

    shape = layer.observed.shape
    ix = np.clip(np.rint(xyz[:, 0]).astype(int), 0, shape[0] - 1)
    iy = np.clip(np.rint(xyz[:, 1]).astype(int), 0, shape[1] - 1)
    xyz = xyz[layer.observed[ix, iy]]
    two_band = _relative(xyz, bands)[0]
    layered = relative_depth(xyz, _depth_mapping(bands, [layer]))
    nearer = bands.on if np.median(two_band) < 0.5 else bands.off
    band_z = _read(nearer.height, xyz[:, 0], xyz[:, 1])
    return {
        "level": layer.depth,
        "observed_share_of_tissue": layer.summary["observed_share_of_tissue"],
        "bands": _drift(xyz, _flat(two_band), band_z),
        "layer": _drift(xyz, _flat(layered), band_z),
    }


# ---------------------------------------------------------------- the map ---


def _version(package: str) -> str:
    try:
        return _metadata.version(package)
    except _metadata.PackageNotFoundError:
        return "unknown"


def build_map(
    bands: Bands,
    layers: Iterable[Layer] = (),
    *,
    conformal_jump: int = 2,
    n_anchors: int = 16,
    verbose: bool = False,
) -> dict:
    """The global map of the bands with `layers` as extra depth surfaces: the output
    of `pywarper.surface.build_mapping` plus ``tissue`` and a ``provenance`` record."""
    layers = list(layers)
    names = [layer.name for layer in layers]
    if len(set(names)) != len(names):
        raise ValueError("two layers share a name.")
    scale = OFF_DEPTH - ON_DEPTH
    extra = {
        layer.name: (layer.height, (layer.depth - ON_DEPTH) / scale, layer.observed)
        for layer in layers
    }
    mapping = build_mapping(
        bands.on.height,
        bands.off.height,
        bands.bounds,
        conformal_jump=conformal_jump,
        n_anchors=n_anchors,
        volume_z=bands.volume_z,
        on_support=bands.on.observed,
        off_support=bands.off.observed,
        extra_surfaces=extra or None,
        verbose=verbose,
    )
    mapping["tissue"] = bands.tissue

    out_of_order = {}
    if layers:
        ix, iy = np.nonzero(bands.tissue[::4, ::4])
        model = _DepthModel(mapping)
        _, _, valid, _ = model.stack(np.column_stack([ix * 4.0, iy * 4.0]))
        ordered = [names[k - 2] if k >= 2 else None for k in model.order]
        for column, layer_name in enumerate(ordered):
            if layer_name is not None:
                out_of_order[layer_name] = float(1.0 - valid[:, column].mean())
    mapping["provenance"] = {
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "pywarper": _version("pywarper"),
        "skeliner": _version("skeliner"),
        "bounds_um": list(bands.bounds),
        "volume_z_um": list(bands.volume_z),
        "bands": bands.summary,
        "layers": {
            layer.name: dict(layer.summary, depth=layer.depth)
            for layer in (bands.on, bands.off, *layers)
        },
        "out_of_order_share_of_tissue": out_of_order,
    }
    return mapping


def _plain(value):
    """numpy scalars and arrays as JSON."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"{type(value).__name__} is not JSON serialisable")


def save_map(mapping: dict, path: str | Path) -> None:
    """Write a map as one `.npz` that is read without pickle; dict entries
    (``meta``, ``provenance``) are stored as JSON text."""
    arrays = {}
    for key, value in mapping.items():
        if isinstance(value, dict):
            value = json.dumps(value, default=_plain)
        value = np.asarray(value)
        if value.dtype == object:
            raise TypeError(f"map entry {key!r} cannot be stored without pickle.")
        arrays[key] = value
    np.savez(path, **arrays)


def load_map(path: str | Path) -> dict:
    """Read a map written by `save_map`."""
    with np.load(path, allow_pickle=False) as stored:
        mapping = {key: stored[key] for key in stored.files}
    for key, value in mapping.items():
        if key in ("meta", "provenance"):
            mapping[key] = json.loads(str(value))
        elif value.ndim == 0 and value.dtype.kind in "iuf":
            mapping[key] = value.item()
    return mapping


def save_layers(bands: Bands, layers: Iterable[Layer], path: str | Path) -> None:
    """Write the bands and the extra layers, without a map, as one `.npz`."""
    every = [bands.on, bands.off, *layers]
    np.savez_compressed(
        path,
        names=np.array([layer.name for layer in every]),
        heights=np.stack([layer.height for layer in every]),
        depths=np.array([layer.depth for layer in every]),
        observed=np.stack([layer.observed for layer in every]),
        summaries=np.array(
            json.dumps([layer.summary for layer in every], default=_plain)
        ),
        tissue=bands.tissue,
        bounds=np.array(bands.bounds, dtype=float),
        volume_z=np.array([np.nan if v is None else v for v in bands.volume_z]),
        spacing=np.array(bands.spacing),
        bands_summary=np.array(json.dumps(bands.summary, default=_plain)),
    )


def load_layers(path: str | Path) -> tuple[Bands, list[Layer]]:
    """Read what `save_layers` wrote: the bands and the extra layers."""
    with np.load(path, allow_pickle=False) as stored:
        summaries = json.loads(str(stored["summaries"]))
        every = [
            Layer(
                name=str(name),
                height=stored["heights"][i],
                depth=float(stored["depths"][i]),
                observed=stored["observed"][i],
                summary=summaries[i],
            )
            for i, name in enumerate(stored["names"])
        ]
        bands = Bands(
            on=every[0],
            off=every[1],
            tissue=stored["tissue"],
            bounds=tuple(stored["bounds"].tolist()),
            volume_z=tuple(
                None if np.isnan(v) else float(v) for v in stored["volume_z"]
            ),
            spacing=float(stored["spacing"]),
            summary=json.loads(str(stored["bands_summary"])),
        )
    return bands, every[2:]


def report(
    bands: Bands, layers: Iterable[Layer] = (), mapping: dict | None = None
) -> str:
    """What the bands, the extra layers and, with `mapping`, the map amount to, as text."""
    tissue = bands.tissue
    gap = bands.summary.get("gap_um", {})
    columns = bands.summary.get("columns", {})
    lines = [
        f"tissue {100 * tissue.mean():.1f}% of the map; columns with both bands "
        f"{columns.get('both', 0):,}, ON only {columns.get('on_only', 0):,}, "
        f"OFF only {columns.get('off_only', 0):,}",
        f"band gap {gap.get('median', np.nan):.1f} um (5-95% {gap.get('p5', np.nan):.1f}-"
        f"{gap.get('p95', np.nan):.1f}), {gap.get('min_over_map', np.nan):.1f}-"
        f"{gap.get('max_over_map', np.nan):.1f} over the whole map",
    ]
    for layer in (bands.on, bands.off, *layers):
        s = layer.summary
        seen = 100 * (layer.observed & tissue).sum() / max(tissue.sum(), 1)
        line = (
            f"{layer.name}: depth {layer.depth:+.2f}, {s.get('cells', 0):,} cells, "
            f"observed over {seen:.1f}% of the tissue"
        )
        if "columns_cut" in s:
            line += (
                f"; {s['columns_cut']:,} columns cut by a face, "
                f"{s['columns_thin']:,} too thin"
            )
        if "samples_used" in s:
            line += (
                f"; {s['samples_used']:,} somata used, "
                f"{s['samples_without_room']:,} without room, "
                f"{s['samples_off_level']:,} off the level"
            )
        if "depth_min_in_tissue" in s:
            line += (
                f"; field {s['depth_min_in_tissue']:+.1f} to "
                f"{s['depth_max_in_tissue']:+.1f} in the tissue"
            )
        lines.append(line)
    if mapping is not None:
        budget = map_distortion(mapping, tissue)
        on, off = budget["on"]["summary"], budget["off"]["summary"]
        lines.append(
            f"map (jump {int(mapping['conformal_jump'])}, {int(mapping['n_anchors'])} "
            f"anchors): length inside the tissue ON {on['length_scale_p1']:.4f}-"
            f"{on['length_scale_p99']:.4f}, OFF {off['length_scale_p1']:.4f}-"
            f"{off['length_scale_p99']:.4f}"
        )
        provenance = mapping.get("provenance")
        if isinstance(provenance, dict):
            for name, share in provenance.get(
                "out_of_order_share_of_tissue", {}
            ).items():
                if share > 0:
                    lines.append(
                        f"{name} is out of order over {100 * share:.2f}% of the "
                        "tissue and is left out there"
                    )
    return "\n".join(lines)
