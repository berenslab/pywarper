from typing import Optional

import numpy as np
from alphashape import alphashape

from .arbor import segment_lengths

# Convex hull

def get_convex_hull(points: np.ndarray) -> np.ndarray:
    """Return the planar convex hull enclosing *points*.

    Parameters
    ----------
    points : (N, 2) ndarray
        x/y coordinates in µm.

    Returns
    -------
    hull : (M, 2) ndarray
        Vertices of the convex hull ordered counter‑clockwise.  If *points*
        contains < 3 entries, the input is returned unchanged.
    """
    if len(points) < 3:
        return points
    hull_poly = alphashape(points, alpha=0)
    return np.vstack(hull_poly.exterior.xy).T


# -- internal polygon utilities -------------------------------------------

def _polygon_area(vertices: np.ndarray) -> float:
    """Signed area of a simple polygon given by *vertices* (M × 2)."""
    x, y = vertices.T
    return 0.5 * (np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _polygon_centroid(vertices: np.ndarray, signed_area: Optional[float] = None) -> np.ndarray:
    """Centroid *x, y* of a simple polygon.

    The arithmetic mean is used when the polygon area is ~ 0.
    """
    if signed_area is None:
        signed_area = _polygon_area(vertices)
    if np.isclose(signed_area, 0.0):
        return vertices.mean(axis=0)

    x, y = vertices.T
    shift_x = np.roll(x, -1)
    shift_y = np.roll(y, -1)
    cross = x * shift_y - shift_x * y
    cx = np.sum((x + shift_x) * cross) / (6.0 * signed_area)
    cy = np.sum((y + shift_y) * cross) / (6.0 * signed_area)
    return np.array([cx, cy], dtype=float)


# -- public polygon wrappers ----------------------------------------------

def get_hull_area(hull: np.ndarray) -> float:
    """Unsigned area (µm²) of a convex *hull*."""
    return abs(_polygon_area(hull))


def get_hull_centroid(hull: np.ndarray) -> np.ndarray:
    """Centroid (x, y in µm) of a convex *hull*."""
    area = _polygon_area(hull)
    return _polygon_centroid(hull, area)

# Center‑of‑mass (COM) metrics

def get_xy_center_of_mass(x: np.ndarray, y: np.ndarray, xy_dist: np.ndarray) -> np.ndarray:
    """Center of mass in the retinal plane.

    Parameters
    ----------
    x, y : (N,) ndarray
        Sample positions along each axis (µm).
    xy_dist : (N, N) ndarray
        2‑D density map over *x* and *y*.

    Returns
    -------
    com_xy : (2,) ndarray
        COM coordinates (µm).
    """
    com_x = xy_dist.sum(axis=1) @ x / xy_dist.sum()
    com_y = xy_dist.sum(axis=0) @ y / xy_dist.sum()
    return np.asarray([com_x, com_y])

def get_z_center_of_mass(z_x: np.ndarray, z_dist: np.ndarray) -> float:
    """Center of mass along *z* (depth, µm)."""
    return float(z_dist @ z_x / z_dist.sum())


def get_asymmetry(soma_xy: np.ndarray, com_xy: np.ndarray) -> float:
    """Planar distance (µm) between soma and dendritic COM."""
    return float(np.hypot(*(soma_xy - com_xy)))


def get_soma_to_stratification_depth(soma_z: float, com_z: float) -> float:
    """Absolute depth difference (µm) between soma and dendritic COM."""
    return float(abs(soma_z - com_z))

# Morphology features

def _build_adjacency(edge_pairs: list[tuple[int, int]], n_nodes: int) -> list[list[int]]:
    """Return an undirected adjacency list from 0‑based *edge_pairs*."""
    adj: list[list[int]] = [[] for _ in range(n_nodes)]
    for u, v in edge_pairs:
        adj[u].append(v)
        adj[v].append(u)
    return adj


def get_branch_point_count(edges: np.ndarray) -> int:
    """Number of dendritic branch points.

    The soma (parent = –1) is excluded.
    """
    parents = edges[edges[:, 1] > 0, 1].astype(int) - 1  # convert to 0‑based
    child_counts = np.bincount(parents)
    return int(np.count_nonzero(child_counts >= 2))

def get_dendritic_length(nodes: np.ndarray, edges: np.ndarray) -> float:
    """Total cable length (µm) of an SWC tree."""
    density, _ = segment_lengths(nodes, edges)
    return float(density.sum())


def get_irreducible_nodes(nodes: np.ndarray, edges: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Indices of *irreducible* nodes (degree ≠ 2 ∪ soma) and their coordinates.

    Returns
    -------
    idx_1b : (K,) int
        1‑based indices.
    xyz : (K, 3) ndarray
        Corresponding coordinates (µm).
    """
    child = edges[:, 0].astype(int) - 1
    parent = edges[:, 1].astype(int) - 1

    # discard root rows (parent < 0)
    edge_pairs = [tuple(sorted((c, p))) for c, p in zip(child, parent) if p >= 0]

    N = nodes.shape[0]
    adj = _build_adjacency(edge_pairs, N)
    degree = np.fromiter((len(n) for n in adj), int, count=N)

    irreducible_mask = degree != 2
    irreducible_mask[edges[parent == -1, 0] - 1] = True  # include soma root

    idx_1b = np.flatnonzero(irreducible_mask) + 1
    return idx_1b, nodes[idx_1b - 1]

# segment statistics

def _segment_stats(nodes: np.ndarray, edges: np.ndarray
                   ) -> tuple[float, np.ndarray, np.ndarray]:
    """Internal helper that walks each irreducible segment once and returns:
        • median segment length (µm)
        • 1‑based indices of irreducible nodes (for convenience)
        • tortuosities per segment
    """
    density, _ = segment_lengths(nodes, edges)

    child = edges[:, 0].astype(int) - 1
    parent = edges[:, 1].astype(int) - 1
    edge_len = {tuple(sorted((c, p))): density[c]
                for c, p in zip(child, parent) if p >= 0}

    N = nodes.shape[0]
    adj = _build_adjacency(list(edge_len.keys()), N)
    degree = np.fromiter((len(n) for n in adj), int, count=N)
    irreducible_mask = degree != 2
    irreducible_mask[edges[parent == -1, 0] - 1] = True   # soma

    visited = set()
    seg_lengths: list[float] = []
    tortuosities: list[float] = []

    for u in np.nonzero(irreducible_mask)[0]:
        for v in adj[u]:
            e = tuple(sorted((u, v)))
            if e in visited:
                continue
            path_len = edge_len[e]
            visited.add(e)
            prev, cur = u, v
            while not irreducible_mask[cur]:
                nxt = adj[cur][0] if adj[cur][0] != prev else adj[cur][1]
                e = tuple(sorted((cur, nxt)))
                path_len += edge_len[e]
                visited.add(e)
                prev, cur = cur, nxt
            eucl = np.linalg.norm(nodes[u] - nodes[cur])
            tortuosities.append(path_len / eucl if eucl > 1e-6 else np.inf)
            seg_lengths.append(path_len)

    seg_lengths_arr = np.asarray(seg_lengths)
    tortuosities_arr = np.asarray(tortuosities)
    med = float(np.median(seg_lengths_arr)) if seg_lengths_arr.size else 0.0
    return med, np.nonzero(irreducible_mask)[0] + 1, tortuosities_arr


def get_median_branch_length(nodes: np.ndarray, edges: np.ndarray) -> float:
    """Median length (µm) of all irreducible segments."""
    med_len, _, _ = _segment_stats(nodes, edges)
    return med_len


def get_average_tortuosity(nodes: np.ndarray, edges: np.ndarray) -> float:
    """Average tortuosity of irreducible segments.

    Tortuosity = path length / straight‑line distance.
    For segments where the Euclidean distance is < 1e‑6 µm, the ratio
    is ignored to avoid numerical blow‑up (returned average is over the
    *remaining* segments).
    """
    _, _, torts = _segment_stats(nodes, edges)
    finite = torts[np.isfinite(torts)]
    return float(np.mean(finite)) if finite.size else 0.0

# radial and angular features

def get_typical_radius(nodes: np.ndarray, edges: np.ndarray, com_xy: np.ndarray) -> float:
    """
    Root-mean-square planar distance (µm) of dendritic cable to COM(xy).
    """
    density, mid = segment_lengths(nodes, edges)
    dx = mid[:, 0] - com_xy[0]
    dy = mid[:, 1] - com_xy[1]
    return float(np.sqrt(np.sum(density * (dx**2 + dy**2)) / density.sum()))

def get_average_angle(nodes: np.ndarray, edges: np.ndarray) -> float:
    """Average positive angle (rad) at irreducible branch points.

    For each irreducible node that has **one upstream** irreducible parent and
    **≥1 downstream** irreducible child(ren), compute the angle between the
    (parent→node) and (node→child) vectors.  The feature is the mean of those
    angles.  Tips (degree 1) contribute nothing; multifurcations contribute
    one angle per child branch.
    """
    # --- precompute fast lookup tables ------------------------------------
    child = edges[:, 0].astype(int) - 1
    parent = edges[:, 1].astype(int) - 1

    N = nodes.shape[0]
    children: list[list[int]] = [[] for _ in range(N)]
    for c, p in zip(child, parent):
        if p >= 0:
            children[p].append(c)

    # irreducible mask & quick parent lookup up to next irreducible
    irr_idx, _ = get_irreducible_nodes(nodes, edges)
    irr_mask = np.zeros(N, bool)
    irr_mask[irr_idx - 1] = True

    avg_angles: list[float] = []

    for n in irr_idx - 1:              # convert back to 0‑based
        # upstream irreducible parent
        p = parent[n]
        while p >= 0 and not irr_mask[p]:
            p = parent[p]
        if p < 0:                       # reached root without irreducible
            continue
        parent_vec = nodes[p] - nodes[n]
        norm_p = np.linalg.norm(parent_vec)
        if norm_p < 1e-6:
            continue

        # downstream irreducible children (could be ≥1)
        for c0 in children[n]:
            c = c0
            while c >= 0 and not irr_mask[c]:
                next_children = children[c]
                c = next_children[0] if next_children else -1
            if c < 0:
                continue
            child_vec = nodes[c] - nodes[n]
            norm_c = np.linalg.norm(child_vec)
            if norm_c < 1e-6:
                continue

            cosang = np.dot(parent_vec, child_vec) / (norm_p * norm_c)
            ang = np.arccos(np.clip(cosang, -1.0, 1.0))
            if ang > 0:
                avg_angles.append(ang)

    return float(np.mean(avg_angles)) if avg_angles else 0.0
