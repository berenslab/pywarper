from typing import Optional

import numpy as np
from alphashape import alphashape

from .arbor import segment_lengths


def get_convex_hull(points: np.ndarray) -> np.ndarray:
    """Get the convex hull of a set of points."""
    if len(points) < 3:
        return points
    hull = alphashape(points, alpha=0)
    return np.array(hull.exterior.xy).T

# -----------------------------------------------------------------------------
# Polygon area & centroid (Shoelace formula) ----------------------------------
# -----------------------------------------------------------------------------

def _polygon_area(vertices: np.ndarray) -> float:
    """Signed area of a simple polygon given by *vertices* (M×2)."""
    x = vertices[:, 0]
    y = vertices[:, 1]
    return 0.5 * (np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _polygon_centroid(vertices: np.ndarray, signed_area: Optional[float] = None) -> np.ndarray:
    """Centroid of a simple polygon (x, y).  Falls back to the arithmetic mean
    if the area is degenerate (≈ 0).
    """
    if signed_area is None:
        signed_area = _polygon_area(vertices)
    if np.isclose(signed_area, 0.0):
        return vertices.mean(axis=0)

    x = vertices[:, 0]
    y = vertices[:, 1]
    shift_x = np.roll(x, -1)
    shift_y = np.roll(y, -1)
    cross = x * shift_y - shift_x * y
    cx = ((x + shift_x) * cross).sum() / (6.0 * signed_area)
    cy = ((y + shift_y) * cross).sum() / (6.0 * signed_area)
    return np.array([cx, cy], dtype=float)


# Public wrappers that compute the convex hull first ---------------------------


def get_hull_area(hull: np.ndarray) -> float:
    """Area (µm²) of the convex hull enclosing *points* (N×2)."""
    area = _polygon_area(hull)
    return abs(area)  # return unsigned


def get_hull_centroid(hull: np.ndarray) -> np.ndarray:
    """Centroid (x, y in µm) of the convex hull enclosing *points*."""
    return _polygon_centroid(hull, _polygon_area(hull))

def get_xy_center_of_mass(
        x: np.ndarray, 
        y: np.ndarray, 
        xy_dist: np.ndarray
    ) -> np.ndarray:
    """Return ((com_x, com_y), com_z) in µm."""
    com_x = (xy_dist.sum(axis=1) @ x) / xy_dist.sum()      # integrate over y
    com_y = (xy_dist.sum(axis=0) @ y) / xy_dist.sum()      # integrate over x
    return np.array([com_x, com_y])

def get_z_center_of_mass(
        z_x: np.ndarray, 
        z_dist: np.ndarray
    )-> float:
    """Return com_z in µm."""
    com_z = (z_dist @ z_x) / z_dist.sum()
    return com_z

def get_asymmetry(soma_xy: np.ndarray, com_xy: np.ndarray) -> float:
    """Return the asymmetry of the soma."""
    dx = soma_xy[0] - com_xy[0]
    dy = soma_xy[1] - com_xy[1]
    asym = np.hypot(dx, dy)

    return asym

def get_soma_to_stratification_depth(soma_z: float, com_z: float) -> float:
    """Return the soma to stratification depth."""
    dz = soma_z - com_z
    return np.abs(dz)

def get_branch_point_count(edges: np.ndarray) -> int:
    """
    Return the number of dendritic branch points in a 1-based SWC `edges`
    array (E×2).  The soma (parent = –1) is not counted.
    """
    # 1. parents of all non-root rows, converted to 0-based
    parents = edges[edges[:, 1] > 0, 1].astype(int) - 1      # shape (E_nonroot,)

    # 2. how many times does each node appear as a parent?
    child_counts = np.bincount(parents)

    # 3. indices whose out-degree ≥ 2  → branch points
    n_branch_nodes = np.count_nonzero(child_counts >= 2)
    return int(n_branch_nodes)

def get_dendritic_length(nodes: np.ndarray, edges: np.ndarray) -> float:
    """Total cable length of an SWC tree in µm."""
    density, _ = segment_lengths(nodes, edges)
    return float(density.sum())

def _build_adjacency(edge_pairs: list[tuple[int, int]], n_nodes: int) -> list[list[int]]:
    """Return an undirected adjacency list from 0‑based *edge_pairs*."""
    adj: list[list[int]] = [[] for _ in range(n_nodes)]
    for u, v in edge_pairs:
        adj[u].append(v)
        adj[v].append(u)
    return adj


def get_irreducible_nodes(nodes: np.ndarray, edges: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Identify *irreducible* nodes = degree ≠ 2 ∪ soma.

    Parameters
    ----------
    nodes  : (N, 3) float
        Coordinates (µm).
    edges  : (E, 2) int
        1‑based SWC child/parent pairs. Parent = –1 marks the soma row.
    return_xyz : bool, default *True*
        Also return the coordinates of the same nodes.

    Returns
    -------
    idx_1b : (K,) int
        1‑based indices of irreducible nodes.
    xyz     : (K, 3) float | None
        Coordinates (µm) or *None* if *return_xyz* is *False*.
    """
    child = edges[:, 0].astype(int) - 1            # 0‑based child ID
    parent = edges[:, 1].astype(int) - 1           # –1 → soma row(s)

    # Filter genuine edges (exclude soma root rows)
    valid = parent >= 0
    edge_pairs = [tuple(sorted((c, p))) for c, p in zip(child[valid], parent[valid])]

    N = nodes.shape[0]
    adj = _build_adjacency(edge_pairs, N)
    degree = np.fromiter((len(n) for n in adj), int, count=N)

    irreducible_mask = degree != 2
    # ensure soma/root is included even if degree == 2
    soma_rows = np.flatnonzero(edges[:, 1] == -1)
    irreducible_mask[edges[soma_rows, 0] - 1] = True

    idx_1b = np.nonzero(irreducible_mask)[0] + 1     # back to 1‑based
    xyz = nodes[idx_1b - 1]
    return idx_1b, xyz


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

# -----------------------------------------------------------------------------
# Internal: combined scan over irreducible segments
# -----------------------------------------------------------------------------

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

def typical_radius(nodes: np.ndarray, edges: np.ndarray, com_xy: np.ndarray) -> float:
    """
    Root-mean-square planar distance (µm) of dendritic cable to COM(xy).
    """
    density, mid = segment_lengths(nodes, edges)
    dx = mid[:, 0] - com_xy[0]
    dy = mid[:, 1] - com_xy[1]
    return float(np.sqrt(np.sum(density * (dx**2 + dy**2)) / density.sum()))

def average_angle(nodes: np.ndarray, edges: np.ndarray) -> float:
    """Average positive angle (rad) at irreducible branch points.

    For each irreducible node that has **one upstream** irreducible parent and
    **≥1 downstream** irreducible child(ren), compute the angle between the
    (parent→node) and (node→child) vectors.  The feature is the mean of those
    angles.  Tips (degree 1) contribute nothing; multifurcations contribute
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
