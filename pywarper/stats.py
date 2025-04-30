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


def get_median_branch_length(nodes: np.ndarray, edges: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """
    Median dendritic *segment* length (µm).

    A *segment* is the path between two successive “irreducible” nodes
    (degree ≠ 2).  Interior degree-2 nodes are collapsed.
    The soma/root row (parent = –1) counts as irreducible regardless of degree.

    Parameters
    ----------
    nodes : (N, 3) float
        XYZ coordinates in µm.
    edges : (E, 2) int
        1-based SWC child/parent pairs (root has parent = –1).

    Returns
    -------
    float
        Median length of all such segments, in µm.
    """
    # -------------------------------------------------------------
    # 1. per-edge lengths  (child row ↔ parent row)
    # -------------------------------------------------------------
    density, _ = segment_lengths(nodes, edges)
    child = edges[:, 0].astype(int) - 1
    parent = edges[:, 1].astype(int) - 1
    parent[child == 0] = -1          # soma row (0) has no parent

    edge_len = {tuple(sorted((c, p))): density[c]
                for c, p in zip(child, parent) if p >= 0}   # skip soma row

    # -------------------------------------------------------------
    # 2. build an undirected adjacency list and node degrees
    # -------------------------------------------------------------
    N = nodes.shape[0]
    adj: list[list[int]] = [[] for _ in range(N)]
    for (c, p) in edge_len.keys():
        adj[c].append(p)
        adj[p].append(c)

    degree = np.array([len(nbrs) for nbrs in adj])

    # “irreducible” = degree ≠ 2   (includes branch points, tips, soma)
    irreducible = (degree != 2)
    soma_rows   = np.flatnonzero(parent == -1)      # usually one row
    irreducible[edges[soma_rows, 0] - 1] = True     # force soma

    # -------------------------------------------------------------
    # 3. walk every edge once, collapse degree-2 chains
    # -------------------------------------------------------------
    visited = set()
    seg_lengths: list[float] = []

    for u in np.flatnonzero(irreducible):
        for v in adj[u]:
            edge = tuple(sorted((u, v)))
            if edge in visited:
                continue                             

            length = edge_len[edge]
            visited.add(edge)

            prev, curr = u, v
            while not irreducible[curr]:
                # the only neighbour that is *not* the one we came from
                nxt = adj[curr][0] if adj[curr][0] != prev else adj[curr][1]
                edge = tuple(sorted((curr, nxt)))
                length += edge_len[edge]
                visited.add(edge)
                prev, curr = curr, nxt

            seg_lengths.append(length)

    median_len = float(np.median(seg_lengths)) if seg_lengths else 0.0

    irreducible_rows = np.flatnonzero(irreducible)
    irreducible_nodes = nodes[irreducible_rows]

    return median_len, irreducible_nodes, irreducible_rows
