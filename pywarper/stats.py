import numpy as np
from alphashape import alphashape


def get_convex_hull(points: np.ndarray) -> np.ndarray:
    """Get the convex hull of a set of points."""
    if len(points) < 3:
        return points
    hull = alphashape(points, alpha=0)
    return np.array(hull.exterior.xy).T