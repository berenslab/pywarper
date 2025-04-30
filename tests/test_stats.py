import numpy as np
import pytest

import pywarper.stats as stats

# --------------------------------------------------------------------------- #
# Shared toy SWC tree (5 nodes, 4 dendritic edges)
#            (5)
#             |
#  (1)──(2)──(3)──(4)
#   ^ soma
# --------------------------------------------------------------------------- #
NODES = np.array(
    [
        [0.0, 0.0, 0.0],   # 1 – soma
        [0.0, 1.0, 0.0],   # 2
        [1.0, 1.0, 0.0],   # 3
        [1.0, 0.0, 0.0],   # 4
        [0.5, 1.5, 0.0],   # 5
    ],
    dtype=float,
)
EDGES = np.array(
    [
        [1, -1],   # soma root
        [2,  1],
        [3,  2],
        [4,  3],
        [5,  2],
    ],
    dtype=int,
)



# --------------------------------------------------------------------------- #
# Geometry helpers                                                            #
# --------------------------------------------------------------------------- #
def test_get_convex_hull_square():
    pts = np.array([[0, 0], [2, 0], [2, 2], [0, 2], [1, 1]])
    hull = stats.get_convex_hull(pts)
    # hull may include the first vertex twice (closed ring) → 4 or 5 rows
    assert hull.shape in {(4, 2), (5, 2)}
    assert pytest.approx(stats.get_hull_area(hull)) == 4.0
    cx, cy = stats.get_hull_centroid(hull)
    assert (pytest.approx(cx), pytest.approx(cy)) == (1.0, 1.0)


# --------------------------------------------------------------------------- #
# Center-of-mass and soma features                                            #
# --------------------------------------------------------------------------- #
def test_center_of_mass_and_asymmetry():
    x = np.linspace(0, 2, 3)          # 0, 1, 2 µm
    y = np.linspace(0, 1, 2)          # 0, 1 µm
    xv, yv = np.meshgrid(x, y, indexing="ij")
    xy_dist = np.ones_like(xv)        # uniform “mass”
    com_xy = stats.get_xy_center_of_mass(x, y, xy_dist)
    assert tuple(np.round(com_xy, 6)) == (1.0, 0.5)

    z_axis = np.array([0, 1, 2])
    z_dist = np.array([1, 1, 2])
    assert pytest.approx(stats.get_z_center_of_mass(z_axis, z_dist)) == 1.25

    soma_xy = np.array([0.0, 0.0])
    asym = stats.get_asymmetry(soma_xy, com_xy)
    assert pytest.approx(asym) == np.hypot(1.0, 0.5)


def test_soma_to_stratification_depth():
    assert stats.get_soma_to_stratification_depth(30.0, 25.0) == 5.0
    assert stats.get_soma_to_stratification_depth(25.0, 30.0) == 5.0


# --------------------------------------------------------------------------- #
# SWC morphology statistics                                                   #
# --------------------------------------------------------------------------- #
def test_branch_point_count():
    # Node 2 has two children → exactly one branch point.
    assert stats.get_branch_point_count(EDGES) == 1


def test_dendritic_length_and_median_segment_len():
    # edge lengths: 1, 1, 1, √½ ≈ 0.7071 µm → total ≈ 3.7071
    assert pytest.approx(stats.get_dendritic_length(NODES, EDGES)) == 3.7071067811865475
    # irreducible segments are those four same edges → median = 1
    assert pytest.approx(stats.get_median_branch_length(NODES, EDGES)) == 1.0


def test_average_tortuosity():
    # Mix of straight and one bent segment → mean tortuosity ≈ 1.1381
    assert pytest.approx(stats.get_average_tortuosity(NODES, EDGES)) == 1.1380711874576983


def test_typical_radius():
    com_xy = np.array([0.5, 0.7])
    assert pytest.approx(stats.get_typical_radius(NODES, EDGES, com_xy)) == 0.500355212883135


def test_average_angle_simple_bend():
    # The branch (2-3-4) forms a right angle → average angle = π/2
    assert pytest.approx(stats.get_average_angle(NODES, EDGES)) == np.pi / 2
