"""The GAM fitter for SAC surfaces, and the return type both fitters share."""

import numpy as np
import pytest

from pygridfit.autosmooth import GAMMA_DEFAULT

from pywarper.surface import SacSurface, fit_sac_surface
from pywarper.utils import read_sumbul_et_al_chat_bands

DATA = "./tests/data"


@pytest.fixture(scope="module")
def chat_band():
    return read_sumbul_et_al_chat_bands(f"{DATA}/Image013-009_01_ChAT-TopBand-Mike.txt")


@pytest.fixture(scope="module")
def gam_surface(chat_band):
    return fit_sac_surface(
        x=chat_band["x"], y=chat_band["y"], z=chat_band["z"], method="gam", k=60
    )


def supported_mask(shape, x, y, halfwidth=30.0):
    """(nx, ny) mask of the x-span each annotated y-line actually covers.

    The annotation is a staircase -- the ChAT band leaves the imaged volume
    diagonally -- so a bounding box would include a wedge neither fitter has any
    data for, and where both extrapolate freely.
    """
    m = np.zeros(shape, dtype=bool)
    for yy in np.unique(y):
        xs = x[y == yy]
        y0 = int(max(0, np.floor(yy - halfwidth)))
        y1 = int(min(shape[1], np.ceil(yy + halfwidth)))
        x0, x1 = int(np.ceil(xs.min())), int(np.floor(xs.max()))
        m[max(x0, 0) : min(x1 + 1, shape[0]), y0:y1] = True
    return m


def test_orientation_and_grid(chat_band, gam_surface):
    """Surfaces are indexed (nx, ny), on the unit grid, whatever the fitter."""
    x, y = chat_band["x"], chat_band["y"]
    expected = (round(float(x.max())), round(float(y.max())))

    assert gam_surface.zmesh.shape == expected
    assert gam_surface.se.shape == expected
    assert gam_surface.xmesh.shape == expected
    assert gam_surface.ymesh.shape == expected
    # xmesh varies along axis 0, ymesh along axis 1
    assert np.allclose(gam_surface.xmesh[:, 0], np.arange(expected[0]))
    assert np.allclose(gam_surface.ymesh[0, :], np.arange(expected[1]))
    assert np.isfinite(gam_surface.zmesh).all()
    assert (gam_surface.se > 0).all()


def test_unpacks_as_three_tuple(chat_band, gam_surface):
    """The historical `zmesh, xmesh, ymesh = ...` call sites keep working."""
    zmesh, xmesh, ymesh = gam_surface
    assert zmesh is gam_surface.zmesh
    assert xmesh is gam_surface.xmesh
    assert ymesh is gam_surface.ymesh

    first, *rest = gam_surface
    assert first is gam_surface.zmesh
    assert len(rest) == 2

    gridfit = fit_sac_surface(x=chat_band["x"], y=chat_band["y"], z=chat_band["z"])
    assert isinstance(gridfit, SacSurface)
    zmesh, _, _ = gridfit
    assert zmesh.shape == gridfit.zmesh.shape
    assert gridfit.se is None


def test_agrees_with_gridfit_where_annotated(chat_band, gam_surface):
    """Inside the annotated region the two fitters answer the same question."""
    x, y, z = chat_band["x"], chat_band["y"], chat_band["z"]
    gridfit = fit_sac_surface(x=x, y=y, z=z, smoothness=15)

    assert gam_surface.zmesh.shape == gridfit.zmesh.shape
    supported = supported_mask(gridfit.zmesh.shape, x, y)
    diff = gam_surface.zmesh[supported] - gridfit.zmesh[supported]
    assert np.sqrt(np.mean(diff**2)) < 2.0
    # and neither runs away from the data where the data exists
    assert gam_surface.zmesh[supported].min() > z.min() - 25
    assert gam_surface.zmesh[supported].max() < z.max() + 25


def test_summary_reports_the_fit(chat_band, gam_surface):
    s = gam_surface.summary
    assert s["method"] == "gam"
    assert s["bs"] == "tp"
    assert s["k"] == 60
    # the smooth is smaller than its basis, i.e. REML actually penalized it
    assert 0 < s["smooth_edf"] < s["basis_dim"]
    assert np.all(np.asarray(s["lambda"]) > 0)
    assert not s["basis_saturated"]

    gridfit = fit_sac_surface(
        x=chat_band["x"], y=chat_band["y"], z=chat_band["z"], smoothness=15, stride=3
    )
    assert gridfit.summary == {
        "method": "gridfit",
        "smoothness": 15,
        "smoothness_auto": False,
        "stride": 3,
    }


def test_auto_smoothness_reports_what_it_chose(chat_band):
    """Under smoothness="auto" the summary must carry the selected number.

    Reporting the string back would leave no record of what actually shaped the
    surface, so a fit could not be reproduced from its own summary.
    """
    auto = fit_sac_surface(
        x=chat_band["x"], y=chat_band["y"], z=chat_band["z"], smoothness="auto"
    )
    assert auto.summary["smoothness_auto"] is True
    chosen = auto.summary["smoothness"]
    assert isinstance(chosen, float) and chosen > 0

    # and refitting at that number reproduces the surface it came from
    pinned = fit_sac_surface(
        x=chat_band["x"], y=chat_band["y"], z=chat_band["z"], smoothness=chosen
    )
    assert pinned.summary["smoothness_auto"] is False
    np.testing.assert_allclose(pinned.zmesh, auto.zmesh)


def test_auto_smoothness_records_the_gamma_that_shaped_it(chat_band):
    """The selected smoothness only means something alongside its gamma.

    gamma sets the length scale GCV optimises for, so two searches differing
    only in gamma return different numbers for the same data. A summary
    carrying the number without the gamma would not describe the fit.
    """
    x, y, z = chat_band["x"], chat_band["y"], chat_band["z"]
    default = fit_sac_surface(x=x, y=y, z=z, smoothness="auto")
    rougher = fit_sac_surface(x=x, y=y, z=z, smoothness="auto", gamma=4.0)

    assert default.summary["gamma"] == GAMMA_DEFAULT
    assert rougher.summary["gamma"] == 4.0
    assert default.summary["edf"] > 0

    # a larger gamma penalises degrees of freedom harder, so it must not come
    # back with a rougher fit than the default did
    assert rougher.summary["smoothness"] >= default.summary["smoothness"]
    assert rougher.summary["edf"] <= default.summary["edf"]


def test_gamma_without_a_search_is_refused(chat_band):
    """gamma steers a search; at a fixed smoothness there is none to steer."""
    x, y, z = chat_band["x"], chat_band["y"], chat_band["z"]
    with pytest.raises(ValueError, match="gamma"):
        fit_sac_surface(x=x, y=y, z=z, smoothness=15, gamma=2.0)


def test_gridfit_only_arguments_are_refused(chat_band):
    """A smoothness the GAM cannot honour is an error, not a silent no-op."""
    x, y, z = chat_band["x"], chat_band["y"], chat_band["z"]
    for kwargs in (
        {"smoothness": 15},
        {"stride": 3},
        {"gamma": 2.0},
        {"smoothness": 15, "stride": 3},
    ):
        with pytest.raises(ValueError, match="gridfit setting"):
            fit_sac_surface(x=x, y=y, z=z, method="gam", **kwargs)


def test_gam_only_arguments_are_refused(chat_band):
    """And symmetrically: gridfit has no basis to size."""
    x, y, z = chat_band["x"], chat_band["y"], chat_band["z"]
    for kwargs in ({"k": 60}, {"bs": "te"}, {"k": 60, "bs": "te"}):
        with pytest.raises(ValueError, match="gam setting"):
            fit_sac_surface(x=x, y=y, z=z, method="gridfit", **kwargs)


def test_unknown_method_is_refused(chat_band):
    with pytest.raises(ValueError, match="method must be"):
        fit_sac_surface(
            x=chat_band["x"], y=chat_band["y"], z=chat_band["z"], method="spline"
        )


def test_backward_compatible_grid_is_one_based(chat_band):
    """The GAM honours the 1-based output grid too, so the two fitters stay
    interchangeable downstream."""
    x, y, z = chat_band["x"], chat_band["y"], chat_band["z"]
    bc = fit_sac_surface(x=x, y=y, z=z, method="gam", k=60, backward_compatible=True)
    assert bc.zmesh.shape == (round(float(x.max())), round(float(y.max())))
    assert np.allclose(bc.xmesh[:, 0], np.arange(1, bc.zmesh.shape[0] + 1))
    assert np.allclose(bc.ymesh[0, :], np.arange(1, bc.zmesh.shape[1] + 1))
