#!/usr/bin/env python3
"""Typed-only disc-domain, transport-mask, and Poisson-wall contract."""

import pytest

from pops.mesh.geometry import Disc, NoWall, DiscDomain, HalfPlane
from pops.mesh.masks import TransportMask, NoMask, Staircase, CutCell, lower_disc_mode
from pops.runtime._system import System  # ADC-545 advanced runtime seam


# --------------------------------------------------------------------------------------------
# (1) LOWERING -- pure, no engine. Tokens remain private implementation output.
# --------------------------------------------------------------------------------------------

def test_wall_lowers_to_private_native_tokens():
    assert Disc(radius=0.4).lower_wall() == ("circle", 0.4)
    with pytest.raises(ValueError, match="explicit center is not supported"):
        Disc(center=(0.5, 0.5), radius=0.123).lower_wall()
    assert NoWall().lower_wall() == ("none", 0.0)


def test_disc_mode_lowers_only_typed_descriptors():
    assert issubclass(NoMask, TransportMask)
    assert NoMask().lower() == "none"
    assert Staircase().lower() == "staircase"
    assert CutCell().lower() == "cutcell"
    assert lower_disc_mode(NoMask()) == "none"
    assert lower_disc_mode(Staircase()) == "staircase"
    assert lower_disc_mode(CutCell()) == "cutcell"
    for token in ("none", "staircase", "cutcell", "bogus"):
        with pytest.raises(TypeError, match="TransportMask"):
            lower_disc_mode(token)


def test_transport_mask_extension_uses_the_small_typed_interface():
    class CustomStaircase(TransportMask):
        mode_token = "staircase"

    class UnsupportedMask(TransportMask):
        mode_token = "unsupported"

    assert lower_disc_mode(CustomStaircase()) == "staircase"
    with pytest.raises(ValueError, match="unsupported native transport token"):
        lower_disc_mode(UnsupportedMask())


def test_disc_domain_lowers_to_set_disc_domain_args():
    dd = DiscDomain(center=(0.5, 0.5), radius=0.4, mode=CutCell())
    assert dd.lower() == (0.5, 0.5, 0.4, "cutcell")
    # Default mode is the inert NoMask -> "none" (mask only, full Cartesian transport).
    assert DiscDomain(center=(0.25, 0.75), radius=0.3).lower() == (0.25, 0.75, 0.3, "none")
    with pytest.raises(TypeError, match="TransportMask"):
        DiscDomain(center=(0.0, 0.0), radius=0.2, mode="staircase")


def test_lowering_rejects_bad_inputs():
    with pytest.raises(TypeError):
        lower_disc_mode(42)
    with pytest.raises(TypeError):
        HalfPlane().lower_wall()           # a half-plane is not a Poisson wall
    with pytest.raises(ValueError):
        Disc(radius=-1.0)                  # radius must be > 0
    with pytest.raises(ValueError):
        DiscDomain(center=(0, 0), radius=0.0)  # radius must be > 0
    with pytest.raises(ValueError, match="exactly two"):
        DiscDomain(center=(0, 0, 0), radius=1.0)
    with pytest.raises(ValueError, match="finite"):
        DiscDomain(center=(float("nan"), 0), radius=1.0)
    with pytest.raises(ValueError, match="finite"):
        Disc(radius=float("inf"))
    with pytest.raises(TypeError, match="real number"):
        Disc(radius=True)
    with pytest.raises(TypeError, match="real number"):
        DiscDomain(center=(False, 0), radius=1.0)


def test_descriptors_inspect_and_available_honestly():
    # CutCell honestly declares it needs embedded-boundary support.
    cc = CutCell().inspect()
    assert cc["category"] == "transport_mask"
    assert cc["requirements"] == {"embedded_boundary_support": True}
    assert cc["capabilities"]["conservative"] is True
    # NoMask is masked_transport=False (inert / bit-identical default).
    assert NoMask().inspect()["capabilities"] == {"masked_transport": False}
    # DiscDomain surfaces its mode's requirements + an explainable availability.
    dd = DiscDomain(center=(0.5, 0.5), radius=0.4, mode=CutCell())
    insp = dd.inspect()
    assert insp["category"] == "disc_domain"
    assert insp["options"]["mode"] == "CutCell"
    assert insp["requirements"] == {"embedded_boundary_support": True}
    assert dd.available().ok  # yes (the mask is available; the runtime gates the native physics)
    # Disc / NoWall walls describe themselves as level-set geometries.
    assert NoWall().inspect()["capabilities"]["wall"] is False
    assert Disc(radius=0.4).inspect()["category"] == "geometry"


# --------------------------------------------------------------------------------------------
# (2) RUNTIME ACCEPTANCE -- real engine, typed selectors only.
# --------------------------------------------------------------------------------------------

try:
    import numpy as np
    import pops
    from pops.runtime.bricks import Dirichlet
    import pops._pops  # noqa: F401
    _HAVE_ENGINE = True
except ImportError:  # pragma: no cover - environment without the build
    _HAVE_ENGINE = False


requires_engine = pytest.mark.skipif(
    not _HAVE_ENGINE, reason="compiled pops extension absent (PYTHONPATH / build?)")


def _build(n=32, L=1.0):
    return System(n=n, L=L, periodic=False)


@requires_engine
def test_set_disc_domain_accepts_typed_disc_domain():
    system = _build()
    system.set_disc_domain(DiscDomain(center=(0.5, 0.5), radius=0.3, mode=NoMask()))
    system.set_geometry_mode(NoMask())
    mask = np.array(system.disc_mask())
    assert mask.shape == (32, 32)
    assert 0 < int(mask.sum()) < 32 * 32


@requires_engine
def test_runtime_rejects_untyped_disc_and_mode_selectors():
    with pytest.raises(TypeError, match="DiscDomain"):
        _build().set_disc_domain("cutcell")
    with pytest.raises(TypeError):
        _build().set_disc_domain(0.5, 0.5, 0.3, "none")
    with pytest.raises(TypeError, match="TransportMask"):
        _build().set_geometry_mode("none")


@requires_engine
def test_set_poisson_accepts_typed_circle_wall():
    system = _build()
    system.set_poisson(
        rhs="charge_density", solver="geometric_mg",
        bc=Dirichlet(), wall=Disc(radius=0.4))
    assert system.poisson_solver() == "geometric_mg"


@requires_engine
def test_set_poisson_typed_no_wall_matches_omission():
    explicit = _build()
    omitted = _build()
    explicit.set_poisson(bc=Dirichlet(), wall=NoWall())
    omitted.set_poisson(bc=Dirichlet())
    assert explicit.poisson_solver() == omitted.poisson_solver() == "geometric_mg"


@requires_engine
def test_set_poisson_rejects_untyped_boundary_and_wall_selectors():
    with pytest.raises(TypeError, match="wall must be a typed"):
        _build().set_poisson(wall=12345)
    with pytest.raises(ValueError, match="explicit center is not supported"):
        _build().set_poisson(wall=Disc(center=(0.25, 0.25), radius=0.4))
    with pytest.raises(TypeError, match="string selectors"):
        _build().set_poisson(wall="none")
    with pytest.raises(TypeError, match="string selectors"):
        _build().set_poisson(bc="dirichlet")
    with pytest.raises(TypeError, match="wall_radius"):
        _build().set_poisson(wall_radius=0.4)
    from pops.runtime._system_install import _lower_wall
    with pytest.raises(TypeError, match="string selectors"):
        _lower_wall("circle")
