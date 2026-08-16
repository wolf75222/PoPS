#!/usr/bin/env python3
"""Typed exact-rank embedded-geometry and transport-mask contract."""

import pytest

from pops.mesh.geometry import Disc, NoWall
from pops.mesh.masks import CutCell, NoMask, Staircase, TransportMask, lower_transport_mask


# --------------------------------------------------------------------------------------------
# (1) LOWERING -- pure, no engine.
# --------------------------------------------------------------------------------------------


def test_disc_mode_lowers_only_typed_descriptors():
    assert issubclass(NoMask, TransportMask)
    assert NoMask().lower() == "none"
    assert Staircase().lower() == "staircase"
    assert CutCell().lower() == "cutcell"
    assert lower_transport_mask(NoMask()) == "none"
    assert lower_transport_mask(Staircase()) == "staircase"
    assert lower_transport_mask(CutCell()) == "cutcell"
    for token in ("none", "staircase", "cutcell", "bogus"):
        with pytest.raises(TypeError, match="TransportMask"):
            lower_transport_mask(token)


def test_transport_mask_extension_uses_the_small_typed_interface():
    class CustomStaircase(TransportMask):
        mode_token = "staircase"

    class UnsupportedMask(TransportMask):
        mode_token = "unsupported"

    assert lower_transport_mask(CustomStaircase()) == "staircase"
    with pytest.raises(ValueError, match="unsupported native transport token"):
        lower_transport_mask(UnsupportedMask())


def test_lowering_rejects_bad_inputs():
    with pytest.raises(TypeError):
        lower_transport_mask(42)
    with pytest.raises(ValueError):
        Disc(radius=-1.0)  # radius must be > 0
    with pytest.raises(ValueError, match="finite"):
        Disc(radius=float("inf"))
    with pytest.raises(TypeError, match="real number"):
        Disc(radius=True)


def test_descriptors_inspect_and_available_honestly():
    # CutCell honestly declares it needs embedded-boundary support.
    cc = CutCell().inspect()
    assert cc["category"] == "transport_mask"
    assert cc["requirements"] == {"embedded_boundary_support": True}
    assert cc["capabilities"]["conservative"] is True
    # NoMask is masked_transport=False (inert / bit-identical default).
    assert NoMask().inspect()["capabilities"] == {"masked_transport": False}
    # Disc / NoWall describe themselves as level-set geometries; neither configures native EB alone.
    assert NoWall().inspect()["capabilities"] == {"provides": "level_set"}
    assert Disc(radius=0.4).inspect()["category"] == "geometry"


# --------------------------------------------------------------------------------------------
# (2) RUNTIME ACCEPTANCE -- real engine, typed selectors only.
# --------------------------------------------------------------------------------------------

try:
    import numpy as np
    import pops
    import pops._pops  # noqa: F401

    _HAVE_ENGINE = True
except ImportError:  # pragma: no cover - environment without the build
    _HAVE_ENGINE = False


requires_engine = pytest.mark.skipif(
    not _HAVE_ENGINE, reason="compiled pops extension absent (PYTHONPATH / build?)"
)


def _build(n=32, L=1.0):
    from pops.runtime._system import System, SystemConfig  # advanced native runtime seam

    config = SystemConfig()
    config.shape = (n, n)
    config.lower = (0.0, 0.0)
    config.upper = (float(L), float(L))
    config.periodicity = (False, False)
    config.boxes = (((0, 0), (n, n)),)
    return System(config)


def _install_half_space(system, mode: str) -> None:
    from pops.analytic import coordinates
    from pops.domain import CartesianDomain
    from pops.mesh.geometry import LevelSet
    from pops.runtime._analytic_expression_lowering import lower_analytic_components

    frame = CartesianDomain("test-typed-eb", (0.0, 0.0), (1.0, 1.0)).frame()
    level_set = LevelSet(coordinates(frame)[0] - 0.5)
    ((opcodes, literals),) = lower_analytic_components(
        (level_set.expression.to_data(),), frame_id=frame.canonical_id
    )
    system._s._set_analytic_level_set(
        list(opcodes), list(literals), mode, 0.0, 0.0, 0.0
    )


@requires_engine
def test_analytic_level_set_configures_typed_transport_mode():
    system = _build()
    _install_half_space(system, NoMask().lower())
    system.set_geometry_mode(NoMask())
    mask = np.array(system.embedded_boundary_mask())
    assert mask.shape == (32, 32)
    assert 0 < int(mask.sum()) < 32 * 32


@requires_engine
def test_runtime_rejects_untyped_mode_selector():
    with pytest.raises(TypeError, match="TransportMask"):
        _build().set_geometry_mode("none")


@requires_engine
def test_uniform_set_poisson_has_no_wall_selector():
    with pytest.raises(TypeError, match="unexpected keyword argument 'wall'"):
        _build().set_poisson(wall=NoWall())
    with pytest.raises(TypeError, match="string selectors"):
        _build().set_poisson(bc="dirichlet")
    with pytest.raises(TypeError, match="wall_radius"):
        _build().set_poisson(wall_radius=0.4)
