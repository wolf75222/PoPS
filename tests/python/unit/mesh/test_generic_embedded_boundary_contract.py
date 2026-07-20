from __future__ import annotations

import pytest

from pops.boundary import EmbeddedBoundaryFlux, ZeroFlux
from pops.analytic import x
from pops.layouts import Uniform
from pops.mesh.geometry import Disc, EmbeddedBoundary, Geometry, LevelSet
from pops.mesh.masks import CutCell, Staircase, TransportMask
from tests.python.support.layout_plan import cartesian_grid


def _annulus() -> object:
    return Disc(center=(0.5, 0.5), radius=0.4) - Disc(
        center=(0.5, 0.5), radius=0.2
    )


def test_generic_csg_uses_the_real_staircase_route() -> None:
    layout = Uniform(
        cartesian_grid(),
        embedded_boundary=EmbeddedBoundary(_annulus(), Staircase(), ZeroFlux()),
    )

    embedded = layout.options()["embedded_boundary"]
    assert embedded["schema_version"] == 1
    assert embedded["boundary"] == {"provider": "zero_flux"}
    assert embedded["transport"]["mode"] == "staircase"
    assert embedded["level_set"]["active_when"] == "phi<0"


def test_generic_csg_cannot_claim_the_disc_only_cutcell_route() -> None:
    with pytest.raises(NotImplementedError, match="true face apertures"):
        Uniform(
            cartesian_grid(),
            embedded_boundary=EmbeddedBoundary(_annulus(), CutCell(), ZeroFlux()),
        )


def test_extension_cannot_alias_the_disc_only_cutcell_route_for_csg() -> None:
    class AliasedCutCell(TransportMask):
        mode_token = "cutcell"

    with pytest.raises(NotImplementedError, match="true face apertures"):
        Uniform(
            cartesian_grid(),
            embedded_boundary=EmbeddedBoundary(
                _annulus(), AliasedCutCell(), ZeroFlux()
            ),
        )


def test_embedded_boundary_requires_an_explicit_supported_flux_provider() -> None:
    class Reflective(EmbeddedBoundaryFlux):
        provider_token = "reflective"

    with pytest.raises(TypeError, match="missing 1 required positional argument"):
        EmbeddedBoundary(_annulus(), Staircase())
    with pytest.raises(ValueError, match="unsupported embedded-boundary flux provider"):
        EmbeddedBoundary(_annulus(), Staircase(), Reflective())


def test_resolved_uniform_layout_never_recalls_a_mutable_geometry_provider() -> None:
    class MutableGeometry(Geometry):
        def __init__(self) -> None:
            self.offset = 0.25
            self.calls = 0

        def level_set(self, frame):
            self.calls += 1
            return LevelSet(x(frame) - self.offset)

    geometry = MutableGeometry()
    layout = Uniform(
        cartesian_grid(),
        embedded_boundary=EmbeddedBoundary(geometry, Staircase(), ZeroFlux()),
    )
    authored = layout.options()["embedded_boundary"]
    assert geometry.calls == 2

    geometry.offset = 0.75
    resolved = layout.resolve_for_case(lambda value: value)
    resolved.validate()

    assert geometry.calls == 2
    assert resolved.options()["embedded_boundary"] == authored
    assert resolved.embedded_boundary is None
