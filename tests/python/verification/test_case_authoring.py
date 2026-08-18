"""Uniform periodic layout helper (Phase 1 shared authoring)."""
from __future__ import annotations

from pops.domain import CartesianDomain
from pops.frames import Cartesian1D
from pops.layouts import Uniform
from pops.mesh import CartesianGrid

from verification.pops_verify.case_authoring import (
    uniform_open_layout,
    uniform_periodic_layout,
)


def test_uniform_periodic_layout_is_1d_uniform():
    frame = CartesianDomain("line", (0.0,), (1.0,)).frame(Cartesian1D())
    layout = uniform_periodic_layout(frame, (16,))
    assert isinstance(layout, Uniform)
    geometry = layout.normalized_geometry()
    assert geometry.cells == (16,)


def test_uniform_open_layout_has_no_periodic_axes():
    frame = CartesianDomain("line", (0.0,), (1.0,)).frame(Cartesian1D())
    layout = uniform_open_layout(frame, (16,))
    assert isinstance(layout, Uniform)
    geometry = layout.normalized_geometry()
    assert geometry.cells == (16,)
    periodic = layout.mesh.periodic
    assert periodic is None or tuple(getattr(periodic, "axes", ())) == ()
