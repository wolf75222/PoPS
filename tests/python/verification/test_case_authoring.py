"""Uniform periodic layout helper (Phase 1 shared authoring)."""
from __future__ import annotations

import pytest

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


def test_bind_public_does_not_fall_back_to_serial_when_mpi_is_requested(monkeypatch):
    import pops
    import verification.pops_verify.case_authoring as authoring

    bind_calls: list[object] = []

    def fake_mpi_world(_artifact):
        raise RuntimeError("the compiled artifact/runtime pair does not prove MPI_COMM_WORLD")

    def fake_bind(artifact, **kwargs):
        bind_calls.append({"artifact": artifact, "kwargs": kwargs})
        return "serial-bound"

    monkeypatch.setattr(authoring.pops.ExecutionContext, "mpi_world", staticmethod(fake_mpi_world))
    monkeypatch.setattr(authoring.pops, "bind", fake_bind)
    monkeypatch.setattr(pops, "bind", fake_bind)

    with pytest.raises(RuntimeError, match="MPI_COMM_WORLD"):
        authoring.bind_public(object(), mpi_mode="on")
    assert bind_calls == []


def test_bind_public_serial_bind_when_mpi_is_off(monkeypatch):
    import verification.pops_verify.case_authoring as authoring

    def fake_mpi_world(_artifact):
        raise AssertionError("serial bind must not probe MPI_COMM_WORLD")

    monkeypatch.setattr(authoring.pops.ExecutionContext, "mpi_world", staticmethod(fake_mpi_world))
    monkeypatch.setattr(authoring.pops, "bind", lambda artifact, **kwargs: "serial-bound")

    assert authoring.bind_public(object(), mpi_mode="off") == "serial-bound"
