from __future__ import annotations

import copy
from dataclasses import FrozenInstanceError
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from pops.domain import (
    BoundarySide,
    DomainBoundary,
    Rectangle,
    RectangleBoundaries,
    RectangleBoundaryNames,
    RectangleFrame,
)
from pops.frames import Cartesian2D, CartesianAxis, CartesianDirection
from pops.mesh.grid import CartesianGrid, PeriodicAxes


def _framed_domain() -> RectangleFrame:
    names = RectangleBoundaryNames(
        x_min="inlet", x_max="outlet", y_min="bottom", y_max="top")
    return Rectangle(
        "unit_square", (0, 0), (1, 1), boundaries=names,
    ).tag("fluid").frame(Cartesian2D())


def test_cartesian_axes_are_typed_immutable_and_canonical() -> None:
    frame = Cartesian2D()
    x, y = frame.axes

    assert x is frame.x
    assert y is frame.y
    assert (x.direction, x.index, x.name) == (CartesianDirection.X, 0, "x")
    assert (y.direction, y.index, y.name) == (CartesianDirection.Y, 1, "y")
    assert len({x, y}) == 2
    assert Cartesian2D.from_dict(frame.to_dict()) == frame
    assert CartesianAxis.from_dict(x.to_dict()) == x
    assert json.loads(json.dumps(frame.to_dict())) == frame.to_dict()

    with pytest.raises(FrozenInstanceError):
        x.direction = CartesianDirection.Y  # type: ignore[misc]
    with pytest.raises(TypeError, match="CartesianDirection"):
        CartesianAxis("x")  # type: ignore[arg-type]


def test_frame_decoder_refuses_unknown_or_noncanonical_axis_order() -> None:
    payload = Cartesian2D().to_dict()
    extra = copy.deepcopy(payload)
    extra["name"] = "legacy"
    with pytest.raises(TypeError, match="unsupported shape"):
        Cartesian2D.from_dict(extra)

    swapped = copy.deepcopy(payload)
    swapped["axes"].reverse()
    with pytest.raises(ValueError, match="not canonical"):
        Cartesian2D.from_dict(swapped)


def test_rectangle_validates_extent_and_boundary_declarations() -> None:
    with pytest.raises(TypeError, match="exactly two"):
        Rectangle("bad", (0,), (1,))
    with pytest.raises(ValueError, match="strictly greater"):
        Rectangle("bad", (0, 0), (0, 1))
    with pytest.raises(ValueError, match="finite"):
        Rectangle("bad", (0, 0), (float("inf"), 1))
    with pytest.raises(TypeError, match="never bool"):
        Rectangle("bad", (False, 0), (1, 1))
    with pytest.raises(TypeError, match="RectangleBoundaryNames"):
        Rectangle("bad", (0, 0), (1, 1), boundaries={"x_min": "left"})
    with pytest.raises(ValueError, match="unique"):
        RectangleBoundaryNames(x_min="wall", x_max="wall")


def test_rectangle_tagging_is_immutable_canonical_and_keeps_boundary_identity() -> None:
    base = Rectangle("box", (0, -1), (2, 3))
    first = base.tag("plasma").tag("fluid")
    second = base.tag("fluid").tag("plasma")

    assert base.tags == ()
    assert first == second
    assert first.canonical_id == second.canonical_id
    assert first.boundaries == base.boundaries
    assert first.tag("fluid") is first
    assert Rectangle.from_dict(first.to_dict()) == first
    assert json.loads(json.dumps(first.to_dict())) == first.to_dict()

    duplicate = copy.deepcopy(first.to_dict())
    duplicate["tags"].append(copy.deepcopy(duplicate["tags"][0]))
    with pytest.raises(ValueError, match="not canonical"):
        Rectangle.from_dict(duplicate)


def test_boundaries_are_typed_stable_and_have_no_string_selector() -> None:
    frame = _framed_domain()
    boundaries = frame.boundaries

    assert boundaries.x_min.name == "inlet"
    assert boundaries.x_min.side is BoundarySide.LOWER
    assert boundaries.x_max.side is BoundarySide.UPPER
    assert boundaries.pair(frame.coordinates.x).lower is boundaries.x_min
    assert boundaries.pair(frame.coordinates.y).upper is boundaries.y_max
    assert tuple(boundary.name for boundary in boundaries.all) == (
        "inlet", "outlet", "bottom", "top")
    assert RectangleBoundaries.from_dict(boundaries.to_dict()) == boundaries
    assert DomainBoundary.from_dict(boundaries.x_min.to_dict()) == boundaries.x_min

    with pytest.raises(TypeError, match="never a string"):
        boundaries.pair("x")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="Cartesian2D"):
        frame.domain.frame("cartesian")


def test_rectangle_frame_round_trip_authenticates_domain_and_coordinates() -> None:
    frame = _framed_domain()
    rebuilt = RectangleFrame.from_dict(frame.to_dict())

    assert rebuilt == frame
    assert rebuilt.canonical_id == frame.canonical_id
    forged = copy.deepcopy(frame.to_dict())
    forged["coordinates"]["dimension"] = 3
    with pytest.raises(ValueError, match="unsupported schema"):
        RectangleFrame.from_dict(forged)


def test_cartesian_grid_derives_order_extent_topology_and_widths() -> None:
    frame = _framed_domain()
    grid = CartesianGrid(frame=frame, cells=(8, 4))

    assert grid.axis_order == frame.axes
    assert grid.cells == (8, 4)
    assert grid.extent == ((0.0, 0.0), (1.0, 1.0))
    assert grid.cell_widths == (0.125, 0.25)
    assert tuple(pair.axis for pair in grid.topology.axis_pairs) == frame.axes
    assert tuple(pair.lower for pair in grid.topology.axis_pairs) == (
        frame.boundaries.x_min, frame.boundaries.y_min)
    assert grid.options()["axis_order"] == ["x", "y"]
    assert grid.capabilities().to_dict() == {
        "geometry": "cartesian", "dim": 2, "bounded_axes": 2, "periodic_axes": 0,
    }
    assert grid.requirements().to_dict() == {}
    assert grid.validate() is True


def test_cartesian_grid_periodicity_is_a_typed_axis_partition() -> None:
    frame = _framed_domain()
    periodic = PeriodicAxes(frame.axes)
    grid = CartesianGrid(frame=frame, cells=(8, 8), periodic=periodic)

    assert grid.topology.periodic_axes == frame.axes
    assert grid.topology.physical_axes == ()
    assert all(grid.topology.is_periodic(axis) for axis in frame.axes)
    assert grid.capabilities().to_dict() == {
        "geometry": "cartesian", "dim": 2, "bounded_axes": 0, "periodic_axes": 2,
    }
    assert CartesianGrid.from_dict(grid.to_dict()) == grid
    assert PeriodicAxes.from_dict(periodic.to_dict()) == periodic

    with pytest.raises(TypeError, match="never bool"):
        CartesianGrid(frame=frame, cells=(8, 8), periodic=True)
    with pytest.raises(ValueError, match="more than once"):
        PeriodicAxes((frame.x, frame.x))
    with pytest.raises(ValueError, match="canonical frame-axis order"):
        PeriodicAxes((frame.y, frame.x))


def test_cartesian_grid_is_immutable_json_serializable_and_fail_closed() -> None:
    grid = CartesianGrid(frame=_framed_domain(), cells=(3, 5))
    payload = grid.to_dict()

    assert CartesianGrid.from_dict(payload) == grid
    assert json.loads(json.dumps(payload)) == payload
    assert CartesianGrid.from_dict(payload).canonical_id == grid.canonical_id
    with pytest.raises(FrozenInstanceError):
        grid.cells = (4, 4)  # type: ignore[misc]
    with pytest.raises(TypeError, match="RectangleFrame"):
        CartesianGrid(frame=Cartesian2D(), cells=(4, 4))
    with pytest.raises(TypeError, match="exactly two"):
        CartesianGrid(frame=_framed_domain(), cells=(4,))
    with pytest.raises(TypeError, match="never bool"):
        CartesianGrid(frame=_framed_domain(), cells=(True, 4))
    with pytest.raises(ValueError, match=">= 1"):
        CartesianGrid(frame=_framed_domain(), cells=(0, 4))

    forged = copy.deepcopy(payload)
    forged["axis_order"] = ["y", "x"]
    with pytest.raises(ValueError, match="not canonical"):
        CartesianGrid.from_dict(forged)


def test_geometry_packages_and_grid_import_without_native_extension() -> None:
    root = Path(__file__).resolve().parents[4]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root / "python")
    program = """
import json
import sys
from pops.frames import Cartesian2D
from pops.domain import Rectangle
from pops.mesh.grid import CartesianGrid
domain = Rectangle('box', (0, 0), (2, 1)).tag('fluid')
frame = domain.frame(Cartesian2D())
grid = CartesianGrid(frame=frame, cells=(8, 4))
assert 'pops._pops' not in sys.modules
print(json.dumps(grid.to_dict(), sort_keys=True))
"""
    completed = subprocess.run(
        [sys.executable, "-c", program], cwd=root, env=env,
        text=True, capture_output=True, check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout)["grid_type"] == "cartesian"
