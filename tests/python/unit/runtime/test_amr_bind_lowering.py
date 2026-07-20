"""AMR bind lowering preserves authored grid topology or refuses it explicitly."""
from __future__ import annotations

import pytest

from pops.domain import Rectangle
from pops.frames import Cartesian2D
from pops.mesh.grid import CartesianGrid, PeriodicAxes
from pops.runtime._amr_bind_lowering import _native_amr_grid_values


def _frame():
    return Rectangle("unit_square", (0, 0), (1, 1)).frame(Cartesian2D())


def test_native_amr_grid_preserves_none_or_all_periodic_axes() -> None:
    frame = _frame()
    closed = CartesianGrid(frame=frame, cells=(16, 16))
    periodic = CartesianGrid(
        frame=frame,
        cells=(16, 16),
        periodic=PeriodicAxes(frame.axes),
    )

    assert _native_amr_grid_values(closed.to_dict())[-1] is False
    assert _native_amr_grid_values(periodic.to_dict())[-1] is True


def test_native_amr_grid_refuses_partial_periodicity_without_erasing_it() -> None:
    frame = _frame()
    partial = CartesianGrid(
        frame=frame,
        cells=(16, 16),
        periodic=PeriodicAxes((frame.x,)),
    )

    with pytest.raises(NotImplementedError, match="partially periodic"):
        _native_amr_grid_values(partial.to_dict())


def test_native_amr_grid_and_patch_rectangles_preserve_a_shifted_origin() -> None:
    shifted_frame = Rectangle(
        "shifted_square", (-2.0, 3.0), (2.0, 7.0)
    ).frame(Cartesian2D())
    grid = CartesianGrid(frame=shifted_frame, cells=(8, 8))

    assert _native_amr_grid_values(grid.to_dict()) == (
        (8, 8), (-2.0, 3.0), (2.0, 7.0), False,
    )

    from pops.runtime._amr_system import AmrSystem

    class NativeProbe:
        @staticmethod
        def nx():
            return 8

        @staticmethod
        def patch_boxes():
            return [(1, 2, 4, 5, 7)]

    system = object.__new__(AmrSystem)
    system._s = NativeProbe()
    system._L = 4.0
    system._xlo = -2.0
    system._ylo = 3.0
    assert system.patch_rectangles() == [(-1.5, 4.0, 1.0, 1.0)]
