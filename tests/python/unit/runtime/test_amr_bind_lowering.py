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
