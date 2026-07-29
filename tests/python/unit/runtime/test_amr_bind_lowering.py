"""AMR bind lowering preserves every authored Cartesian axis topology."""
from __future__ import annotations

import pytest

from pops.amr import AMRRegrid
from pops.domain import Rectangle
from pops.frames import Cartesian2D
from pops.mesh.grid import CartesianGrid, PeriodicAxes
from pops.runtime._amr_bind_lowering import (
    _native_amr_grid_values,
    _physical_patch_rectangles,
    _regrid_every,
)
from pops.runtime._runtime_authorities import (
    _materialized_shared_interface_levels,
    _validate_refined_shared_interface_execution,
)
from pops.time import Clock, every


def _frame():
    return Rectangle("unit_square", (0, 0), (1, 1)).frame(Cartesian2D())


def test_native_regrid_lowering_preserves_explicit_frozen_and_scheduled_policies() -> None:
    assert AMRRegrid.frozen().to_data() == {
        "schema_version": 1,
        "authority_type": "amr_regrid",
        "mode": "frozen",
    }
    assert _regrid_every({"regrid": AMRRegrid.frozen().to_data()}) == 0
    scheduled = AMRRegrid(schedule=every(3, clock=Clock("macro")))
    assert _regrid_every({"regrid": scheduled.to_data()}) == 3


def test_frozen_two_level_capacity_installs_only_the_materialized_coarse_level() -> None:
    class NativeHierarchyProbe:
        @staticmethod
        def n_levels() -> int:
            return 1

    class ResolvedHierarchyProbe:
        level_count = 2

    assert _materialized_shared_interface_levels(
        NativeHierarchyProbe(), ResolvedHierarchyProbe()) == (0,)


def test_refined_shared_interface_bind_accepts_exact_mpi_world() -> None:
    mpi = {"communicator_identity": "MPI_COMM_WORLD"}
    _validate_refined_shared_interface_execution((0,), mpi, 2)
    _validate_refined_shared_interface_execution((0, 1), mpi, 1)
    _validate_refined_shared_interface_execution((0, 1), mpi, 2)


def test_dynamic_refined_shared_interface_bind_remains_serial() -> None:
    with pytest.raises(NotImplementedError, match="rematerialization"):
        _validate_refined_shared_interface_execution(
            (0, 1),
            {"communicator_identity": "MPI_COMM_WORLD"},
            2,
            dynamic_regrid=True,
        )


def test_shared_interface_bind_rejects_non_prefix_and_unknown_communicator() -> None:
    with pytest.raises(ValueError, match="contiguous L0 prefix"):
        _validate_refined_shared_interface_execution((), {}, 1)
    with pytest.raises(ValueError, match="contiguous L0 prefix"):
        _validate_refined_shared_interface_execution((0, 2), {}, 1)
    with pytest.raises(TypeError, match="exact bool"):
        _validate_refined_shared_interface_execution(
            (0, 1), {"communicator_identity": "serial"}, 1, dynamic_regrid=1
        )
    with pytest.raises(TypeError, match="serial or exact MPI_COMM_WORLD"):
        _validate_refined_shared_interface_execution(
            (0, 1), {"communicator_identity": "MPI_COMM_SELF"}, 1)


def test_native_amr_grid_preserves_none_or_all_periodic_axes() -> None:
    frame = _frame()
    closed = CartesianGrid(frame=frame, cells=(16, 16))
    periodic = CartesianGrid(
        frame=frame,
        cells=(16, 16),
        periodic=PeriodicAxes(frame.axes),
    )

    assert _native_amr_grid_values(closed.to_dict())[-1] == (False, False)
    assert _native_amr_grid_values(periodic.to_dict())[-1] == (True, True)


def test_native_amr_grid_preserves_partial_periodicity() -> None:
    frame = _frame()
    partial = CartesianGrid(
        frame=frame,
        cells=(16, 16),
        periodic=PeriodicAxes((frame.x,)),
    )

    assert _native_amr_grid_values(partial.to_dict())[-1] == (True, False)

    y_only = CartesianGrid(
        frame=frame,
        cells=(16, 16),
        periodic=PeriodicAxes((frame.y,)),
    )
    assert _native_amr_grid_values(y_only.to_dict())[-1] == (False, True)


def test_native_amr_grid_preserves_rectangular_cells_and_bounds() -> None:
    frame = Rectangle(
        "rectangular", (-3.0, 2.0), (5.0, 5.0)
    ).frame(Cartesian2D())
    grid = CartesianGrid(
        frame=frame,
        cells=(24, 10),
        periodic=PeriodicAxes((frame.y,)),
    )

    assert _native_amr_grid_values(grid.to_dict()) == (
        (24, 10), (-3.0, 2.0), (5.0, 5.0), (False, True),
    )


def test_native_amr_grid_and_patch_rectangles_preserve_a_shifted_origin() -> None:
    shifted_frame = Rectangle(
        "shifted_square", (-2.0, 3.0), (2.0, 7.0)
    ).frame(Cartesian2D())
    grid = CartesianGrid(frame=shifted_frame, cells=(8, 8))

    assert _native_amr_grid_values(grid.to_dict()) == (
        (8, 8), (-2.0, 3.0), (2.0, 7.0), (False, False),
    )

    assert _physical_patch_rectangles(
        [(1, 2, 4, 5, 7)],
        cells=(8, 8),
        lengths=(4.0, 4.0),
        lower=(-2.0, 3.0),
    ) == [(-1.5, 4.0, 1.0, 1.0)]


def test_patch_rectangles_use_independent_axis_spacing() -> None:
    assert _physical_patch_rectangles(
        [(1, 2, 1, 5, 2)],
        cells=(12, 4),
        lengths=(6.0, 2.0),
        lower=(-1.0, 3.0),
    ) == [(-0.5, 3.25, 1.0, 0.5)]
