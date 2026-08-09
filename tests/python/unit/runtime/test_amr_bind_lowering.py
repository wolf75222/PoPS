"""AMR bind lowering preserves every authored Cartesian axis topology."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pops
import pytest

from pops.amr import AMRRegrid, PreparedHierarchyNativeLowering
from pops.domain import Rectangle
from pops.frames import Cartesian2D
from pops.mesh.grid import CartesianGrid, PeriodicAxes
from pops.runtime._amr_bind_lowering import (
    _install_native_hierarchy_config,
    _native_amr_grid_values,
    _physical_patch_bounds,
    _regrid_every,
)
from pops.runtime._amr_system_install import _AmrSystemInstall
from pops.runtime._runtime_authorities import (
    _materialized_shared_interface_levels,
    _validate_refined_shared_interface_execution,
)
from pops.time import Clock, every


def _frame():
    return Rectangle("unit_square", (0, 0), (1, 1)).frame(Cartesian2D())


def _native_amr_layout(grid: CartesianGrid, name: str):
    from pops.mesh import NativeSpatialLayout

    geometry = grid.normalized_geometry()
    spatial = grid.native_spatial_data()
    return NativeSpatialLayout(
        layout_id=name,
        coordinate_system=geometry.coordinate_system,
        cell_measure=geometry.cell_measure,
        axis_names=geometry.axis_names,
        shape=geometry.cells,
        lower=geometry.lower,
        upper=geometry.upper,
        periodicity=tuple(spatial["periodicity"]),
        centering="cell",
        decomposition={"kind": "adaptive"},
    )


def test_native_regrid_lowering_preserves_explicit_frozen_and_scheduled_policies() -> None:
    assert AMRRegrid.frozen().to_data() == {
        "schema_version": 1,
        "authority_type": "amr_regrid",
        "mode": "frozen",
    }
    assert _regrid_every({"regrid": AMRRegrid.frozen().to_data()}) == 0
    scheduled = AMRRegrid(schedule=every(3, clock=Clock("macro")))
    assert _regrid_every({"regrid": scheduled.to_data()}) == 3


@pytest.mark.parametrize("dimension", (1, 2, 3))
def test_native_config_installs_every_ranked_hierarchy_transition(dimension: int) -> None:
    ratios = (
        tuple(2 + axis for axis in range(dimension)),
        tuple(3 + axis for axis in range(dimension)),
    )
    buffers = (
        tuple(1 + axis for axis in range(dimension)),
        tuple(2 + axis for axis in range(dimension)),
    )
    lowering = PreparedHierarchyNativeLowering(
        {"provider": "test-ranked-hierarchy"},
        dimension,
        3,
        ratios,
        buffers,
        (1, 2),
    )
    config = SimpleNamespace()

    _install_native_hierarchy_config(config, lowering, dimension=dimension)

    assert config.level_count == 3
    assert config.transition_ratios == ratios
    assert config.transition_buffers == buffers
    assert config.transition_lookaheads == (1, 2)
    assert not hasattr(config, "regrid_margin")
    assert not hasattr(config, "regrid_grow")


def test_native_config_refuses_hierarchy_from_another_specialization() -> None:
    lowering = PreparedHierarchyNativeLowering(
        {"provider": "test-ranked-hierarchy"},
        3,
        2,
        ((2, 2, 2),),
        ((1, 2, 3),),
        (1,),
    )

    with pytest.raises(ValueError, match="config specialization"):
        _install_native_hierarchy_config(SimpleNamespace(), lowering, dimension=2)


def test_frozen_capacity_installs_exact_materialized_prefix() -> None:
    class NativeHierarchyProbe:
        @staticmethod
        def n_levels() -> int:
            return 3

    class ResolvedHierarchyProbe:
        level_count = 4

    assert _materialized_shared_interface_levels(
        NativeHierarchyProbe(), ResolvedHierarchyProbe()
    ) == (0, 1, 2)


def test_refined_shared_interface_bind_accepts_exact_mpi_world() -> None:
    mpi = {"communicator_identity": "MPI_COMM_WORLD"}
    _validate_refined_shared_interface_execution((0,), mpi, 2)
    _validate_refined_shared_interface_execution((0, 1), mpi, 1)
    _validate_refined_shared_interface_execution((0, 1), mpi, 2)
    _validate_refined_shared_interface_execution((0, 1, 2), mpi, 1)
    _validate_refined_shared_interface_execution((0, 1, 2), mpi, 2)


def test_dynamic_refined_shared_interface_bind_accepts_serial_and_exact_mpi_world() -> None:
    _validate_refined_shared_interface_execution(
        (0, 1, 2),
        {"communicator_identity": "serial"},
        1,
        dynamic_regrid=True,
    )
    _validate_refined_shared_interface_execution(
        (0, 1, 2),
        {"communicator_identity": "MPI_COMM_WORLD"},
        2,
        dynamic_regrid=True,
    )


def test_implicit_pair_requires_exact_frozen_two_level_prefix_at_complete_bind() -> None:
    serial = {
        "communicator_identity": "serial",
        "device_identity": "host",
        "memory_space": 1,
    }
    _validate_refined_shared_interface_execution(
        (0,), serial, 1, implicit_jacvec_pair=True, complete_bind=False
    )
    _validate_refined_shared_interface_execution(
        (0, 1), serial, 1, implicit_jacvec_pair=True, complete_bind=True
    )
    for levels in ((0,), (0, 1, 2)):
        with pytest.raises(NotImplementedError, match=r"exactly materialized levels \(L0, L1\)"):
            _validate_refined_shared_interface_execution(
                levels, serial, 1, implicit_jacvec_pair=True, complete_bind=True
            )


@pytest.mark.parametrize(
    ("execution", "ranks"),
    [
        (
            {
                "communicator_identity": "MPI_COMM_WORLD",
                "device_identity": "host",
                "memory_space": 1,
            },
            1,
        ),
        (
            {
                "communicator_identity": "MPI_COMM_WORLD",
                "device_identity": "host",
                "memory_space": 1,
            },
            2,
        ),
        (
            {
                "communicator_identity": "serial",
                "device_identity": "host",
                "memory_space": 1,
            },
            2,
        ),
    ],
)
def test_implicit_pair_refuses_mpi_before_native_interface_install(execution, ranks) -> None:
    with pytest.raises(NotImplementedError, match="currently serial-only"):
        _validate_refined_shared_interface_execution(
            (0, 1),
            execution,
            ranks,
            implicit_jacvec_pair=True,
            complete_bind=True,
        )


@pytest.mark.parametrize(
    "execution",
    [
        {
            "communicator_identity": "serial",
            "device_identity": "gpu",
            "memory_space": 2,
        },
        {
            "communicator_identity": "serial",
            "device_identity": "cpu",
            "memory_space": 3,
        },
    ],
)
def test_implicit_pair_refuses_device_or_managed_memory_before_native_install(
    execution,
) -> None:
    with pytest.raises(NotImplementedError, match="currently host-memory-only"):
        _validate_refined_shared_interface_execution(
            (0,),
            execution,
            1,
            implicit_jacvec_pair=True,
            complete_bind=False,
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
    with pytest.raises(TypeError, match="complete-bind contracts must be exact bools"):
        _validate_refined_shared_interface_execution(
            (0, 1),
            {"communicator_identity": "serial"},
            1,
            implicit_jacvec_pair=True,
            complete_bind=1,
        )
    with pytest.raises(TypeError, match="serial or exact MPI_COMM_WORLD"):
        _validate_refined_shared_interface_execution(
            (0, 1), {"communicator_identity": "MPI_COMM_SELF"}, 1
        )


def test_implicit_pair_envelope_precedes_program_and_interface_install(
    monkeypatch,
) -> None:
    import pops.runtime._amr_system_install as amr_install
    import pops.runtime._bound_snapshot as bound_snapshot
    import pops.runtime._component_execution_context as component_execution
    import pops.runtime._install_param_routing as param_routing
    import pops.runtime._lifecycle as lifecycle
    import pops.runtime._runtime_authorities as authorities

    events = []
    bind_schema = object()
    artifact = SimpleNamespace(
        bind_schema=bind_schema,
        so_path="compiled-amr-program.so",
        plan=SimpleNamespace(
            field_plans={},
            capabilities={
                "shared_interfaces": {"implicit_jacvec_pair": True},
            },
        ),
    )
    install_plan = SimpleNamespace(
        artifact=artifact,
        instances={},
        params={},
        aux={},
        bootstrap_plan=None,
        amr_transfer=None,
        execution_context=object(),
    )

    class Probe(_AmrSystemInstall):
        def __init__(self) -> None:
            self._s = SimpleNamespace()

        def _finish_program_install(self, *args, **kwargs) -> None:
            del args, kwargs
            events.append("program")

        def _finalize_bind(self, snapshot) -> None:
            assert snapshot == "snapshot"
            events.append("freeze")

    monkeypatch.setattr(lifecycle, "guard_assembling", lambda *_: None)
    monkeypatch.setattr(
        bound_snapshot,
        "_require_exact_install_inputs",
        lambda *_: install_plan,
    )
    monkeypatch.setattr(
        bound_snapshot,
        "build_amr_snapshot",
        lambda *args, **kwargs: "snapshot",
    )
    monkeypatch.setattr(
        amr_install,
        "validate_install_arguments",
        lambda *args, **kwargs: events.append("arguments"),
    )
    monkeypatch.setattr(
        component_execution,
        "component_execution_data",
        lambda _: {
            "communicator_identity": "serial",
            "device_identity": "host",
            "memory_space": 1,
        },
    )
    monkeypatch.setattr(param_routing, "route_block_params", lambda *args: {})
    native = SimpleNamespace(n_ranks=lambda: 1)
    monkeypatch.setitem(sys.modules, "pops._pops", native)
    monkeypatch.setattr(pops, "_pops", native, raising=False)
    validate_envelope = authorities._validate_shared_interface_implicit_execution_envelope

    def spy_envelope(execution_data, rank_count) -> None:
        events.append("implicit-envelope")
        validate_envelope(execution_data, rank_count)

    monkeypatch.setattr(
        authorities,
        "_validate_shared_interface_implicit_execution_envelope",
        spy_envelope,
    )
    monkeypatch.setattr(
        authorities,
        "finalize_runtime_authorities",
        lambda engine, plan, *, complete=False: events.append(
            "interfaces-complete" if complete else "interfaces-incremental"
        ),
    )

    Probe()._install_compiled(
        artifact,
        instances={},
        params={},
        aux={},
        field_plans={},
        bind_schema=bind_schema,
        initial_values=(),
        bootstrap_plan=None,
        amr_transfer=None,
        install_plan=install_plan,
    )

    assert events == [
        "arguments",
        "implicit-envelope",
        "program",
        "interfaces-incremental",
        "interfaces-complete",
        "freeze",
    ]


def test_native_amr_grid_preserves_none_or_all_periodic_axes() -> None:
    frame = _frame()
    closed = CartesianGrid(frame=frame, cells=(16, 16))
    periodic = CartesianGrid(
        frame=frame,
        cells=(16, 16),
        periodic=PeriodicAxes(frame.axes),
    )

    assert _native_amr_grid_values(_native_amr_layout(closed, "closed"))[-1] == (False, False)
    assert _native_amr_grid_values(_native_amr_layout(periodic, "periodic"))[-1] == (True, True)


def test_native_amr_grid_preserves_partial_periodicity() -> None:
    frame = _frame()
    partial = CartesianGrid(
        frame=frame,
        cells=(16, 16),
        periodic=PeriodicAxes((frame.x,)),
    )

    assert _native_amr_grid_values(_native_amr_layout(partial, "partial"))[-1] == (True, False)

    y_only = CartesianGrid(
        frame=frame,
        cells=(16, 16),
        periodic=PeriodicAxes((frame.y,)),
    )
    assert _native_amr_grid_values(_native_amr_layout(y_only, "y-only"))[-1] == (False, True)


def test_native_amr_grid_preserves_rectangular_cells_and_bounds() -> None:
    frame = Rectangle("rectangular", (-3.0, 2.0), (5.0, 5.0)).frame(Cartesian2D())
    grid = CartesianGrid(
        frame=frame,
        cells=(24, 10),
        periodic=PeriodicAxes((frame.y,)),
    )

    assert _native_amr_grid_values(_native_amr_layout(grid, "rectangular")) == (
        (24, 10),
        (-3.0, 2.0),
        (5.0, 5.0),
        (False, True),
    )


def test_native_amr_grid_and_patch_bounds_preserve_a_shifted_origin() -> None:
    shifted_frame = Rectangle("shifted_square", (-2.0, 3.0), (2.0, 7.0)).frame(Cartesian2D())
    grid = CartesianGrid(frame=shifted_frame, cells=(8, 8))

    assert _native_amr_grid_values(_native_amr_layout(grid, "shifted")) == (
        (8, 8),
        (-2.0, 3.0),
        (2.0, 7.0),
        (False, False),
    )

    assert _physical_patch_bounds(
        [(1, (2, 4), (5, 7))],
        cells=(8, 8),
        lengths=(4.0, 4.0),
        lower=(-2.0, 3.0),
    ) == [(-1.5, 4.0, 1.0, 1.0)]


def test_patch_bounds_use_independent_axis_spacing() -> None:
    assert _physical_patch_bounds(
        [(1, (2, 1), (5, 2))],
        cells=(12, 4),
        lengths=(6.0, 2.0),
        lower=(-1.0, 3.0),
    ) == [(-0.5, 3.25, 1.0, 0.5)]


@pytest.mark.parametrize(
    ("dimension", "shape", "lower", "upper", "periodicity"),
    (
        (1, (9,), (-1.0,), (2.0,), (True,)),
        (3, (4, 5, 6), (-1.0, 2.0, 4.0), (1.0, 5.0, 10.0), (True, False, True)),
    ),
)
def test_native_amr_grid_accepts_exact_ranked_cartesian_layouts(
    dimension, shape, lower, upper, periodicity
) -> None:
    from pops.mesh import NativeSpatialLayout

    layout = NativeSpatialLayout(
        layout_id="rank-%d" % dimension,
        coordinate_system="pops://coordinates/cartesian-%dd@1" % dimension,
        cell_measure={
            1: "pops://cell-measures/cartesian-length@1",
            3: "pops://cell-measures/cartesian-volume@1",
        }[dimension],
        axis_names=("x", "y", "z")[:dimension],
        shape=shape,
        lower=lower,
        upper=upper,
        periodicity=periodicity,
        centering="cell",
        decomposition={"kind": "adaptive"},
    )

    assert _native_amr_grid_values(layout) == (shape, lower, upper, periodicity)


def test_physical_patch_bounds_preserve_rank_three_axes() -> None:
    assert _physical_patch_bounds(
        [(2, (4, 2, 0), (7, 5, 3))],
        cells=(8, 4, 2),
        lengths=(4.0, 2.0, 8.0),
        lower=(-1.0, 3.0, 10.0),
    ) == [(-0.5, 3.25, 10.0, 0.5, 0.5, 4.0)]
