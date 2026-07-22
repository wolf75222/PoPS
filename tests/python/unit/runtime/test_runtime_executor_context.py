"""Native runtime resources fail closed before any System constructor is reachable."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from pops import _platform_contracts as platform_contracts
from pops._platform_contracts import (
    ExecutionContext,
    ExecutionResource,
    proven_serial_manifest,
)
from pops.runtime import _multi_layout_executor as multi_executor
from pops.runtime import _platform_manifest as platform_manifest
from pops.runtime import _runtime_executor as executor
from tests.python.unit.runtime.test_runtime_planning import _install


def _context(*, datatype="float64"):
    backend = proven_serial_manifest(
        backend="production", target="system", abi="test|clang++|c++23", runtime=True
    )
    return ExecutionContext(
        backend=backend,
        communicator=ExecutionResource("communicator", "serial"),
        datatype=ExecutionResource("datatype", datatype),
        device=ExecutionResource("device", "host"),
    )


@pytest.mark.parametrize(
    "context,match",
    [
        (_context(datatype="float32"), "exact float64"),
    ],
)
def test_unsupported_execution_context_resources_are_rejected(context, match):
    with pytest.raises(NotImplementedError, match=match):
        executor._require_supported_execution_context(
            type(
                "Plan",
                (),
                {
                    "execution_context": context,
                },
            )()
        )


@pytest.mark.parametrize(
    "facts,error,match",
    [
        (
            {
                "mpi_active": True,
                "kokkos_backend": "host",
                "kokkos_device": "host",
                "field_memory_space": "host",
                "kokkos_shared_space": "HostSpace",
                "kokkos_stream": "host::synchronous",
            },
            NotImplementedError,
            "MPI to be inactive",
        ),
        (
            {
                "mpi_active": False,
                "kokkos_backend": "Cuda",
                "kokkos_device": "cuda",
                "field_memory_space": "managed",
                "kokkos_shared_space": "CudaUVMSpace",
                "kokkos_stream": "cuda::default",
            },
            ValueError,
            "device differs",
        ),
    ],
)
def test_mismatched_native_state_is_rejected_before_system_constructor(
    monkeypatch, facts, error, match
):
    calls = []

    def forbidden_constructor(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("System constructor became reachable")

    monkeypatch.setattr(executor, "_native_runtime_facts", lambda: facts)
    # This unit isolates the executor's native-state preflight.  The planning layer is covered
    # separately and an installed MPI/OpenMP wheel must not make this serial fixture fail before the
    # executor is reached.
    monkeypatch.setattr(platform_contracts, "validate_launch", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        platform_contracts, "validate_component_launch", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        platform_manifest, "validate_native_device_resource", lambda *args, **kwargs: None
    )
    monkeypatch.setitem(
        sys.modules, "pops.runtime._system", SimpleNamespace(System=forbidden_constructor)
    )
    with pytest.raises(error, match=match):
        executor.install_runtime_executor(_install())
    assert calls == []




def test_before_step_transfer_cycle_captures_every_native_source_before_any_apply():
    first = SimpleNamespace(mapping_id="A-to-B", source="A", target="B")
    second = SimpleNamespace(mapping_id="B-to-A", source="B", target="A")
    states = {"A": 1, "B": 2}
    events = []

    class Session:
        def __init__(self, transfer):
            self.transfer = transfer
            self.snapshot = None

        def capture(self, generation, attempt):
            events.append(("capture", self.transfer.mapping_id, generation, attempt))
            self.snapshot = states[self.transfer.source]

        def apply(self, generation, attempt):
            events.append(("apply", self.transfer.mapping_id, generation, attempt))
            states[self.transfer.target] = self.snapshot
            return object()

    class NativeEngine:
        def time(self):
            return 0.0

        def macro_step(self):
            return 0

        def step(self, _dt):
            return None

    native = object.__new__(multi_executor._MultiLayoutUniformExecutor)
    native._active_transfer_generation = 7
    native._transfer_attempt = 0
    native._transfer_routes = tuple(
        multi_executor._NativeTransferRoute(
            transfer=row,
            source_block=row.source,
            target_block=row.target,
            session=Session(row),
            source_element_count=1,
            destination_element_count=1,
        )
        for row in (first, second)
    )
    native._engines = {"A": NativeEngine(), "B": NativeEngine()}
    native._mapping_evaluations = {"A-to-B": 0, "B-to-A": 0}
    native._authenticate_mapping_receipt = lambda *args, **kwargs: None
    native._last_mapping_receipts = ()
    native._common_clock = lambda _method: 0

    native.step(0.125)

    assert events == [
        ("capture", "A-to-B", 7, 1),
        ("capture", "B-to-A", 7, 1),
        ("apply", "A-to-B", 7, 1),
        ("apply", "B-to-A", 7, 1),
    ]
    assert states == {"A": 2, "B": 1}
    assert native.mapping_report() == {"A-to-B": 1, "B-to-A": 1}


def test_rejected_multi_layout_attempt_restores_every_child_then_recaptures():
    from pops._bootstrap import StepAttemptRejected

    states = {"A": 1, "B": 2}
    events = []
    first = SimpleNamespace(mapping_id="A-to-B", source="A", target="B")
    second = SimpleNamespace(mapping_id="B-to-A", source="B", target="A")

    class Session:
        def __init__(self, transfer):
            self.transfer = transfer
            self.snapshot = None

        def begin_transaction(self, generation):
            events.append(("begin", self.transfer.mapping_id, generation))

        def capture(self, generation, attempt):
            events.append(("capture", self.transfer.mapping_id, attempt))
            self.snapshot = states[self.transfer.source]

        def apply(self, generation, attempt):
            events.append(("apply", self.transfer.mapping_id, attempt))
            states[self.transfer.target] = self.snapshot
            return object()

        def reject_attempt(self, generation, attempt):
            events.append(("reject", self.transfer.mapping_id, attempt))
            self.snapshot = None

        def rollback_transaction(self, generation):
            events.append(("rollback", self.transfer.mapping_id, generation))

    class Engine:
        def __init__(self, name, reject_once=False):
            self.name = name
            self.reject_once = reject_once
            self.snapshot = None
            self.clock = 0

        def _begin_step_transaction(self):
            self.snapshot = (states[self.name], self.clock)

        def _rollback_step_transaction(self):
            states[self.name], self.clock = self.snapshot
            self.snapshot = None

        def time(self):
            return float(self.clock)

        def macro_step(self):
            return self.clock

        def step(self, _dt):
            if self.reject_once:
                self.reject_once = False
                raise StepAttemptRejected("injected native rejection")
            states[self.name] += 10
            self.clock += 1

    native = object.__new__(multi_executor._MultiLayoutUniformExecutor)
    native._engines = {"A": Engine("A"), "B": Engine("B", reject_once=True)}
    native._transfer_routes = tuple(
        multi_executor._NativeTransferRoute(
            transfer=row,
            source_block=row.source,
            target_block=row.target,
            session=Session(row),
            source_element_count=1,
            destination_element_count=1,
        )
        for row in (first, second)
    )
    native._mapping_evaluations = {"A-to-B": 0, "B-to-A": 0}
    native._mapping_snapshot = None
    native._transfer_generation = 0
    native._active_transfer_generation = None
    native._transfer_attempt = 0
    native._last_mapping_receipts = ()
    native._authenticate_mapping_receipt = lambda *args, **kwargs: None
    native._synchronize_child_temporal_states = lambda: None

    native._begin_step_transaction()
    with pytest.raises(StepAttemptRejected, match="injected native rejection"):
        native.step(0.125)
    assert states == {"A": 1, "B": 2}
    native.step(0.0625)
    assert states == {"A": 12, "B": 11}
    assert native.mapping_report() == {"A-to-B": 1, "B-to-A": 1}
    assert [event for event in events if event[0] == "capture"] == [
        ("capture", "A-to-B", 1),
        ("capture", "B-to-A", 1),
        ("capture", "A-to-B", 2),
        ("capture", "B-to-A", 2),
    ]
    native._rollback_step_transaction()
    assert states == {"A": 1, "B": 2}
    assert native.mapping_report() == {"A-to-B": 0, "B-to-A": 0}


def test_runtime_install_rejects_concurrent_overwrite_transfer_targets():
    common = {
        "operation_abi": 1,
        "target_layout_id": "layout-C",
        "target_subject_id": "state-C",
        "synchronization_uri": "pops://synchronization/before-step@1",
    }
    transfers = (
        SimpleNamespace(mapping_id="A-to-C", **common),
        SimpleNamespace(mapping_id="B-to-C", **common),
    )

    with pytest.raises(ValueError, match="explicit merge protocol"):
        multi_executor._require_unique_transfer_targets(transfers)


def test_cartesian_grid_lowering_is_exact_and_refuses_unrepresentable_geometry():
    from pops.domain import Rectangle
    from pops.frames import Cartesian2D
    from pops.mesh import CartesianGrid, PeriodicAxes
    from pops.runtime._runtime_mesh_lowering import _uniform_system_values

    with pytest.raises(NotImplementedError, match="exact pops.mesh.CartesianGrid"):
        _uniform_system_values(SimpleNamespace(n=16, L=2.0, periodic=False))

    square = CartesianGrid(
        frame=Rectangle("square", (0.0, 0.0), (2.0, 2.0)).frame(Cartesian2D()),
        cells=(16, 16),
    )
    assert _uniform_system_values(square) == (16, 2.0, (False, False), 0.0, 0.0)

    periodic = CartesianGrid(
        frame=square.frame,
        cells=(16, 16),
        periodic=PeriodicAxes(square.frame.axes),
    )
    assert _uniform_system_values(periodic) == (16, 2.0, (True, True), 0.0, 0.0)

    partial = CartesianGrid(
        frame=square.frame,
        cells=(16, 16),
        periodic=PeriodicAxes((square.frame.x,)),
    )
    assert _uniform_system_values(partial) == (16, 2.0, (True, False), 0.0, 0.0)

    rectangular_cells = CartesianGrid(frame=square.frame, cells=(16, 8))
    with pytest.raises(NotImplementedError, match="rectangular CartesianGrid"):
        _uniform_system_values(rectangular_cells)
    shifted = CartesianGrid(
        frame=Rectangle("shifted", (1.0, 0.0), (3.0, 2.0)).frame(Cartesian2D()),
        cells=(16, 16),
    )
    assert _uniform_system_values(shifted) == (
        16, 2.0, (False, False), 1.0, 0.0,
    )


def test_conservative_multi_layout_average_requires_one_physical_domain():
    from types import SimpleNamespace

    from pops.runtime._multi_layout_executor import (
        _require_conservative_cell_average_geometry,
    )

    fine = SimpleNamespace(n=16, L=1.0, periodicity=(True, False), xlo=-2.0, ylo=3.0)
    coarse = SimpleNamespace(n=8, L=1.0, periodicity=(True, False), xlo=-2.0, ylo=3.0)
    _require_conservative_cell_average_geometry(fine, coarse)

    with pytest.raises(ValueError, match="identical physical extents"):
        _require_conservative_cell_average_geometry(
            fine,
            SimpleNamespace(n=8, L=2.0, periodicity=(True, False), xlo=-2.0, ylo=3.0),
        )
    with pytest.raises(ValueError, match="identical physical origins"):
        _require_conservative_cell_average_geometry(
            fine,
            SimpleNamespace(n=8, L=1.0, periodicity=(True, False), xlo=0.0, ylo=3.0),
        )
    with pytest.raises(ValueError, match="identical boundary topology"):
        _require_conservative_cell_average_geometry(
            fine,
            SimpleNamespace(n=8, L=1.0, periodicity=(False, True), xlo=-2.0, ylo=3.0),
        )
    with pytest.raises(TypeError, match=r"exact \(x, y\) periodicity tuple"):
        _require_conservative_cell_average_geometry(
            fine,
            SimpleNamespace(n=8, L=1.0, periodicity=True, xlo=-2.0, ylo=3.0),
        )
