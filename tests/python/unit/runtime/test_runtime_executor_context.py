"""Native runtime resources fail closed before any System constructor is reachable."""

from __future__ import annotations

from dataclasses import replace
import sys
from types import SimpleNamespace

import pytest

from pops._platform_contracts import (
    CapabilityProof,
    ExecutionContext,
    ExecutionResource,
    proven_serial_manifest,
)
from pops.runtime import _multi_layout_executor as multi_executor
from pops.runtime import _runtime_executor as executor
from tests.python.unit.runtime.test_runtime_planning import _install


def _context(*, communicator="serial", datatype="float64", device="host"):
    backend = proven_serial_manifest(
        backend="production", target="system", abi="test|clang++|c++23", runtime=True
    )
    if communicator != "serial":
        backend = replace(
            backend,
            communicator=CapabilityProof.proven(communicator, "test.explicit-communicator"),
        )
    if device not in ("host", "cpu"):
        backend = replace(
            backend,
            device=CapabilityProof.proven(device, "test.explicit-device"),
        )
    return ExecutionContext(
        backend=backend,
        communicator=ExecutionResource(
            "communicator", communicator, None if communicator == "serial" else object()
        ),
        datatype=ExecutionResource("datatype", datatype),
        device=ExecutionResource("device", device, None if device in ("host", "cpu") else object()),
    )


@pytest.mark.parametrize(
    "context,match",
    [
        (_context(communicator="mpi"), "serial communicator"),
        (_context(datatype="float32"), "exact float64"),
        (_context(device="cuda:0"), "host/cpu"),
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
    "facts,match",
    [
        ({"mpi_active": True, "kokkos_backend": "Serial"}, "MPI inactive"),
        ({"mpi_active": False, "kokkos_backend": "Cuda"}, "Kokkos execution space"),
    ],
)
def test_process_global_native_state_is_rejected_before_system_constructor(
    monkeypatch, facts, match
):
    calls = []

    def forbidden_constructor(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("System constructor became reachable")

    monkeypatch.setattr(executor, "_native_runtime_facts", lambda: facts)
    monkeypatch.setitem(
        sys.modules, "pops.runtime._system", SimpleNamespace(System=forbidden_constructor)
    )
    with pytest.raises(NotImplementedError, match=match):
        executor.install_runtime_executor(_install())
    assert calls == []


def test_before_step_transfers_read_one_atomic_source_snapshot():
    first = SimpleNamespace(mapping_id="A-to-B", source="A", target="B")
    second = SimpleNamespace(mapping_id="B-to-C", source="B", target="C")
    states = {"A": 1, "B": 2, "C": 3}
    applied = []
    native = object.__new__(multi_executor._MultiLayoutUniformExecutor)
    native._runtime_plan = SimpleNamespace(communication=SimpleNamespace(transfers=(first, second)))
    native._engines = {}
    native._capture_mapping_source = lambda transfer: states[transfer.source]

    def apply(transfer, source):
        applied.append((transfer.mapping_id, source))
        states[transfer.target] = source

    native._apply_mapping = apply
    native._common_clock = lambda _method: 0

    native.step(0.125)

    assert applied == [("A-to-B", 1), ("B-to-C", 2)]
    assert states == {"A": 1, "B": 1, "C": 2}


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
    assert _uniform_system_values(square) == (16, 2.0, False)

    periodic = CartesianGrid(
        frame=square.frame,
        cells=(16, 16),
        periodic=PeriodicAxes(square.frame.axes),
    )
    assert _uniform_system_values(periodic) == (16, 2.0, True)

    partial = CartesianGrid(
        frame=square.frame,
        cells=(16, 16),
        periodic=PeriodicAxes((square.frame.x,)),
    )
    with pytest.raises(NotImplementedError, match="partially periodic"):
        _uniform_system_values(partial)

    rectangular_cells = CartesianGrid(frame=square.frame, cells=(16, 8))
    with pytest.raises(NotImplementedError, match="rectangular CartesianGrid"):
        _uniform_system_values(rectangular_cells)
    shifted = CartesianGrid(
        frame=Rectangle("shifted", (1.0, 0.0), (3.0, 2.0)).frame(Cartesian2D()),
        cells=(16, 16),
    )
    with pytest.raises(NotImplementedError, match="no origin"):
        _uniform_system_values(shifted)


def test_conservative_multi_layout_average_requires_one_physical_domain():
    from types import SimpleNamespace

    from pops.runtime._multi_layout_executor import (
        _require_conservative_cell_average_geometry,
    )

    fine = SimpleNamespace(n=16, L=1.0, periodic=True)
    coarse = SimpleNamespace(n=8, L=1.0, periodic=True)
    _require_conservative_cell_average_geometry(fine, coarse)

    with pytest.raises(ValueError, match="identical physical extents"):
        _require_conservative_cell_average_geometry(
            fine, SimpleNamespace(n=8, L=2.0, periodic=True)
        )
    with pytest.raises(ValueError, match="identical boundary topology"):
        _require_conservative_cell_average_geometry(
            fine, SimpleNamespace(n=8, L=1.0, periodic=False)
        )
