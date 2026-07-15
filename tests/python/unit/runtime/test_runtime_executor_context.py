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
from pops.runtime._component_execution_context import component_execution_data
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
        (_context(communicator="mpi"), "serial or exact MPI_COMM_WORLD"),
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
        ({"mpi_active": True, "kokkos_backend": "Serial"}, "MPI to be inactive"),
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


class _FakeComm:
    def __init__(self, *, rank=0, size=2, fortran_handle=0):
        self._rank = rank
        self._size = size
        self._fortran_handle = fortran_handle

    @staticmethod
    def Compare(left, right):
        return 0 if left is right else 1

    def Get_rank(self):
        return self._rank

    def Get_size(self):
        return self._size

    def py2f(self):
        return self._fortran_handle


class _FakeDatatype:
    def __init__(self, fortran_handle=0):
        self._fortran_handle = fortran_handle

    def py2f(self):
        return self._fortran_handle


def _exact_mpi_context(monkeypatch):
    from pops.codegen._compiled_artifact import CompiledSimulationArtifact
    from pops.runtime import _platform_manifest

    world = _FakeComm()
    double = _FakeDatatype()
    mpi = SimpleNamespace(
        Comm=_FakeComm,
        Datatype=_FakeDatatype,
        COMM_WORLD=world,
        DOUBLE=double,
        IDENT=0,
    )
    monkeypatch.setitem(sys.modules, "mpi4py", SimpleNamespace(MPI=mpi))
    backend = replace(
        proven_serial_manifest(
            backend="production", target="system", abi="test|clang++|c++23", runtime=True
        ),
        communicator=CapabilityProof.proven(
            "MPI_COMM_WORLD", "test.explicit-mpi-world"
        ),
    )
    artifact = object.__new__(CompiledSimulationArtifact)
    object.__setattr__(
        artifact,
        "platform_manifest",
        proven_serial_manifest(
            backend="production", target="system", abi="test|clang++|c++23"
        ),
    )
    monkeypatch.setattr(_platform_manifest, "native_runtime_backend", lambda _platform: backend)
    return ExecutionContext.mpi_world(artifact, world), mpi


def test_exact_mpi_world_context_projects_zero_valued_fortran_handles(monkeypatch):
    context, mpi = _exact_mpi_context(monkeypatch)

    assert context.communicator.handle is mpi.COMM_WORLD
    assert context.datatype.handle is mpi.DOUBLE
    assert component_execution_data(context) == {
        "execution_identity": context.identity.token,
        "context_version": 1,
        "memory_space": 1,
        "backend_identity": context.backend.identity.token,
        "device_identity": "host",
        "scalar_type": 2,
        "storage_precision": 4,
        "compute_precision": 4,
        "accumulation_precision": 4,
        "reduction_precision": 4,
        "stream_handle": 0,
        "stream_identity": "host::synchronous",
        "communicator_f_handle": 0,
        "communicator_datatype_f_handle": 0,
        "communicator_identity": "MPI_COMM_WORLD",
        "communicator_datatype_identity": "MPI_DOUBLE",
    }
    monkeypatch.setattr(
        executor,
        "_native_runtime_facts",
        lambda: {
            "mpi_compiled": True,
            "mpi_active": True,
            "communicator": "MPI_COMM_WORLD",
            "mpi_rank": 0,
            "mpi_ranks": 2,
            "kokkos_backend": "Serial",
        },
    )
    executor._require_supported_execution_context(
        SimpleNamespace(execution_context=context)
    )


def test_mpi_world_factory_rejects_equal_looking_or_duplicated_communicators(monkeypatch):
    context, mpi = _exact_mpi_context(monkeypatch)

    class _EqualLooking:
        def __eq__(self, _other):
            return True

    from pops.codegen._compiled_artifact import CompiledSimulationArtifact

    artifact = object.__new__(CompiledSimulationArtifact)
    object.__setattr__(artifact, "platform_manifest", context.backend)
    with pytest.raises(ValueError, match="only mpi4py.MPI.COMM_WORLD"):
        ExecutionContext.mpi_world(artifact, _EqualLooking())
    with pytest.raises(ValueError, match="only mpi4py.MPI.COMM_WORLD"):
        ExecutionContext.mpi_world(artifact, _FakeComm())


def test_mpi_component_projection_rejects_noncanonical_double(monkeypatch):
    context, _mpi = _exact_mpi_context(monkeypatch)
    changed = replace(
        context,
        datatype=ExecutionResource("datatype", "float64", handle=_FakeDatatype()),
    )
    with pytest.raises(ValueError, match="exact mpi4py.MPI.DOUBLE"):
        component_execution_data(changed)


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
