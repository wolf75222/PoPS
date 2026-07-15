"""Collective HDF5 proofs at both the writer and accepted-runtime boundaries.

The low-level proof writes disjoint ``OutputSnapshot`` hyperslabs and compares the reopened result
with a serial publication.  The production-path proof starts from an accepted uniform runtime's
rank-owned local boxes and traverses consumer planning, ``RuntimeConsumerPublisher`` and the exact
HDF5 provider.  Missing optional MPI/HDF5 support is an explicit skip, never degraded execution.
"""

from __future__ import annotations

from dataclasses import replace
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest

from pops._platform_contracts import (
    CapabilityProof,
    ExecutionContext,
    ExecutionResource,
    proven_serial_manifest,
)
from pops.codegen._compiled_artifact import (
    CompiledBlockArtifact,
    CompiledSimulationArtifact,
)
from pops.codegen._plans import BindInputs, InstallPlan
from pops.identity import make_identity
from pops.model import ComponentManifest, Handle, OwnerKind, OwnerPath
from pops.output import (
    ArrayPiece,
    FieldKey,
    FieldPayload,
    HDF5,
    HDF5Writer,
    LevelGeometry,
    OutputClock,
    OutputProvenance,
    OutputRequest,
    OutputSnapshot,
    read_hdf5,
)
from pops.output._consumer_contracts import (
    ConsumerGraph,
    ConsumerKind,
    ConsumerManifest,
    ConsumerQuantity,
    ParallelMode,
)
from pops.runtime._runtime_consumers import RuntimeConsumerPublisher
from pops.runtime._runtime_instance import RuntimeInstance
from pops.runtime._temporal_restart import TemporalRestartState
from pops.time import AcceptedStep, Clock, Every, Schedule
from tests.python.unit.codegen._typed_artifact_fixture import CompiledComponent
from tests.python.unit.runtime.test_runtime_planning import _install


pytestmark = pytest.mark.mpi
_MPI_CHILD = "POPS_HDF5_MPI_TEST_CHILD"


def _identity(domain: str, name: str):
    return make_identity(domain, {"name": name})


def _snapshots(communicator):
    rank = int(communicator.Get_rank())
    size = int(communicator.Get_size())
    rows_per_rank, nx = 2, 5
    ny = rows_per_rank * size
    global_values = np.arange(ny * nx, dtype=np.float64).reshape(ny, nx) + np.float64(0.125)
    lower = (rank * rows_per_rank, 0)
    upper = ((rank + 1) * rows_per_rank, nx)
    local_piece = ArrayPiece(lower, upper, global_values[lower[0] : upper[0], :])
    all_pieces = tuple(communicator.allgather(local_piece))

    layout = _identity("layout-plan", "collective-hdf5-layout")
    manifest = _identity("component-manifest", "collective-hdf5-model")
    owner = OwnerPath.case("collective-hdf5-case").child(OwnerKind.BLOCK, "fluid")
    state = Handle("rho", kind="state", owner=owner)
    key = FieldKey(state, manifest, layout, 0, "accepted")
    geometry = LevelGeometry(
        layout,
        "uniform",
        0,
        (0.0, 0.0),
        (1.0 / ny, 1.0 / nx),
        (ny, nx),
        ((0, 0, ny, nx),),
        np.zeros((ny, nx), dtype=np.bool_),
        np.full((ny, nx), 1.0 / (ny * nx), dtype=np.float64),
    )
    clock = OutputClock.at("macro", 0.25, 3, stage="accepted")
    provenance = OutputProvenance(
        _identity("resolved-plan", "collective-hdf5-plan"),
        _identity("bind", "collective-hdf5-bind"),
        _identity("run", "collective-hdf5-run"),
        "accepted-step-transaction",
    )

    def snapshot(pieces):
        field = FieldPayload(key, "cell", "kg.m-3", (), (ny, nx), tuple(pieces))
        return OutputSnapshot(
            clock,
            provenance,
            (geometry,),
            (field,),
            {"case": "collective-hdf5", "ranks": size},
        )

    return snapshot((local_piece,)), snapshot(all_pieces), key, global_values


class _AcceptedUniformMPIState:
    """Source-only accepted state exposing the production uniform local-box protocol."""

    def __init__(self, plan: InstallPlan, communicator, clock: Clock) -> None:
        self._s = self
        geometry = plan.artifact.layout_plan.layouts[0].geometry
        self._nx, self._ny = geometry.cells
        self._rank = int(communicator.Get_rank())
        self._size = int(communicator.Get_size())
        if self._size > self._ny:
            raise ValueError("the source-only MPI fixture requires at least one row per rank")
        rows, extra = divmod(self._ny, self._size)
        self._jlo = self._rank * rows + min(self._rank, extra)
        self._jhi = self._jlo + rows + int(self._rank < extra)
        self._time, self._step = 1.0, 1
        self._last_run_identity = None
        self._temporal_restart_state = TemporalRestartState()
        self._temporal_restart_state.configure_program(
            {
                "schema_version": 1,
                "kind": "pops.temporal-program-schedule",
                "primary_clock": clock.qualified_id,
                "clocks": [
                    {
                        "id": clock.qualified_id,
                        "descriptor": clock.to_data(),
                        "ticks_per_macro": 1,
                    }
                ],
                "subcycles": [],
                "synchronizations": [],
                "schedules": [],
                "histories": [],
            },
            time=0.0,
            macro_step=0,
        )
        self._temporal_restart_state.accept(
            before_time=0.0,
            before_step=0,
            time=self._time,
            macro_step=self._step,
        )

    def time(self) -> float:
        return self._time

    def macro_step(self) -> int:
        return self._step

    def nx(self) -> int:
        return self._nx

    def ny(self) -> int:
        return self._ny

    def local_boxes(self, block: str):
        assert block == "fluid"
        return ((0, self._jlo, self._nx - 1, self._jhi - 1),)

    def local_state(self, block: str, index: int):
        assert block == "fluid" and index == 0
        values = np.arange(self._ny * self._nx, dtype=np.float64).reshape(
            self._ny, self._nx
        ) + np.float64(1.125)
        return values[None, self._jlo : self._jhi, :]


class _MPICompiledComponent(CompiledComponent):
    """Source-only compiled-component evidence for the selected MPI execution context."""

    def __init__(self, name: str, communicator_id: str) -> None:
        super().__init__(name, target="system")
        self.communicator = communicator_id

    def __pops_artifact_model_metadata__(self):
        data = super().__pops_artifact_model_metadata__()
        return {**data, "capabilities": {**data["capabilities"], "mpi": True}}


def _runtime_with_collective_hdf5(output: Path, communicator):
    """Build the exact runtime-consumer route without importing the native extension."""
    base = _install()
    layout = base.artifact.layout_plan.layouts[0].handle
    clock = Clock("solution", owner=OwnerPath.consumer("runtime-hdf5-mpi"))
    quantity = ConsumerQuantity(
        Handle("rho", kind="state", owner=OwnerPath.model("runtime-hdf5-mpi")),
        "state:u",
        layout.qualified_id,
    )
    consumer = ConsumerManifest(
        Handle("density", kind="consumer", owner=OwnerPath.consumer("runtime-hdf5-mpi")),
        ConsumerKind.SCIENTIFIC_OUTPUT,
        (quantity,),
        Schedule(Every(AcceptedStep(clock), 1)),
        str(output),
        HDF5(parallel=True),
        ParallelMode.COLLECTIVE,
    )
    graph = ConsumerGraph((consumer,))
    record = replace(base.artifact.plan, consumer_graph=graph)
    communicator_id = "pops-test-mpi-world-%d" % int(communicator.Get_size())

    block_component = _MPICompiledComponent("fluid", communicator_id)
    program = _MPICompiledComponent("program", communicator_id)
    program.program_block_routes = ((0, "fluid"),)
    block = CompiledBlockArtifact(
        "fluid",
        block_component,
        base.artifact.blocks[0].spatial,
        base.artifact.blocks[0].state_spaces,
    )
    artifact = CompiledSimulationArtifact(record, program, (block,))
    backend = proven_serial_manifest(
        backend=artifact.platform_manifest.backend.require("test artifact backend"),
        target=artifact.platform_manifest.target.require("test artifact target"),
        abi=artifact.platform_manifest.abi.require("test artifact ABI"),
        runtime=True,
    )
    backend = replace(
        backend,
        communicator=artifact.platform_manifest.communicator,
        capabilities={
            **backend.capabilities,
            "rank_count": CapabilityProof.proven(
                int(communicator.Get_size()), "pops.test.runtime-hdf5-mpi.v1"
            ),
        },
    )
    context = ExecutionContext(
        backend,
        ExecutionResource("communicator", communicator_id, handle=communicator),
        ExecutionResource("datatype", "float64"),
        ExecutionResource("device", "host"),
    )
    inputs = BindInputs()
    plan = InstallPlan(
        artifact,
        inputs,
        {"fluid": {"model": block.model, "spatial": block.spatial}},
        artifact.bind_schema.resolve_bind({}, compile_values=artifact.plan.compile_values),
        {},
        execution_context=context,
    )
    component_manifest = ComponentManifest(
        uri="pops://tests/runtime-hdf5-mpi/fluid",
        component_type="compiled_spatial_operator",
        version="1.0.0",
        writes=({"resource": "state:u"},),
        requirements=(
            {
                "capability": "collective",
                "resource": "state:u",
                "operation": "gather",
                "strategy": "ordered_tree",
            },
        ),
        effects=({"kind": "state_write", "resource": "state:u"},),
        clocks=({"clock": "solution", "access": "stage"},),
        target={
            "variants": [
                {
                    "dimension": 2,
                    "scalar": "float64",
                    "device": "host",
                    "features": [],
                }
            ],
        },
        determinism={"classification": "reproducible", "scope": ["rank_count"]},
        precision={
            "inputs": ["float64"],
            "accumulation": "float64",
            "outputs": ["float64"],
        },
        entry_points={"step": "pops_runtime_step"},
    )
    executor = _AcceptedUniformMPIState(plan, communicator, clock)
    runtime = RuntimeInstance(
        plan,
        executor=executor,
        component_manifests={"fluid": component_manifest},
    )
    return runtime, consumer


def _parallel_hdf5_world(test_name: str):
    h5py = pytest.importorskip("h5py", reason="collective HDF5 requires h5py")
    if not h5py.get_config().mpi:
        pytest.skip("collective HDF5 requires h5py built with MPI")
    mpi = pytest.importorskip("mpi4py.MPI", reason="collective HDF5 requires mpi4py")
    communicator = mpi.COMM_WORLD
    if int(communicator.Get_size()) == 1 and os.environ.get(_MPI_CHILD) != "1":
        mpiexec = shutil.which("mpiexec") or shutil.which("mpirun")
        if mpiexec is None:
            pytest.skip("collective HDF5 requires an MPI launcher for the two-rank proof")
        env = dict(os.environ)
        env[_MPI_CHILD] = "1"
        result = subprocess.run(
            [
                mpiexec,
                "-n",
                "2",
                sys.executable,
                "-m",
                "pytest",
                "-q",
                f"{__file__}::{test_name}",
            ],
            cwd=Path(__file__).resolve().parents[4],
            env=env,
            text=True,
            capture_output=True,
            timeout=120,
            check=False,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        return None
    assert int(communicator.Get_size()) >= 2, "MPI child did not start with two ranks"
    return communicator


def test_collective_hdf5_roundtrip_matches_serial(tmp_path):
    communicator = _parallel_hdf5_world(test_collective_hdf5_roundtrip_matches_serial.__name__)
    if communicator is None:
        return
    rank = int(communicator.Get_rank())

    local_snapshot, serial_snapshot, key, expected = _snapshots(communicator)
    shared_root = communicator.bcast(str(tmp_path) if rank == 0 else None, root=0)
    collective_target = Path(shared_root) / "collective.h5"
    serial_target = Path(shared_root) / "serial.h5"
    collective_request = OutputRequest("density-output", (key,), True)
    serial_request = OutputRequest("density-output", (key,), False)

    prepared = HDF5Writer().prepare(
        local_snapshot,
        collective_request,
        collective_target,
        communicator=communicator,
    )
    collective_receipt = prepared.publish()

    failure = None
    if rank == 0:
        try:
            serial_receipt = (
                HDF5Writer().prepare(serial_snapshot, serial_request, serial_target).publish()
            )
            collective = read_hdf5(collective_receipt.path).require_selection(collective_request)
            serial = read_hdf5(serial_receipt.path).require_selection(serial_request)
            collective_name = collective.manifest["datasets"]["fields"][key.identity.token]
            serial_name = serial.manifest["datasets"]["fields"][key.identity.token]

            np.testing.assert_array_equal(collective.arrays[collective_name], expected)
            np.testing.assert_array_equal(
                collective.arrays[collective_name], serial.arrays[serial_name]
            )
            assert collective.output_identity == collective_receipt.output_identity
            assert serial.output_identity == serial_receipt.output_identity
            for section in (
                "clock",
                "provenance",
                "geometries",
                "fields",
                "diagnostics",
                "metadata",
            ):
                assert (
                    collective.manifest["snapshot"][section] == serial.manifest["snapshot"][section]
                )
            assert collective.manifest["snapshot"]["selection"]["parallel"] is True
            assert serial.manifest["snapshot"]["selection"]["parallel"] is False
        except BaseException as exc:  # broadcast a rank-0 verification failure before asserting
            failure = "%s: %s" % (type(exc).__name__, exc)
    failure = communicator.bcast(failure, root=0)
    assert failure is None, failure


def test_runtime_consumer_publishes_collective_hdf5_from_uniform_local_boxes(tmp_path):
    """Accepted runtime state reaches HDF5 only through the production consumer transaction."""
    communicator = _parallel_hdf5_world(
        test_runtime_consumer_publishes_collective_hdf5_from_uniform_local_boxes.__name__
    )
    if communicator is None:
        return
    assert "pops._pops" not in sys.modules
    rank = int(communicator.Get_rank())
    shared_root = communicator.bcast(str(tmp_path) if rank == 0 else None, root=0)
    target = Path(shared_root) / "runtime-consumer.h5"
    runtime, consumer = _runtime_with_collective_hdf5(target, communicator)
    assert type(runtime._publisher) is RuntimeConsumerPublisher

    (transaction,) = runtime._fire_consumers()

    assert transaction.status == "accepted"
    assert len(transaction.published) == 1
    receipt = transaction.published[0]
    assert receipt.publisher_id == "pops.exact-output.v1"
    assert runtime.consumer_cursors.for_consumer(consumer.qualified_id).committed_samples == 1
    receipts = communicator.allgather(receipt.to_data())
    assert all(row == receipts[0] for row in receipts)
    assert "pops._pops" not in sys.modules

    failure = None
    if rank == 0:
        try:
            reopened = read_hdf5(target)
            (dataset,) = reopened.manifest["datasets"]["fields"].values()
            ny, nx = runtime.ny(), runtime.nx()
            expected = np.arange(ny * nx, dtype=np.float64).reshape(1, ny, nx) + np.float64(1.125)
            np.testing.assert_array_equal(reopened.arrays[dataset], expected)
            snapshot = reopened.manifest["snapshot"]
            assert reopened.manifest["format"] == "hdf5"
            assert snapshot["selection"]["consumer_id"] == consumer.qualified_id
            assert snapshot["selection"]["parallel"] is True
            assert snapshot["clock"]["macro_step"] == 1
            assert snapshot["provenance"]["source"] == "runtime-instance-accepted-state"
            assert snapshot["metadata"] == {
                "consumer_graph": runtime.consumer_graph.identity.token,
                "runtime_plan": runtime._runtime_plan.identity.token,
            }
            pieces = snapshot["fields"][0]["pieces"]
            assert len(pieces) == int(communicator.Get_size())
            assert pieces[0]["lower"] == [0, 0]
            assert pieces[-1]["upper"] == [ny, nx]
        except BaseException as exc:  # keep every rank on the same assertion path
            failure = "%s: %s" % (type(exc).__name__, exc)
    failure = communicator.bcast(failure, root=0)
    assert failure is None, failure
