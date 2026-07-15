"""Collective HDF5 proofs at the writer and final public-lifecycle boundaries.

The low-level proof compares disjoint collective hyperslabs with a serial publication. The native
proof launches the complete ``Case -> validate -> resolve(Production) -> compile -> bind -> run``
path, inspects only the public rank-owned state surface, and validates the collective file. Missing
optional MPI/HDF5 prerequisites are explicit skips; execution never degrades to a serial writer.
"""

from __future__ import annotations

from dataclasses import replace
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pops
import pytest

from pops.codegen import Production
from pops.domain import Rectangle
from pops.frames import Cartesian2D
from pops.identity import make_identity
from pops.layouts import Uniform
from pops.math import ddt, div
from pops.mesh import CartesianGrid, PeriodicAxes
from pops.model import Handle, OwnerKind, OwnerPath
from pops.numerics import DiscretizationPlan, FiniteVolume, reconstruction, riemann, variables
from pops.output import (
    ArrayPiece,
    ConsumerGraph,
    FieldKey,
    FieldPayload,
    HDF5,
    HDF5Writer,
    LevelGeometry,
    OutputClock,
    OutputProvenance,
    OutputRequest,
    OutputSnapshot,
    ScientificOutput,
    read_hdf5,
)
from pops.representations import Conservative
from pops.spaces import CellState
from pops.time import FixedDt, StagePoint, TimePoint, every


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


def _native_case() -> tuple[pops.Case, Uniform, np.ndarray]:
    """Author a nonuniform scalar state whose accepted zero-flux step preserves every cell."""
    frame = Rectangle(
        "collective_hdf5_square", lower=(0.0, 0.0), upper=(1.0, 1.0)
    ).frame(Cartesian2D())
    x_axis, y_axis = frame.axes
    model = pops.Model("collective_hdf5_scalar", frame=frame)
    state = model.state(
        "U",
        components=("rho",),
        representation=Conservative(),
        space=CellState(frame=frame),
    )
    (rho,) = state
    flux = model.flux(
        "stationary_flux",
        frame=frame,
        state=state,
        components={x_axis: (0.0 * rho,), y_axis: (0.0 * rho,)},
        waves={x_axis: (0.0,), y_axis: (0.0,)},
    )
    rate = model.rate("stationary_rate", equation=ddt(state) == -div(flux))
    numerics = DiscretizationPlan()
    numerics.rates.add(
        rate,
        FiniteVolume(
            flux=flux,
            variables=variables.Conservative(state),
            reconstruction=reconstruction.FirstOrder(),
            riemann=riemann.Rusanov(),
        ),
    )

    case = pops.Case("collective_hdf5_native_case")
    block = case.block("fluid", model=model)
    block_state = block[state]
    case.numerics(numerics, block=block)
    program = pops.Program("collective_hdf5_forward_euler")
    temporal = program.state(block_state)
    stage = StagePoint("collective_hdf5_stage", {"main": TimePoint(program.clock, 0)})
    derivative = program.value("collective_hdf5_rate", rate(temporal.n), at=stage)
    accepted = program.value(
        "collective_hdf5_accepted",
        temporal.n + program.dt * derivative,
        at=temporal.next.point,
    )
    program.commit(temporal.next, accepted)
    program.step_strategy(FixedDt(0.125))
    case.program(program)
    case.consumers(ConsumerGraph.from_consumers((ScientificOutput(
        format=HDF5(parallel=True),
        schedule=every(1, clock=program.clock),
        fields=(block_state,),
        target="density",
    ),)))

    cells = 8
    layout = Uniform(CartesianGrid(
        frame=frame,
        cells=(cells, cells),
        periodic=PeriodicAxes(frame.axes),
    ))
    initial = np.arange(cells * cells, dtype=np.float64).reshape(1, cells, cells)
    initial += np.float64(1.125)
    return case, layout, initial


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
            timeout=360,
            check=False,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        return None
    assert int(communicator.Get_size()) >= 2, "MPI child did not start with two ranks"
    return communicator


def _assert_identity_consensus(communicator, **identities):
    local = {name: identity.token for name, identity in identities.items()}
    gathered = communicator.allgather(local)
    assert all(item == gathered[0] for item in gathered), gathered


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
                    collective.manifest["snapshot"][section]
                    == serial.manifest["snapshot"][section]
                )
            assert collective.manifest["snapshot"]["selection"]["parallel"] is True
            assert serial.manifest["snapshot"]["selection"]["parallel"] is False
        except BaseException as exc:
            failure = "%s: %s" % (type(exc).__name__, exc)
    failure = communicator.bcast(failure, root=0)
    assert failure is None, failure


def test_collective_hdf5_refuses_rank_local_metadata_before_write(tmp_path):
    communicator = _parallel_hdf5_world(
        test_collective_hdf5_refuses_rank_local_metadata_before_write.__name__
    )
    if communicator is None:
        return
    rank = int(communicator.Get_rank())
    snapshot, _serial_snapshot, key, _expected = _snapshots(communicator)
    snapshot = replace(snapshot, metadata={"case": "collective-hdf5", "rank": rank})
    shared_root = Path(communicator.bcast(str(tmp_path) if rank == 0 else None, root=0))
    target = shared_root / "must-not-exist.h5"

    with pytest.raises(ValueError, match="metadata differs across ranks"):
        HDF5Writer().prepare(
            snapshot,
            OutputRequest("density-output", (key,), True),
            target,
            communicator=communicator,
        )
    communicator.Barrier()
    assert not target.exists()
    assert not tuple(shared_root.glob(".*must-not-exist*.tmp"))


@pytest.mark.compiler
@pytest.mark.native_loader
def test_public_lifecycle_publishes_native_local_state_collectively(tmp_path):
    communicator = _parallel_hdf5_world(
        test_public_lifecycle_publishes_native_local_state_collectively.__name__
    )
    if communicator is None:
        return
    rank = int(communicator.Get_rank())
    shared_root = Path(communicator.bcast(str(tmp_path) if rank == 0 else None, root=0))
    os.environ["POPS_CACHE_DIR"] = str(shared_root / ("native-cache-rank-%d" % rank))

    case, layout, initial = _native_case()
    validated = pops.validate(case)
    resolved = pops.resolve(validated, layout=layout, backend=Production())
    artifact = pops.compile(resolved)
    assert artifact.platform_manifest.communicator.require(
        "test artifact communicator"
    ) == "MPI_COMM_WORLD"
    context = pops.ExecutionContext.mpi_world(artifact, communicator)
    runtime = pops.bind(
        artifact,
        initial_state={"fluid": initial},
        resources={"execution_context": context},
    )
    _assert_identity_consensus(
        communicator,
        resolved_plan=resolved.plan_identity,
        artifact=artifact.artifact_identity,
        execution=context.identity,
        bind=runtime.bind_identity,
    )
    report = pops.run(runtime, t_end=0.125, max_steps=1, output_dir=shared_root)
    assert report.accepted_steps == 1
    _assert_identity_consensus(
        communicator,
        run=report.run_identity,
        bind=report.bind_identity,
        execution=report.execution_identity,
        artifact=report.artifact_identity,
    )

    local_boxes = runtime.local_boxes("fluid")
    local_rows = []
    for index, (ilo, jlo, ihi, jhi) in enumerate(local_boxes):
        values = np.asarray(runtime.local_state("fluid", index), dtype=np.float64)
        assert values.shape == (1, jhi - jlo + 1, ihi - ilo + 1)
        np.testing.assert_array_equal(values, initial[:, jlo : jhi + 1, ilo : ihi + 1])
        local_rows.append((ilo, jlo, ihi, jhi))
    rank_boxes = communicator.allgather(tuple(local_rows))
    assert any(rank_boxes), "the native world launch materialized no rank-owned state"
    coverage = np.zeros(initial.shape[1:], dtype=np.int64)
    for boxes in rank_boxes:
        for ilo, jlo, ihi, jhi in boxes:
            coverage[jlo : jhi + 1, ilo : ihi + 1] += 1
    np.testing.assert_array_equal(coverage, np.ones_like(coverage))

    communicator.Barrier()
    failure = None
    if rank == 0:
        try:
            (target,) = tuple(shared_root.glob("*.h5"))
            reopened = read_hdf5(target)
            (dataset,) = reopened.manifest["datasets"]["fields"].values()
            np.testing.assert_array_equal(reopened.arrays[dataset], initial)
            snapshot = reopened.manifest["snapshot"]
            assert reopened.manifest["format"] == "hdf5"
            assert snapshot["selection"]["parallel"] is True
            assert snapshot["clock"]["macro_step"] == 1
            assert snapshot["provenance"]["source"] == "runtime-instance-accepted-state"
            assert len(snapshot["fields"][0]["pieces"]) == sum(map(len, rank_boxes))
        except BaseException as exc:
            failure = "%s: %s" % (type(exc).__name__, exc)
    failure = communicator.bcast(failure, root=0)
    assert failure is None, failure
