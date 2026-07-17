"""Collective HDF5 writer proofs under a real nested MPI launch.

The final public Uniform/AMR lifecycle is an explicit MPI-manifest entrypoint in
``tests/python/integration/mpi/test_scientific_output_mpi.py``.  This focused writer proof compares
disjoint collective hyperslabs with a serial publication and exercises pre-HDF5 error consensus.
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

from pops import _pops
from pops._native_collectives import (
    allgather_value,
    barrier,
    broadcast_value,
    rank as world_rank,
    size as world_size,
)
from pops.identity import make_identity
from pops.model import Handle, OwnerKind, OwnerPath
from pops.output import (
    ArrayPiece,
    FieldKey,
    FieldPayload,
    HDF5Writer,
    LevelGeometry,
    OutputClock,
    OutputProvenance,
    OutputRequest,
    OutputSnapshot,
    ParallelMode,
    read_hdf5,
)
from tests.python.support.requirements import require_mpi_or_skip


pytestmark = pytest.mark.mpi
_MPI_CHILD = "POPS_HDF5_MPI_TEST_CHILD"


def _missing_mpi_requirement(reason: str) -> None:
    require_mpi_or_skip(reason, optional_skip=pytest.skip)


def _identity(domain: str, name: str):
    return make_identity(domain, {"name": name})


def _snapshots(communicator):
    rank = world_rank(communicator)
    size = world_size(communicator)
    rows_per_rank, nx = 2, 5
    ny = rows_per_rank * size
    global_values = np.arange(ny * nx, dtype=np.float64).reshape(ny, nx) + np.float64(0.125)
    lower = (rank * rows_per_rank, 0)
    upper = ((rank + 1) * rows_per_rank, nx)
    local_piece = ArrayPiece(
        lower,
        upper,
        global_values[lower[0] : upper[0], :],
        rank,
        rank,
        False,
    )
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
        tuple(
            (owner * rows_per_rank, 0, (owner + 1) * rows_per_rank, nx)
            for owner in range(size)
        ),
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

    serial_pieces = tuple(ArrayPiece(
        (owner * rows_per_rank, 0),
        ((owner + 1) * rows_per_rank, nx),
        global_values[owner * rows_per_rank : (owner + 1) * rows_per_rank, :],
        owner,
        0,
        False,
    ) for owner in range(size))
    return snapshot((local_piece,)), snapshot(serial_pieces), key, global_values


def _parallel_hdf5_world(test_name: str):
    try:
        import h5py  # noqa: F401 -- serial native reopen verification
    except ImportError:
        _missing_mpi_requirement("collective HDF5 requires h5py")
    if getattr(_pops, "__has_parallel_hdf5__", False) is not True:
        _missing_mpi_requirement("collective HDF5 requires the compiled C++ parallel-HDF5 route")
    communicator = _pops.mpi_world()
    if world_size(communicator) == 1 and os.environ.get(_MPI_CHILD) != "1":
        mpiexec = shutil.which("mpiexec") or shutil.which("mpirun")
        if mpiexec is None:
            _missing_mpi_requirement(
                "collective HDF5 requires an MPI launcher for the two-rank proof")
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
    assert world_size(communicator) >= 2, "MPI child did not start with two ranks"
    return communicator


def test_collective_hdf5_roundtrip_matches_serial(tmp_path):
    communicator = _parallel_hdf5_world(test_collective_hdf5_roundtrip_matches_serial.__name__)
    if communicator is None:
        return
    rank = world_rank(communicator)

    local_snapshot, serial_snapshot, key, expected = _snapshots(communicator)
    shared_root = broadcast_value(
        communicator, str(tmp_path) if rank == 0 else None, root=0)
    collective_target = Path(shared_root) / "collective.h5"
    serial_target = Path(shared_root) / "serial.h5"
    collective_request = OutputRequest(
        "density-output", (key,), ParallelMode.COLLECTIVE,
        rank=rank, size=world_size(communicator))
    serial_request = OutputRequest(
        "density-output", (key,), ParallelMode.SERIAL)

    prepared = HDF5Writer(ParallelMode.COLLECTIVE).prepare_session(
        local_snapshot,
        collective_request,
        collective_target,
        communicator=communicator,
    )
    prepared.stage()
    collective_receipt = prepared.publish()

    failure = None
    if rank == 0:
        try:
            serial_session = HDF5Writer().prepare_session(
                serial_snapshot, serial_request, serial_target)
            serial_session.stage()
            serial_receipt = serial_session.publish()
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
                "diagnostics",
                "metadata",
            ):
                assert (
                    collective.manifest["snapshot"][section]
                    == serial.manifest["snapshot"][section]
                )
            collective_owners = {
                piece["owner_rank"]
                for field in collective.manifest["snapshot"]["fields"]
                for piece in field["pieces"]
            }
            serial_owners = {
                piece["owner_rank"]
                for field in serial.manifest["snapshot"]["fields"]
                for piece in field["pieces"]
            }
            assert collective_owners == set(range(world_size(communicator)))
            assert serial_owners == {0}
            assert collective.manifest["snapshot"]["selection"]["parallel_mode"] \
                == "collective"
            assert collective.manifest["snapshot"]["selection"]["ranks"] \
                == list(range(world_size(communicator)))
            assert serial.manifest["snapshot"]["selection"]["parallel_mode"] == "serial"
            assert serial.manifest["snapshot"]["selection"]["rank"] == 0
        except BaseException as exc:
            failure = "%s: %s" % (type(exc).__name__, exc)
    failure = broadcast_value(communicator, failure, root=0)
    assert failure is None, failure


def test_collective_hdf5_refuses_rank_local_metadata_before_write(tmp_path):
    communicator = _parallel_hdf5_world(
        test_collective_hdf5_refuses_rank_local_metadata_before_write.__name__
    )
    if communicator is None:
        return
    rank = world_rank(communicator)
    snapshot, _serial_snapshot, key, _expected = _snapshots(communicator)
    snapshot = replace(snapshot, metadata={"case": "collective-hdf5", "rank": rank})
    shared_root = Path(broadcast_value(
        communicator, str(tmp_path) if rank == 0 else None, root=0))
    target = shared_root / "must-not-exist.h5"

    with pytest.raises(ValueError, match="metadata differs across ranks"):
        session = HDF5Writer(ParallelMode.COLLECTIVE).prepare_session(
            snapshot,
            OutputRequest(
                "density-output", (key,), ParallelMode.COLLECTIVE,
                rank=rank, size=world_size(communicator)),
            target,
            communicator=communicator,
        )
        session.stage()
    barrier(communicator)
    assert not target.exists()
    assert not tuple(shared_root.glob(".*must-not-exist*.tmp"))


def test_collective_hdf5_refuses_divergent_target_before_write(tmp_path):
    communicator = _parallel_hdf5_world(
        test_collective_hdf5_refuses_divergent_target_before_write.__name__
    )
    if communicator is None:
        return
    rank = world_rank(communicator)
    snapshot, _serial_snapshot, key, _expected = _snapshots(communicator)
    shared_root = Path(broadcast_value(
        communicator, str(tmp_path) if rank == 0 else None, root=0))
    target = shared_root / ("rank-%d-must-not-exist.h5" % rank)
    with pytest.raises(ValueError, match="preflight differs across ranks"):
        session = HDF5Writer(ParallelMode.COLLECTIVE).prepare_session(
            snapshot,
            OutputRequest(
                "density-output", (key,), ParallelMode.COLLECTIVE,
                rank=rank, size=world_size(communicator)),
            target,
            communicator=communicator,
        )
        session.stage()
    barrier(communicator)
    assert not tuple(shared_root.glob("*must-not-exist.h5"))


def test_native_collective_hdf5_binding_failure_is_all_rank_consensus(tmp_path):
    communicator = _parallel_hdf5_world(
        test_native_collective_hdf5_binding_failure_is_all_rank_consensus.__name__
    )
    if communicator is None:
        return
    rank = world_rank(communicator)
    shared_root = Path(broadcast_value(
        communicator, str(tmp_path) if rank == 0 else None, root=0))
    target = shared_root / "binding-must-not-enter-hdf5.h5"
    values = (
        [[1.0, 2.0], [3.0, 4.0]]
        if rank == 0
        else np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    )
    fields = ({
        "dataset": "fields/0000/values",
        "dtype": np.dtype(np.float64).str,
        "shape": (2, 2),
        "pieces": ({"lower": (0, 0), "upper": (2, 2), "values": values},),
    },)
    error = None
    try:
        _pops._write_parallel_hdf5(
            communicator,
            str(target),
            "{}",
            {"geometry/0000/coverage": np.zeros((2, 2), dtype=np.bool_)},
            fields,
        )
    except RuntimeError as exc:
        error = str(exc)
    errors = allgather_value(communicator, error)
    assert all(item is not None and "binding input validation" in item for item in errors)
    assert len(set(errors)) == 1
    barrier(communicator)
    assert not target.exists()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
