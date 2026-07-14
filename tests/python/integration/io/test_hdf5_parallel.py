"""Collective HDF5 writes exact distributed pieces and reopens like the serial route.

The writer contract is exercised directly: each MPI rank owns a disjoint hyperslab of one public
``OutputSnapshot``.  When both mpi4py and an MPI-enabled h5py are available, the collective file is
published, authenticated by ``read_hdf5`` and compared with a serial publication of the same global
pieces.  Missing optional MPI/HDF5 support is an explicit skip, never a degraded serial execution.
"""
from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest

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
    read_hdf5,
)


pytestmark = pytest.mark.mpi
_MPI_CHILD = "POPS_HDF5_MPI_TEST_CHILD"


def _identity(domain: str, name: str):
    return make_identity(domain, {"name": name})


def _snapshots(communicator):
    rank = int(communicator.Get_rank())
    size = int(communicator.Get_size())
    rows_per_rank, nx = 2, 5
    ny = rows_per_rank * size
    global_values = (
        np.arange(ny * nx, dtype=np.float64).reshape(ny, nx) + np.float64(0.125)
    )
    lower = (rank * rows_per_rank, 0)
    upper = ((rank + 1) * rows_per_rank, nx)
    local_piece = ArrayPiece(lower, upper, global_values[lower[0]:upper[0], :])
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


def test_collective_hdf5_roundtrip_matches_serial(tmp_path):
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
                f"{__file__}::test_collective_hdf5_roundtrip_matches_serial",
            ],
            cwd=Path(__file__).resolve().parents[4],
            env=env,
            text=True,
            capture_output=True,
            timeout=120,
            check=False,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        return
    assert int(communicator.Get_size()) >= 2, "MPI child did not start with two ranks"
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
            serial_receipt = HDF5Writer().prepare(
                serial_snapshot, serial_request, serial_target
            ).publish()
            collective = read_hdf5(collective_receipt.path).require_selection(
                collective_request
            )
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
                assert collective.manifest["snapshot"][section] == \
                    serial.manifest["snapshot"][section]
            assert collective.manifest["snapshot"]["selection"]["parallel"] is True
            assert serial.manifest["snapshot"]["selection"]["parallel"] is False
        except BaseException as exc:  # broadcast a rank-0 verification failure before asserting
            failure = "%s: %s" % (type(exc).__name__, exc)
    failure = communicator.bcast(failure, root=0)
    assert failure is None, failure
