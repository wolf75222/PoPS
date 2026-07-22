#!/usr/bin/env python3
"""Real two-rank Catalyst live-enabled visualization probe.

This file is intentionally a script-style MPI entrypoint rather than a pytest test.  It must be
launched by the neutral ParaView Python host, for example
``scripts/paraview_python.sh --mpi 2 tests/python/integration/mpi/probe_catalyst_live_mpi.py``.
Missing Catalyst/Conduit modules are a hard failure: this probe must never turn absence of the
optional runtime into evidence that live MPI works.

The probe avoids a second numerical model solely to keep the failure surface focused.  It uses the
same exact objects consumed by a ``RuntimeInstance`` monitor: a native PoPS world, a collectively
duplicated observer lane, one rank-local piece of a canonical distributed ``ObserverFrame``, and
the production ``CatalystPythonProvider`` lifecycle.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import shutil
import sys
from types import SimpleNamespace
import tempfile
import textwrap
import time
from typing import Any


_PIPELINE_SOURCE = r'''# script-version: 2.0
import os
from pathlib import Path
import threading

from paraview import catalyst
from paraview.simple import TrivialProducer
from vtkmodules.vtkParallelCore import vtkMultiProcessController


producer = TrivialProducer(registrationName="mesh")

options = catalyst.Options()
options.GlobalTrigger = "TimeStep"
options.CatalystLiveTrigger = "TimeStep"
options.EnableCatalystLive = 1
options.CatalystLiveURL = os.environ.get("POPS_CATALYST_LIVE_URL", "localhost:22222")

def catalyst_execute(info):
    if threading.current_thread() is threading.main_thread():
        raise RuntimeError("real Catalyst pipeline did not execute on the PoPS worker")
    step = int(info.timestep)
    producer.UpdatePipeline()
    field_info = producer.GetCellDataInformation()["U"]
    if field_info is None:
        raise RuntimeError("real Catalyst pipeline did not receive cell field U")
    if int(field_info.GetNumberOfComponents()) != 1:
        raise RuntimeError("real Catalyst pipeline received an invalid U component count")

    controller = vtkMultiProcessController.GetGlobalController()
    if controller is None:
        raise RuntimeError("real Catalyst pipeline has no global process controller")
    rank = int(controller.GetLocalProcessId())
    size = int(controller.GetNumberOfProcesses())
    marker = Path(os.environ["POPS_CATALYST_MPI_MARKER_DIR"])
    (marker / ("execute-step-%04d-rank-%04d.txt" % (step, rank))).write_text(
        "step=%d\nrank=%d\nsize=%d\nfield=U\nlive=enabled\nworker=background\n"
        % (step, rank, size),
        encoding="utf-8",
    )
'''


_MPI_IMAGE = re.compile(
    r"^lib(?:mpi|pmpi|mpicxx)(?:\.[0-9]+)*(?:\.dylib|\.so(?:\.[0-9]+)*)$")
_ACTIVE_MPI_ENV = {
    "libmpi": "POPS_ACTIVE_MPI_LIBRARY",
    "libpmpi": "POPS_ACTIVE_PMPI_LIBRARY",
    "libmpicxx": "POPS_ACTIVE_MPICXX_LIBRARY",
}


def _loaded_shared_libraries() -> tuple[Path, ...]:
    """Return canonical shared-library images without invoking a platform debugger."""

    if sys.platform == "darwin":
        import ctypes

        dyld = ctypes.CDLL("/usr/lib/libSystem.B.dylib")
        dyld._dyld_image_count.restype = ctypes.c_uint32
        dyld._dyld_get_image_name.argtypes = (ctypes.c_uint32,)
        dyld._dyld_get_image_name.restype = ctypes.c_char_p
        rows = []
        for index in range(int(dyld._dyld_image_count())):
            raw = dyld._dyld_get_image_name(index)
            if raw:
                rows.append(Path(raw.decode("utf-8")).resolve())
        return tuple(sorted(set(rows)))
    if sys.platform.startswith("linux"):
        rows = []
        for line in Path("/proc/self/maps").read_text(encoding="utf-8").splitlines():
            candidate = line.rsplit(None, 1)[-1]
            if candidate.startswith("/"):
                rows.append(Path(candidate).resolve())
        return tuple(sorted(set(rows)))
    raise RuntimeError(
        "Catalyst MPI loaded-library authentication is unsupported on %s" % sys.platform)


def _authenticate_loaded_mpi_stack() -> tuple[str, ...]:
    """Prove Catalyst retained exactly one image for each active MPI library."""

    prefix_text = os.environ.get("CONDA_PREFIX")
    if not prefix_text:
        raise RuntimeError("Catalyst MPI probe requires an active CONDA_PREFIX")
    active_lib = (Path(prefix_text).resolve() / "lib")
    images = _loaded_shared_libraries()
    mpi_images = tuple(path for path in images if _MPI_IMAGE.fullmatch(path.name))
    if not mpi_images:
        raise RuntimeError("Catalyst MPI probe found no loaded MPI shared library")
    foreign = tuple(path for path in mpi_images if not path.is_relative_to(active_lib))
    if foreign:
        raise RuntimeError(
            "Catalyst loaded a second MPI implementation outside the active Conda prefix: %s"
            % ", ".join(map(str, foreign)))
    for family, variable in _ACTIVE_MPI_ENV.items():
        configured = os.environ.get(variable)
        if not configured:
            raise RuntimeError("Catalyst MPI probe requires %s" % variable)
        expected = Path(configured).resolve()
        if not expected.is_file() or not expected.is_relative_to(active_lib):
            raise RuntimeError("Catalyst MPI probe received an invalid %s" % variable)
        family_pattern = re.compile(
            r"^%s(?:\.[0-9]+)*(?:\.dylib|\.so(?:\.[0-9]+)*)$" % re.escape(family))
        loaded = tuple(path for path in mpi_images if family_pattern.fullmatch(path.name))
        if loaded != (expected,):
            raise RuntimeError(
                "Catalyst must load exactly %s for %s, found %s"
                % (expected, family, ", ".join(map(str, loaded)) or "none"))
    return tuple(str(path) for path in mpi_images)


def _error_text(error: BaseException) -> str:
    return "%s: %s" % (type(error).__name__, error)


def _collective_agree(world: Any, phase: str, error: BaseException | None) -> None:
    """Make a local probe failure identical on every rank before the next phase."""

    envelope = {
        "rank": int(world.rank),
        "error": None if error is None else _error_text(error),
    }
    encoded = json.dumps(envelope, sort_keys=True, separators=(",", ":")).encode("utf-8")
    rows = []
    for payload in world.allgather_bytes(encoded):
        try:
            rows.append(json.loads(bytes(payload).decode("utf-8")))
        except BaseException as caught:  # noqa: BLE001 - malformed MPI evidence is fatal
            raise RuntimeError(
                "Catalyst MPI %s returned malformed collective evidence: %s"
                % (phase, _error_text(caught))
            ) from caught
    if len(rows) != int(world.size) or any(
        not isinstance(row, dict)
        or set(row) != {"rank", "error"}
        or row["rank"] != owner
        or (row["error"] is not None and not isinstance(row["error"], str))
        for owner, row in enumerate(rows)
    ):
        raise RuntimeError(
            "Catalyst MPI %s returned malformed rank evidence" % phase)
    failures = [
        "rank %d: %s" % (owner, row["error"])
        for owner, row in enumerate(rows)
        if row["error"] is not None
    ]
    if failures:
        raise RuntimeError(
            "Catalyst MPI %s failed collectively: %s" % (phase, "; ".join(failures))
        )


def _shared_probe_directory(world: Any) -> Path:
    local = b""
    error = None
    if int(world.rank) == 0:
        try:
            local = tempfile.mkdtemp(prefix="pops-catalyst-live-mpi-").encode("utf-8")
        except BaseException as caught:  # noqa: BLE001 - report before broadcast
            error = caught
    _collective_agree(world, "temporary-directory creation", error)
    encoded = world.broadcast_bytes(local, 0)
    try:
        path = Path(bytes(encoded).decode("utf-8"))
        if not path.is_dir():
            raise FileNotFoundError("shared Catalyst probe directory is not visible")
    except BaseException as caught:  # noqa: BLE001 - make shared-filesystem failures collective
        error = caught
        path = Path(".")
    else:
        error = None
    _collective_agree(world, "temporary-directory visibility", error)
    return path


def _live_probe_directory() -> Path | None:
    value = os.environ.get("POPS_CATALYST_LIVE_PROBE_DIR")
    return None if value is None else Path(value).expanduser().resolve()


def _wait_for_client_evidence(
    world: Any,
    name: str,
    validator: Any,
) -> None:
    """Wait on rank zero only, after the Catalyst worker has completed its current frame."""

    root = _live_probe_directory()
    if root is None:
        return
    error = None
    if int(world.rank) == 0:
        try:
            if not root.is_dir():
                raise FileNotFoundError("Catalyst Live probe directory is missing: %s" % root)
            marker = root / name
            failure = root / "client-failed.json"
            deadline = time.monotonic() + 30.0
            while not marker.is_file():
                if failure.is_file():
                    raise RuntimeError(
                        "Catalyst Live client failed: %s"
                        % failure.read_text(encoding="utf-8").strip())
                if time.monotonic() >= deadline:
                    raise TimeoutError("timed out waiting for Catalyst Live %s" % name)
                time.sleep(0.01)
            evidence = json.loads(marker.read_text(encoding="utf-8"))
            validator(evidence)
        except BaseException as caught:  # noqa: BLE001 - report before the next MPI phase
            error = caught
    _collective_agree(world, "live-client %s" % name, error)


def _distributed_frame(rank: int, size: int, macro_step: int) -> Any:
    import numpy as np

    from pops.identity import make_identity
    from pops.model import Handle, OwnerKind, OwnerPath
    from pops.output._consumer_contracts import ParallelMode
    from pops.output.data import (
        ArrayPiece,
        FieldKey,
        FieldPayload,
        LevelGeometry,
        OutputClock,
        OutputProvenance,
        OutputRequest,
        OutputSnapshot,
    )
    from pops.output.observers import ObserverFrame

    if size != 2 or rank not in (0, 1):
        raise ValueError("real Catalyst live MPI probe requires exactly two ranks")

    def identity(domain: str, name: str) -> Any:
        return make_identity(domain, {"name": name})

    layout = identity("layout-plan", "catalyst-live-mpi-uniform")
    component = identity("component-manifest", "catalyst-live-mpi-component")
    owner = OwnerPath.case("catalyst-live-mpi-case").child(OwnerKind.BLOCK, "fluid")
    reference = Handle("U", kind="state", owner=owner)
    key = FieldKey(reference, component, layout, 0, "accepted")

    boxes = ((0, 0, 2, 2), (0, 2, 2, 4))
    geometry = LevelGeometry(
        layout,
        "uniform",
        0,
        (0.0, 0.0),
        (0.25, 0.5),
        (2, 4),
        boxes,
        np.zeros((2, 4), dtype=np.bool_),
        np.full((2, 4), 0.125, dtype=np.float64),
    )
    lower = boxes[rank][:2]
    upper = boxes[rank][2:]
    values = np.full((1, 2, 2), float(rank + 1), dtype=np.float64)
    piece = ArrayPiece(lower, upper, values, rank, rank, False)
    field = FieldPayload(key, "cell", "1", ("u",), (2, 4), (piece,))
    run_identity = identity("run", "catalyst-live-mpi-run")
    snapshot = OutputSnapshot(
        OutputClock.at("macro", 0.05 * macro_step, macro_step, stage="accepted"),
        OutputProvenance(
            identity("resolved-plan", "catalyst-live-mpi-plan"),
            identity("bind", "catalyst-live-mpi-bind"),
            run_identity,
            "accepted-step-transaction",
        ),
        (geometry,),
        (field,),
        {"test": "real-catalyst-live-mpi"},
    )
    request = OutputRequest(
        "catalyst-live-mpi", (key,), ParallelMode.COLLECTIVE, rank=rank, size=size)
    return ObserverFrame(snapshot, request)


def main() -> None:
    # Importing the native extension first lets PoPS initialize MPI_THREAD_MULTIPLE before the
    # neutral host imports ParaView or Catalyst.
    from pops import _pops

    world = _pops.mpi_world()
    rank = int(world.rank)
    size = int(world.size)
    if size != 2:
        raise RuntimeError(
            "real Catalyst live MPI probe requires mpiexec -n 2 (observed %d ranks)" % size)
    if int(world.thread_level) < 3:  # MPI_THREAD_MULTIPLE has the standard value 3.
        raise RuntimeError(
            "real Catalyst live MPI probe requires MPI_THREAD_MULTIPLE; PoPS reports %d"
            % int(world.thread_level))

    import_error = None
    try:
        import catalyst  # noqa: F401 - proves the real Catalyst lifecycle module
        import catalyst_conduit  # noqa: F401 - proves the real Conduit binding
        import paraview  # noqa: F401 - required when Catalyst loads the live pipeline
    except BaseException as caught:  # noqa: BLE001 - optional dependency absence must fail
        import_error = caught
    _collective_agree(world, "required-module import", import_error)

    root = _shared_probe_directory(world)
    pipeline = root / "catalyst_live_mpi_pipeline.py"
    marker_dir = root / "markers"
    creation_error = None
    if rank == 0:
        try:
            marker_dir.mkdir()
            pipeline.write_text(textwrap.dedent(_PIPELINE_SOURCE), encoding="utf-8")
        except BaseException as caught:  # noqa: BLE001 - report before any Catalyst call
            creation_error = caught
    _collective_agree(world, "pipeline creation", creation_error)
    os.environ["POPS_CATALYST_MPI_MARKER_DIR"] = str(marker_dir)

    frames = None
    frame_error = None
    try:
        steps = (4, 5) if _live_probe_directory() is not None else (4,)
        frames = tuple(_distributed_frame(rank, size, step) for step in steps)
    except BaseException as caught:  # noqa: BLE001 - prevent asymmetric Catalyst entry
        frame_error = caught
    _collective_agree(world, "distributed-frame construction", frame_error)
    if frames is None:
        raise RuntimeError("distributed Catalyst frame construction returned no frames")

    lane = world.duplicate_observer_lane("real-catalyst-live-mpi")
    session = None
    try:
        from pops.output.observers import Catalyst, ObserverRun

        context = SimpleNamespace(
            communicator=SimpleNamespace(identity="MPI_COMM_WORLD", handle=world))
        session_error = None
        try:
            declaration = Catalyst(pipeline=str(pipeline))
            session = declaration.open_runtime_session(
                {"worker_communicator": lane}, context)
            authority = session.authority
            if authority.get("threading") != "dedicated_collective" \
                    or authority.get("worker_mpi") is not True:
                raise RuntimeError(
                    "real Catalyst session did not authenticate a collective MPI worker")
        except BaseException as caught:  # noqa: BLE001 - make provider failures collective
            session_error = caught
        _collective_agree(world, "session construction", session_error)
        if session is None:
            raise RuntimeError("real Catalyst session construction returned no session")

        run = ObserverRun(
            frames[0].snapshot.provenance.run_identity,
            {"test": "real-catalyst-live-mpi", "ranks": size},
        )
        delivery_error = None
        reports = ()
        mpi_images: tuple[str, ...] = ()
        worker = None
        queue = None
        try:
            from pops.runtime._observer_runtime import (
                PostCommitObserverQueue,
                PostCommitObserverWorker,
            )

            worker = PostCommitObserverWorker(
                thread_name="real-catalyst-live-mpi-worker")
            queue = PostCommitObserverQueue(
                session,
                run,
                consumer_id="real-catalyst-live-mpi",
                worker_communicator=lane,
                shared_worker=worker,
            )
            queue.submit(frames[0])
            queue.flush()
            if len(frames) == 2:
                def validate_extract(evidence: Any) -> None:
                    if not isinstance(evidence, dict) \
                            or evidence.get("source") != "mesh" \
                            or not isinstance(evidence.get("port"), int):
                        raise RuntimeError("Catalyst Live client extract evidence is invalid")

                _wait_for_client_evidence(
                    world, "client-extract-requested.json", validate_extract)
                queue.submit(frames[1])
                queue.flush()

                def validate_frame(evidence: Any) -> None:
                    expected = {
                        "cells": 8,
                        "color_map": "Viridis",
                        "field": "U",
                        "range": [1.0, 2.0],
                        "representation": "Surface With Edges",
                        "step": 5,
                    }
                    if evidence != expected:
                        raise RuntimeError(
                            "Catalyst Live client frame evidence differs: %r" % evidence)

                _wait_for_client_evidence(
                    world, "client-frame.json", validate_frame)
            reports = queue.close()
            worker.close()
            worker = None
            if len(reports) != len(frames) \
                    or any(report.status != "delivered" for report in reports):
                raise RuntimeError(
                    "Catalyst worker did not deliver every collective frame: %r"
                    % [(report.status, report.reason) for report in reports])
            for frame, report in zip(frames, reports, strict=True):
                receipt = report.receipt
                if receipt is None or receipt.frame_identity != frame.identity:
                    raise RuntimeError("Catalyst receipt authenticates a different frame")
                if receipt.provider_id != "pops.output.catalyst-python.v1":
                    raise RuntimeError("Catalyst receipt exposes an unexpected provider")
                if receipt.detail.get("implementation") != "paraview":
                    raise RuntimeError("Catalyst did not load the ParaView implementation")
                marker = marker_dir / (
                    "execute-step-%04d-rank-%04d.txt" % (frame.macro_step, rank))
                expected = (
                    "step=%d\nrank=%d\nsize=2\nfield=U\nlive=enabled\n"
                    "worker=background\n" % (frame.macro_step, rank))
                if marker.read_text(encoding="utf-8") != expected:
                    raise RuntimeError(
                        "Catalyst live pipeline marker does not authenticate step %d rank %d"
                        % (frame.macro_step, rank))
            mpi_images = _authenticate_loaded_mpi_stack()
        except BaseException as caught:  # noqa: BLE001 - backend already agrees on its lane
            delivery_error = caught
        finally:
            if queue is not None:
                try:
                    queue.close()
                except BaseException as caught:  # noqa: BLE001 - retain the primary failure
                    if delivery_error is None:
                        delivery_error = caught
            if worker is not None:
                try:
                    worker.close()
                except BaseException as caught:  # noqa: BLE001 - retain the primary failure
                    if delivery_error is None:
                        delivery_error = caught
        _collective_agree(world, "post-commit worker delivery", delivery_error)
        if len(frames) == 2:
            def validate_closed(evidence: Any) -> None:
                if evidence != {"received": True}:
                    raise RuntimeError(
                        "Catalyst Live client close evidence differs: %r" % evidence)

            _wait_for_client_evidence(world, "client-closed.json", validate_closed)
    finally:
        lane.close_collectively()

    world.barrier()
    marker_set_error = None
    if rank == 0:
        try:
            expected_markers = {
                "execute-step-%04d-rank-%04d.txt" % (frame.macro_step, owner)
                for frame in frames for owner in range(size)
            }
            actual_markers = {
                path.name for path in marker_dir.glob("execute-step-*-rank-*.txt")}
            if actual_markers != expected_markers:
                raise RuntimeError(
                    "Catalyst pipeline marker set differs from both MPI ranks: %r"
                    % sorted(actual_markers))
        except BaseException as caught:  # noqa: BLE001 - report before peers continue
            marker_set_error = caught
    _collective_agree(world, "complete pipeline marker set", marker_set_error)

    cleanup_error = None
    if rank == 0:
        try:
            shutil.rmtree(root)
        except BaseException as caught:  # noqa: BLE001 - cleanup is part of this probe
            cleanup_error = caught
    _collective_agree(world, "temporary-directory cleanup", cleanup_error)
    if rank == 0:
        print(
            "PASS real Catalyst MPI %s: ranks=2 thread_level=%d "
            "frames=%d pipeline=executed worker=background mpi_images=active-exact(%d) "
            "lifecycle=initialize/execute/finalize"
            % (
                "live-client-connected" if len(frames) == 2 else "live-enabled",
                int(world.thread_level),
                len(frames),
                len(mpi_images),
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
