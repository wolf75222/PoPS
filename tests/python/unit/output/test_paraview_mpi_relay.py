"""Filesystem-independent PVTU relay over the explicit writer communicator."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from pops.output import MpiRelayToRoot, ParallelMode
from pops.output._writers.paraview import (
    ParaViewWriter,
    _vtu_schema,
    read_paraview_parallel,
)
from tests.python.unit.output.test_exact_writers import _snapshot


def test_per_rank_relay_never_reads_a_peer_filesystem_path(
    tmp_path: Path, monkeypatch,
) -> None:
    import pops.output._writers.paraview as module

    snapshot, serial_request, _foreign = _snapshot()
    selected = snapshot.select(serial_request)
    coarse = next(field for field in selected if field.key.level == 0)
    fine = next(field for field in selected if field.key.level == 1)
    rank_snapshots = (
        replace(
            snapshot,
            fields=(
                replace(coarse, pieces=(replace(coarse.pieces[0], replicated=True),)),
                replace(fine, pieces=()),
            ),
        ),
        replace(
            snapshot,
            fields=(
                replace(
                    coarse,
                    pieces=(replace(
                        coarse.pieces[0], owner_rank=1, replicated=True),),
                ),
                replace(
                    fine,
                    pieces=(replace(fine.pieces[0], owner_rank=1),),
                ),
            ),
        ),
    )
    requests = tuple(
        replace(
            serial_request,
            parallel_mode=ParallelMode.PER_RANK,
            rank=rank,
            size=2,
            diagnostics=(),
        )
        for rank in range(2)
    )
    remote_leaf = ParaViewWriter(
        ParallelMode.PER_RANK,
        collection=False,
    )._stage_file(
        rank_snapshots[1], requests[1], tmp_path / "remote-source-rank-1.vtu")
    remote_bytes = remote_leaf.temporary.read_bytes()
    inaccessible_remote_target = Path("/peer-filesystem-is-not-mounted/piece-rank-1.vtu")
    remote_artifact = {
        "rank": 1,
        "target": str(inaccessible_remote_target),
        "output_identity": remote_leaf.output_identity.token,
        "schema": _vtu_schema(remote_leaf.temporary),
        "selection_identity": requests[1].publication_identity.token,
        "byte_size": len(remote_bytes),
    }

    def allgather(_communicator, value):
        if set(value) == {"rank", "error", "artifact"}:
            return value, {"rank": 1, "error": None, "artifact": remote_artifact}
        return value, {"rank": 1, "error": None}

    gather_calls = []

    def gather(_communicator, payload, *, root=0):
        assert root == 0
        gather_calls.append(bytes(payload))
        return bytes(payload), remote_bytes

    monkeypatch.setattr(module, "allgather_value", allgather)
    monkeypatch.setattr(module, "gather_bytes", gather)
    root_target = tmp_path / "root" / "piece-rank-0.vtu"
    writer = ParaViewWriter(
        ParallelMode.PER_RANK,
        collection=False,
        placement=MpiRelayToRoot(chunk_bytes=268_435_456),
    )
    session = writer.prepare_session(
        rank_snapshots[0], requests[0], root_target, communicator=object())

    session.stage()
    receipt = session.publish()
    session.finalize()

    parallel = read_paraview_parallel(receipt.path)
    assert len(gather_calls) == 1
    assert parallel.kind == "pvtu"
    assert parallel.paths == (
        root_target,
        root_target.parent / inaccessible_remote_target.name,
    )
    assert all(path.is_file() for path in parallel.paths)
    assert not inaccessible_remote_target.exists()
    remote_leaf.discard()


def test_relay_agrees_on_every_local_open_before_any_chunk_collective(
    tmp_path: Path, monkeypatch,
) -> None:
    import pops.output._writers.paraview as module

    snapshot, serial_request, _foreign = _snapshot()
    request = replace(
        serial_request,
        parallel_mode=ParallelMode.PER_RANK,
        rank=0,
        size=2,
        diagnostics=(),
    )
    target = tmp_path / "root" / "piece-rank-0.vtu"
    writer = ParaViewWriter(
        ParallelMode.PER_RANK,
        collection=False,
        placement=MpiRelayToRoot(chunk_bytes=268_435_456),
    )
    session = writer.prepare_session(snapshot, request, target, communicator=object())
    local_leaf = writer._stage_file(snapshot, request, target)
    local_artifact = {
        "rank": 0,
        "target": str(target.resolve()),
        "output_identity": local_leaf.output_identity.token,
        "schema": _vtu_schema(local_leaf.temporary),
        "selection_identity": request.publication_identity.token,
        "byte_size": local_leaf.temporary.stat().st_size,
    }
    remote_artifact = dict(
        local_artifact,
        rank=1,
        target=str(tmp_path / "remote" / "piece-rank-1.vtu"),
    )
    session._vtu = local_leaf
    session._rank_rows = (local_artifact, remote_artifact)

    def allgather(_communicator, value):
        return value, {
            "rank": 1,
            "error": "OSError: simulated rank-local VTU open failure",
        }

    def gather(*_args, **_kwargs):
        pytest.fail("chunk collective started before all VTUs were open")

    monkeypatch.setattr(module, "allgather_value", allgather)
    monkeypatch.setattr(module, "gather_bytes", gather)

    with pytest.raises(RuntimeError, match="simulated rank-local VTU open failure"):
        session._relay_per_rank_vtus()

    local_leaf.discard()
