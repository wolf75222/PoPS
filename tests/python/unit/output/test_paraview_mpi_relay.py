"""Filesystem-independent PVTU relay over the explicit writer communicator."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from pops.model import Handle, OwnerKind, OwnerPath
from pops.output import MpiRelayToRoot, ParallelMode, ReopenedParaViewMultiBlock
from pops.output._writers.paraview import (
    ParaViewWriter,
    _vtu_schema,
    read_paraview,
    read_paraview_parallel,
)
from pops.output.data import OutputRequest
from tests.python.unit.output.test_exact_writers import _snapshot
from tests.python.unit.output.test_paraview_dimensional_vtu import (
    _snapshot as _dimensional_snapshot,
)


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


def _with_owner(snapshot, rank: int):
    fields = []
    for field in snapshot.fields:
        pieces = tuple(replace(piece, owner_rank=rank) for piece in field.pieces)
        fields.append(replace(field, pieces=pieces))
    return replace(snapshot, fields=tuple(fields))


def _mixed_cell_face_snapshot():
    cell_snapshot, _serial = _dimensional_snapshot((2, 2), centering="cell")
    face_snapshot, _face = _dimensional_snapshot((2, 2), centering="face_x")
    face_key = replace(
        face_snapshot.fields[0].key,
        reference=Handle(
            "face_phi",
            kind="state",
            owner=OwnerPath.case("case").child(OwnerKind.BLOCK, "scalar"),
        ),
    )
    mixed = replace(
        cell_snapshot,
        fields=(cell_snapshot.fields[0], replace(face_snapshot.fields[0], key=face_key)),
    )
    request = OutputRequest(
        "vtk",
        (cell_snapshot.fields[0].key, face_key),
        ParallelMode.PER_RANK,
    )
    return mixed, request


def test_per_rank_relay_copies_multiblock_vtm_and_sibling_leaves(
    tmp_path: Path, monkeypatch,
) -> None:
    import pops.output._writers.paraview as module

    mixed, serial_request = _mixed_cell_face_snapshot()
    requests = tuple(
        replace(serial_request, rank=rank, size=2, diagnostics=())
        for rank in range(2)
    )
    remote_companions = []
    remote_leaf = ParaViewWriter(
        ParallelMode.PER_RANK,
        collection=False,
    )._stage_file(
        _with_owner(mixed, 1),
        requests[1],
        tmp_path / "remote-source-rank-1.vtu",
        companions=remote_companions,
    )
    assert remote_leaf.target.suffix == ".vtm"
    remote_files = (remote_leaf, *remote_companions)
    remote_bytes = [item.temporary.read_bytes() for item in remote_files]
    inaccessible = Path("/peer-filesystem-is-not-mounted")
    remote_artifact = {
        "rank": 1,
        "target": str(inaccessible / remote_leaf.target.name),
        "output_identity": remote_leaf.output_identity.token,
        "schema": module._publication_schema(
            remote_leaf.temporary,
            companion_paths=tuple(item.temporary for item in remote_companions),
            block_names=tuple(
                item.target.stem.rsplit("__", 1)[-1] for item in remote_companions
            ),
        ),
        "selection_identity": requests[1].publication_identity.token,
        "byte_size": len(remote_bytes[0]),
        "parts": [
            {
                "name": item.target.name,
                "suffix": item.target.suffix,
                "target": str(inaccessible / item.target.name),
                "output_identity": item.output_identity.token,
                "byte_size": len(payload),
            }
            for item, payload in zip(remote_files, remote_bytes, strict=True)
        ],
    }

    def allgather(_communicator, value):
        if set(value) == {"rank", "error", "artifact"}:
            return value, {"rank": 1, "error": None, "artifact": remote_artifact}
        return value, {"rank": 1, "error": None}

    gather_calls = []
    remote_queue = list(remote_bytes)

    def gather(_communicator, payload, *, root=0):
        assert root == 0
        gather_calls.append(bytes(payload))
        return bytes(payload), remote_queue.pop(0)

    monkeypatch.setattr(module, "allgather_value", allgather)
    monkeypatch.setattr(module, "gather_bytes", gather)
    root_target = tmp_path / "root" / "mixed-rank-0.vtu"
    writer = ParaViewWriter(
        ParallelMode.PER_RANK,
        collection=False,
        placement=MpiRelayToRoot(chunk_bytes=268_435_456),
    )
    session = writer.prepare_session(
        mixed, requests[0], root_target, communicator=object())

    session.stage()
    receipt = session.publish()
    session.finalize()

    assert receipt is not None
    assert receipt.path.suffix == ".vtm"
    assert len(gather_calls) == len(remote_bytes)
    reopened = read_paraview(receipt.path)
    assert isinstance(reopened, ReopenedParaViewMultiBlock)
    assert [name for name, _block in reopened.blocks] == ["cell", "face_x"]
    relayed_vtm = root_target.parent / remote_leaf.target.name
    relayed_leaves = [
        root_target.parent / item.target.name for item in remote_companions
    ]
    assert relayed_vtm.is_file()
    assert all(path.is_file() for path in relayed_leaves)
    remote = read_paraview(relayed_vtm)
    assert isinstance(remote, ReopenedParaViewMultiBlock)
    assert [name for name, _block in remote.blocks] == ["cell", "face_x"]
    assert "face_phi" not in remote.block("cell").arrays
    assert "phi" not in remote.block("face_x").arrays
    assert not (inaccessible / remote_leaf.target.name).exists()
    remote_leaf.discard()
    for item in remote_companions:
        item.discard()
