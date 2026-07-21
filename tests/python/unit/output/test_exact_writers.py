"""ADC-686 exact scientific formats, selection, metadata and transaction protocol."""
from __future__ import annotations

import json
import os
from importlib import import_module
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from pops._platform_contracts import (
    ExecutionContext,
    ExecutionResource,
    proven_serial_manifest,
)
from pops.identity import Identity, make_identity
from pops.model import Handle, OwnerKind, OwnerPath
from pops.output import (
    ArrayPiece, BalanceTerms, DiagnosticKey, DiagnosticPayload,
    FieldKey, FieldPayload,
    HDF5, HDF5Writer, LevelGeometry, NPZ, NPZWriter, OutputClock,
    OutputProvenance, OutputPublicationReceipt, OutputRequest, OutputSnapshot,
    ParaView, ParaViewWriter, composite_integrals, deterministic_target, read_hdf5,
    read_npz, read_paraview, writer_session_authority,
)
from pops.output._consumer_contracts import FailRun, ParallelMode, ScheduleCursor
from pops.output._writers.common import _OutputRecoveryRequired, _StagedOutputFile
from pops.output._writers.hdf5 import _rebuild_parallel_snapshot_data
from pops.output.provider import consumer_format_data
from pops.runtime._consumer import (
    AcceptedSideEffect, ConsumerPayload, PreparedPublication, PublicationReceipt, PublicationTarget,
)
from pops.runtime._output_publisher import (
    ConsumerOutputPublisher,
    OutputPreparation,
    PreparedConsumerOutput,
)
from pops.runtime._runtime_instance import RuntimeInstance


def _identity(domain, name):
    return make_identity(domain, {"name": name})


def _handle(block, name="rho"):
    owner = OwnerPath.case("case").child(OwnerKind.BLOCK, block)
    return Handle(name, kind="state", owner=owner)


def _piece(values, *, lower=(0, 0), box_index=0, owner_rank=0, replicated=False):
    values = np.asarray(values, dtype=np.float64)
    upper = (lower[0] + values.shape[-2], lower[1] + values.shape[-1])
    return ArrayPiece(
        lower, upper, values, box_index, owner_rank, replicated)


def _snapshot(*, fine_value=2.0):
    layout = _identity("layout-plan", "amr")
    manifest = _identity("component-manifest", "fluid")
    coarse = LevelGeometry(
        layout, "amr", 0, (0.0, 0.0), (0.5, 0.5), (2, 2), ((0, 0, 2, 2),),
        np.asarray([[False, True], [False, False]]), np.ones((2, 2)))
    fine = LevelGeometry(
        layout, "amr", 1, (10.0, 20.0), (0.25, 0.25), (4, 4), ((1, 1, 3, 3),),
        np.zeros((4, 4), dtype=bool), np.full((4, 4), 0.25))
    rho = _handle("fluid")
    foreign = _handle("radiation", "energy")
    coarse_key = FieldKey(rho, manifest, layout, 0, "accepted")
    fine_key = FieldKey(rho, manifest, layout, 1, "accepted")
    foreign_key = FieldKey(
        foreign, _identity("component-manifest", "radiation"), layout, 0, "accepted")
    fields = (
        FieldPayload(coarse_key, "cell", "kg.m-3", (), (2, 2),
                     (_piece(np.ones((2, 2))),)),
        FieldPayload(fine_key, "cell", "kg.m-3", (), (4, 4),
                     (_piece(np.full((2, 2), fine_value), lower=(1, 1)),)),
        FieldPayload(foreign_key, "cell", "J.m-3", (), (2, 2),
                     (_piece(np.full((2, 2), 99.0)),)),
    )
    balance = BalanceTerms(11.0, 2.0, 5.0, 3.0, 1.0)
    diagnostic_key = DiagnosticKey(
        Handle("mass_balance", kind="diagnostic", owner=OwnerPath.consumer("balances")),
        manifest, layout, 0, "accepted", "composite_metric_balance")
    diagnostic = DiagnosticPayload(
        diagnostic_key, balance.residual, "kg", {
            "storage_change": balance.storage_change,
            "outward_boundary_flux": balance.outward_boundary_flux,
            "sources": balance.sources, "reflux": balance.reflux,
            "projection": balance.projection,
        })
    snapshot = OutputSnapshot(
        OutputClock.at("macro", 0.125, 7, stage="accepted"),
        OutputProvenance(
            _identity("resolved-plan", "plan"), _identity("bind", "bind"),
            _identity("run", "run"), "accepted-step-transaction"),
        (coarse, fine), fields, {"case": "qualified-output"}, diagnostics=(diagnostic,))
    request = OutputRequest(
        "density-output", (coarse_key, fine_key), ParallelMode.SERIAL,
        diagnostics=(diagnostic_key,))
    return snapshot, request, foreign_key


def _accepted_output_effect(request, target):
    occurrence = _identity("consumer-occurrence", "sample-7")
    payload = ConsumerPayload(
        _identity("runtime-plan-bundle", "runtime"), occurrence, (), ())
    before = ScheduleCursor(request.consumer_id)
    after = ScheduleCursor(request.consumer_id, occurrence.token, 1)
    return AcceptedSideEffect(
        0,
        request.consumer_id,
        _identity("consumer-manifest", "density-output"),
        PublicationTarget(
            str(target), NPZ().consumer_data(), None, ParallelMode.SERIAL),
        payload,
        FailRun(),
        before,
        after,
    )


def _stage_writer(writer, snapshot, request, target, *, communicator=None):
    session = writer.prepare_session(
        snapshot, request, target, communicator=communicator)
    session.stage()
    return session


def _recovery_paths(root):
    return tuple(sorted(root.glob(".pops-quarantine-*/owned")))


def _serial_execution_context():
    return ExecutionContext(
        backend=proven_serial_manifest(
            backend="production", target="system", abi="test|c++|c++23", runtime=True),
        communicator=ExecutionResource("communicator", "serial"),
        datatype=ExecutionResource("datatype", "float64"),
        device=ExecutionResource("device", "host"),
    )


def _external_writer_preparation(
    tmp_path, *, replace_component=False, recreate_after_detach=False,
):
    from pops.runtime._runtime_consumers import _PreparedExternalWriter

    snapshot, request, _ = _snapshot()
    target = tmp_path / "external.npz"
    effect = _accepted_output_effect(request, target)

    class NativeHandle:
        def __init__(self):
            self.operations = []
            self.cleanup_requests = []
            self.original_temporary = None
            self.original_component = None

        def _invoke_component_operation(
            self, _uri, _version, operation, request_data,
        ):
            self.operations.append(operation)
            temporary = Path(request_data["temporary_path"])
            component = Path(request_data["published_path"])
            if operation == "verify":
                self.original_temporary = temporary
                self.original_component = component
                temporary.write_bytes(b"native-writer-output")
                return {
                    "bytes_written": len(b"native-writer-output"),
                    "content_digest": "native-writer-output-v1",
                }
            if operation == "publish":
                os.rename(temporary, component)
                if replace_component:
                    component.unlink()
                    component.write_bytes(b"third-party-component")
                return {
                    "bytes_written": len(b"native-writer-output"),
                    "content_digest": "native-writer-output-v1",
                }
            if operation in {"discard", "rollback"}:
                self.cleanup_requests.append(request_data)
                if recreate_after_detach:
                    self.original_temporary.write_bytes(b"third-party-temporary")
                    self.original_component.write_bytes(b"third-party-component")
                    target.write_bytes(b"third-party-target")
                # Deliberately destructive callback: only private tombstones may be addressable.
                temporary.unlink(missing_ok=True)
                component.unlink(missing_ok=True)
                return None
            raise AssertionError("unexpected native Writer operation %r" % operation)

    native = NativeHandle()
    installed = SimpleNamespace(
        native_handle=native,
        interface=SimpleNamespace(
            to_data=lambda: {"uri": "pops://test/writer", "version": 1}),
        artifact_identity=_identity("component-artifact", "external-writer"),
    )
    prepared = _PreparedExternalWriter(
        effect,
        OutputPreparation(NPZ(), snapshot, request, target),
        installed,
        _serial_execution_context(),
    )
    return prepared, native


def test_npz_prepare_reopen_publish_is_exact_and_deterministic(tmp_path):
    snapshot, request, foreign = _snapshot()
    target = deterministic_target(tmp_path, "fields", request, snapshot, ".npz")
    prepared = _stage_writer(NPZWriter(), snapshot, request, target)
    assert prepared.temporary.is_file() and not target.exists()
    reopened = read_npz(prepared.temporary).require_selection(request)
    assert reopened.manifest["snapshot"]["clock"] == snapshot.clock.to_data()
    assert reopened.manifest["snapshot"]["provenance"] == snapshot.provenance.to_data()
    diagnostic = reopened.manifest["snapshot"]["diagnostics"][0]
    assert set(diagnostic["terms"]) == {
        "storage_change", "outward_boundary_flux", "sources", "reflux", "projection"}
    assert foreign.identity.token not in reopened.manifest["datasets"]["fields"]
    assert len(reopened.manifest["datasets"]["fields"]) == 2
    receipt = prepared.publish()
    assert receipt.path == target and target.is_file() and not prepared.temporary.exists()
    prepared.abort_prepare()
    prepared.abort_prepare()
    assert target.is_file()
    assert deterministic_target(tmp_path, "fields", request, snapshot, ".npz") == target
    read_npz(target).require_selection(request)


def test_deterministic_target_is_bounded_and_hashes_full_long_identities(tmp_path):
    snapshot, request, _ = _snapshot()
    long_consumer = "consumer-" + "a" * 400
    long_clock = "clock-" + "b" * 400
    first_request = replace(request, consumer_id=long_consumer)
    first_snapshot = replace(snapshot, clock=replace(snapshot.clock, clock_id=long_clock))
    first = deterministic_target(tmp_path, "fields-" + "p" * 400,
                                 first_request, first_snapshot, ".npz")
    repeated = deterministic_target(tmp_path, "fields-" + "p" * 400,
                                    first_request, first_snapshot, ".npz")
    changed_request = replace(request, consumer_id=long_consumer[:-1] + "z")
    changed = deterministic_target(tmp_path, "fields-" + "p" * 400,
                                   changed_request, first_snapshot, ".npz")

    assert first == repeated
    assert first != changed
    assert len(first.name.encode("utf-8")) <= 255
    assert first.suffix == ".npz"


def test_npz_collision_and_discard_never_publish_partial_content(tmp_path):
    snapshot, request, _ = _snapshot()
    target = deterministic_target(tmp_path, "fields", request, snapshot, ".npz")
    _stage_writer(NPZWriter(), snapshot, request, target).publish()
    changed, _, _ = _snapshot(fine_value=3.0)
    competing = _stage_writer(NPZWriter(), changed, request, target)
    with pytest.raises(FileExistsError, match="collision"):
        competing.publish()
    competing.abort_prepare()
    competing.abort_prepare()
    assert not competing.temporary.exists()
    first = read_npz(target)
    fine = next(key for key in request.selection if key.level == 1)
    fine_record = first.manifest["datasets"]["fields"][fine.identity.token]
    (fine_piece,) = fine_record["pieces"]
    fine_name = fine_piece["name"]
    assert np.all(first.arrays[fine_name] == 2.0)


def test_npz_publication_rollback_removes_only_the_artifact_it_created(tmp_path):
    snapshot, request, _ = _snapshot()
    target = deterministic_target(tmp_path, "fields", request, snapshot, ".npz")
    created = _stage_writer(NPZWriter(), snapshot, request, target)
    created.publish()
    created.rollback()
    created.rollback()
    assert not target.exists()

    _stage_writer(NPZWriter(), snapshot, request, target).publish()
    idempotent = _stage_writer(NPZWriter(), snapshot, request, target)
    idempotent.publish()
    idempotent.rollback()
    assert target.is_file()
    read_npz(target).require_selection(request)


def test_npz_rollback_refuses_to_remove_a_replaced_target(tmp_path):
    snapshot, request, _ = _snapshot()
    target = deterministic_target(tmp_path, "fields", request, snapshot, ".npz")
    session = _stage_writer(NPZWriter(), snapshot, request, target)
    session.publish()
    target.unlink()
    target.write_bytes(b"third-party-replacement")

    with pytest.raises(RuntimeError, match="rollback refused a replaced target"):
        session.rollback()
    assert target.read_bytes() == b"third-party-replacement"
    (recovery,) = _recovery_paths(tmp_path)
    assert os.path.samefile(recovery, target)
    session.cleanup_recoveries()
    assert target.read_bytes() == b"third-party-replacement"
    assert _recovery_paths(tmp_path) == ()


def test_npz_abort_refuses_to_remove_a_replaced_temporary(tmp_path):
    snapshot, request, _ = _snapshot()
    target = deterministic_target(tmp_path, "fields", request, snapshot, ".npz")
    session = _stage_writer(NPZWriter(), snapshot, request, target)
    temporary = session.temporary
    temporary.unlink()
    temporary.write_bytes(b"third-party-temporary")

    with pytest.raises(RuntimeError, match="refuses to delete a replaced temporary"):
        session.abort_prepare()
    assert temporary.read_bytes() == b"third-party-temporary"
    (recovery,) = _recovery_paths(tmp_path)
    assert os.path.samefile(recovery, temporary)
    session.cleanup_recoveries()
    assert temporary.read_bytes() == b"third-party-temporary"
    assert _recovery_paths(tmp_path) == ()


def test_npz_publish_source_race_preserves_every_unauthenticated_path(tmp_path, monkeypatch):
    snapshot, request, _ = _snapshot()
    target = deterministic_target(tmp_path, "fields", request, snapshot, ".npz")
    session = _stage_writer(NPZWriter(), snapshot, request, target)
    temporary = session.temporary
    real_link = os.link
    raced = False

    def replace_before_link(source, destination, *args, **kwargs):
        nonlocal raced
        if not raced and Path(source) == temporary and Path(destination) == target:
            temporary.unlink()
            temporary.write_bytes(b"third-party-temporary")
            raced = True
        real_link(source, destination, *args, **kwargs)

    monkeypatch.setattr(os, "link", replace_before_link)
    with pytest.raises(RuntimeError, match="staging inode changed during publication"):
        session.publish()
    assert target.read_bytes() == b"third-party-temporary"

    with pytest.raises(RuntimeError, match="refuses to delete a replaced temporary"):
        session.rollback()
    assert target.read_bytes() == b"third-party-temporary"
    assert temporary.read_bytes() == b"third-party-temporary"
    recoveries = _recovery_paths(tmp_path)
    assert len(recoveries) == 2
    assert all(os.path.samefile(recovery, temporary) for recovery in recoveries)


def test_npz_publish_rejects_a_target_replaced_after_link(tmp_path, monkeypatch):
    snapshot, request, _ = _snapshot()
    target = deterministic_target(tmp_path, "fields", request, snapshot, ".npz")
    session = _stage_writer(NPZWriter(), snapshot, request, target)
    temporary = session.temporary
    real_link = os.link
    raced = False

    def replace_target_after_link(source, destination, *args, **kwargs):
        nonlocal raced
        real_link(source, destination, *args, **kwargs)
        if not raced and Path(source) == temporary and Path(destination) == target:
            target.unlink()
            target.write_bytes(b"third-party-target")
            raced = True

    monkeypatch.setattr(os, "link", replace_target_after_link)
    with pytest.raises(RuntimeError, match="staging inode changed during publication"):
        session.publish()

    with pytest.raises(RuntimeError, match="rollback refused a replaced target"):
        session.rollback()
    assert target.read_bytes() == b"third-party-target"
    assert not temporary.exists()
    (recovery,) = _recovery_paths(tmp_path)
    assert os.path.samefile(recovery, target)


def test_npz_abort_quarantines_a_temporary_replaced_during_removal(tmp_path, monkeypatch):
    common = import_module("pops.output._writers.common")
    snapshot, request, _ = _snapshot()
    target = deterministic_target(tmp_path, "fields", request, snapshot, ".npz")
    session = _stage_writer(NPZWriter(), snapshot, request, target)
    temporary = session.temporary
    real_rename = common._rename_no_replace
    raced = False

    def replace_temporary_before_quarantine(source, destination, *args, **kwargs):
        nonlocal raced
        if not raced and source == temporary.name and kwargs.get("src_dir_fd") is not None:
            temporary.unlink()
            temporary.write_bytes(b"third-party-temporary")
            raced = True
        real_rename(source, destination, *args, **kwargs)

    monkeypatch.setattr(common, "_rename_no_replace", replace_temporary_before_quarantine)
    with pytest.raises(RuntimeError, match="refuses to delete a replaced temporary"):
        session.abort_prepare()
    assert temporary.read_bytes() == b"third-party-temporary"
    (recovery,) = _recovery_paths(tmp_path)
    assert os.path.samefile(recovery, temporary)


def test_npz_rollback_quarantines_a_target_replaced_during_removal(tmp_path, monkeypatch):
    common = import_module("pops.output._writers.common")
    snapshot, request, _ = _snapshot()
    target = deterministic_target(tmp_path, "fields", request, snapshot, ".npz")
    session = _stage_writer(NPZWriter(), snapshot, request, target)
    session.publish()
    real_rename = common._rename_no_replace
    raced = False

    def replace_target_before_quarantine(source, destination, *args, **kwargs):
        nonlocal raced
        if not raced and source == target.name and kwargs.get("src_dir_fd") is not None:
            target.unlink()
            target.write_bytes(b"third-party-target")
            raced = True
        real_rename(source, destination, *args, **kwargs)

    monkeypatch.setattr(common, "_rename_no_replace", replace_target_before_quarantine)
    with pytest.raises(RuntimeError, match="rollback refused a replaced target"):
        session.rollback()
    assert target.read_bytes() == b"third-party-target"
    (recovery,) = _recovery_paths(tmp_path)
    assert os.path.samefile(recovery, target)


@pytest.mark.parametrize(
    ("module_name", "writer_factory", "suffix", "reader_name"),
    (
        ("pops.output._writers.npz", NPZWriter, ".npz", "read_npz"),
        ("pops.output._writers.paraview", ParaViewWriter, ".vtu", "read_paraview"),
        ("pops.output._writers.hdf5", HDF5Writer, ".h5", "read_hdf5"),
    ),
)
def test_writer_factory_rejects_substitution_before_staged_constructor(
    tmp_path, monkeypatch, module_name, writer_factory, suffix, reader_name,
):
    module = import_module(module_name)
    snapshot, request, _ = _snapshot()
    target = tmp_path / ("substitution" + suffix)
    captured = []
    real_temporary_path = module.temporary_path
    real_reader = getattr(module, reader_name)

    def capture_authority(path):
        authority = real_temporary_path(path)
        captured.append(authority)
        return authority

    def replace_after_verification(path):
        reopened = real_reader(path)
        authority = captured[-1]
        authority.path.unlink()
        authority.path.write_bytes(b"third-party-substitution")
        return reopened

    monkeypatch.setattr(module, "temporary_path", capture_authority)
    monkeypatch.setattr(module, reader_name, replace_after_verification)
    session = writer_factory().prepare_session(snapshot, request, target)
    with pytest.raises(RuntimeError, match="replaced before authority transfer"):
        session.stage()

    (authority,) = captured
    assert not authority.is_open
    assert authority.path.read_bytes() == b"third-party-substitution"


def test_collective_rank_zero_retains_mkstemp_authority_through_quarantine_cleanup(
    tmp_path, monkeypatch,
):
    native_collectives = import_module("pops._native_collectives")
    hdf5 = import_module("pops.output._writers.hdf5")
    communicator = object()
    monkeypatch.setattr(native_collectives, "rank", lambda _communicator: 0)
    monkeypatch.setattr(native_collectives, "size", lambda _communicator: 1)
    monkeypatch.setattr(
        native_collectives, "allgather_value", lambda _communicator, value: (value,))
    monkeypatch.setattr(
        native_collectives,
        "broadcast_value",
        lambda _communicator, value, root=0: value,
    )
    monkeypatch.setattr(native_collectives, "barrier", lambda _communicator: None)

    authority = hdf5._parallel_temporary_path(tmp_path / "collective.h5", communicator)
    descriptor = os.fstat(authority.fileno())
    assert authority.owner == (int(descriptor.st_dev), int(descriptor.st_ino))
    assert hdf5._collective_temporary_owner(communicator, authority) == authority.owner
    temporary = authority.path
    assert hdf5._collective_remove(communicator, authority) is None
    assert not authority.is_open
    assert not temporary.exists()
    assert _recovery_paths(tmp_path) == ()


def test_collective_temporary_consensus_failure_quarantines_and_closes_rank_zero_fd(
    tmp_path, monkeypatch,
):
    native_collectives = import_module("pops._native_collectives")
    hdf5 = import_module("pops.output._writers.hdf5")
    communicator = object()
    captured = []
    real_temporary_path = hdf5.temporary_path

    def capture_authority(target):
        authority = real_temporary_path(target)
        captured.append(authority)
        return authority

    monkeypatch.setattr(hdf5, "temporary_path", capture_authority)
    monkeypatch.setattr(native_collectives, "rank", lambda _communicator: 0)
    monkeypatch.setattr(native_collectives, "size", lambda _communicator: 1)
    monkeypatch.setattr(
        native_collectives,
        "allgather_value",
        lambda _communicator, _value: ({"malformed": True},),
    )
    monkeypatch.setattr(
        native_collectives,
        "broadcast_value",
        lambda _communicator, value, root=0: value,
    )

    with pytest.raises(RuntimeError, match="temporary-file authority is malformed"):
        hdf5._parallel_temporary_path(tmp_path / "collective-failure.h5", communicator)

    (authority,) = captured
    assert not authority.is_open
    assert not authority.path.exists()
    assert _recovery_paths(tmp_path) == ()


def test_collective_owner_mismatch_is_consensus_evidence_before_cleanup(tmp_path, monkeypatch):
    common = import_module("pops.output._writers.common")
    hdf5 = import_module("pops.output._writers.hdf5")
    native_collectives = import_module("pops._native_collectives")
    communicator = object()
    authority = common.temporary_path(tmp_path / "owner-mismatch.h5")

    monkeypatch.setattr(native_collectives, "rank", lambda _communicator: 0)
    monkeypatch.setattr(native_collectives, "size", lambda _communicator: 2)

    def mixed_owner_rows(_communicator, envelope):
        peer = dict(envelope, rank=1, owner=(envelope["owner"][0], envelope["owner"][1] + 1))
        return envelope, peer

    monkeypatch.setattr(native_collectives, "allgather_value", mixed_owner_rows)
    with pytest.raises(RuntimeError, match="inode authority differs across ranks"):
        hdf5._collective_temporary_owner(communicator, authority)

    common._cleanup_staging_authority(
        authority, replaced_message="test cleanup refused replacement")
    assert not authority.is_open
    assert not authority.path.exists()


def test_collective_constructor_peer_failure_cleans_rank_zero_without_barrier_split(
    tmp_path, monkeypatch,
):
    common = import_module("pops.output._writers.common")
    hdf5 = import_module("pops.output._writers.hdf5")
    native_collectives = import_module("pops._native_collectives")
    communicator = object()
    target = tmp_path / "mixed-constructor.h5"
    authority = common.temporary_path(target)

    monkeypatch.setattr(native_collectives, "rank", lambda _communicator: 0)
    monkeypatch.setattr(native_collectives, "size", lambda _communicator: 2)
    monkeypatch.setattr(
        native_collectives,
        "allgather_value",
        lambda _communicator, envelope: (
            envelope,
            {"rank": 1, "error": "RuntimeError: injected peer constructor failure"},
        ),
    )
    monkeypatch.setattr(
        native_collectives,
        "broadcast_value",
        lambda _communicator, value, root=0: value,
    )
    barriers = []
    monkeypatch.setattr(
        native_collectives, "barrier", lambda _communicator: barriers.append("cleanup"))

    with pytest.raises(RuntimeError, match="injected peer constructor failure"):
        hdf5._construct_collective_staged_output(
            communicator,
            rank=0,
            size=2,
            authority=authority,
            target=target,
            format="hdf5",
            output_identity=_identity("scientific-output", "mixed-constructor"),
            selection_identity=_identity("output-selection", "mixed-constructor"),
            verify=lambda _path: None,
        )

    assert barriers == ["cleanup"]
    assert not authority.is_open
    assert not authority.path.exists()
    assert _recovery_paths(tmp_path) == ()


def test_collective_remove_consensus_includes_nonroot_base_exception(
    tmp_path, monkeypatch,
):
    hdf5 = import_module("pops.output._writers.hdf5")
    native_collectives = import_module("pops._native_collectives")
    communicator = object()

    class RankLocalClose(BaseException):
        pass

    class Authority:
        path = tmp_path / "rank-local-close.h5"

        @staticmethod
        def close():
            raise RankLocalClose("injected non-root close failure")

    monkeypatch.setattr(native_collectives, "rank", lambda _communicator: 1)
    monkeypatch.setattr(native_collectives, "size", lambda _communicator: 2)

    def gather(_communicator, envelope):
        return {"rank": 0, "error": None}, envelope

    monkeypatch.setattr(native_collectives, "allgather_value", gather)
    barriers = []
    monkeypatch.setattr(
        native_collectives, "barrier", lambda _communicator: barriers.append("done"))

    failure = hdf5._collective_remove(communicator, Authority())

    assert "rank 1: RankLocalClose: injected non-root close failure" in failure
    assert barriers == ["done"]


def test_output_finalize_releases_fd_and_normal_paths_leave_no_quarantine(tmp_path):
    snapshot, request, _ = _snapshot()
    target = deterministic_target(tmp_path, "fields", request, snapshot, ".npz")
    session = _stage_writer(NPZWriter(), snapshot, request, target)
    staging = session._staged._staging
    session.publish()
    assert staging.is_open

    session.finalize()
    assert not staging.is_open
    with pytest.raises(RuntimeError, match="finalized writer session"):
        session.rollback()
    assert target.is_file()
    assert _recovery_paths(tmp_path) == ()


def test_post_rename_error_is_recoverable_and_retains_primary_diagnostic(tmp_path, monkeypatch):
    snapshot, request, _ = _snapshot()
    target = deterministic_target(tmp_path, "fields", request, snapshot, ".npz")
    session = _stage_writer(NPZWriter(), snapshot, request, target)
    temporary = session.temporary
    real_stat = os.stat
    owned_calls = 0

    def fail_authentication_after_rename(path, *args, **kwargs):
        nonlocal owned_calls
        if path == "owned" and kwargs.get("dir_fd") is not None:
            owned_calls += 1
            if owned_calls == 2:
                raise OSError("injected post-rename authentication failure")
        return real_stat(path, *args, **kwargs)

    monkeypatch.setattr(os, "stat", fail_authentication_after_rename)
    with pytest.raises(RuntimeError, match="post-rename authentication failure"):
        session.abort_prepare()
    assert not temporary.exists()
    (recovery,) = session.recoveries
    assert recovery.quarantine_path.is_file()

    recovery.restore()
    session.cleanup_recoveries()
    assert temporary.is_file()
    assert _recovery_paths(tmp_path) == ()


def test_recovery_cleanup_retries_after_unlink_then_rmdir_failure(tmp_path, monkeypatch):
    snapshot, request, _ = _snapshot()
    target = deterministic_target(tmp_path, "fields", request, snapshot, ".npz")
    session = _stage_writer(NPZWriter(), snapshot, request, target)
    temporary = session.temporary
    temporary.unlink()
    temporary.write_bytes(b"third-party-temporary")

    with pytest.raises(RuntimeError, match="refuses to delete a replaced temporary"):
        session.abort_prepare()
    (recovery,) = session.recoveries
    assert os.path.samefile(temporary, recovery.quarantine_path)

    real_rmdir = os.rmdir
    failed_once = False

    def fail_first_quarantine_rmdir(path, *args, **kwargs):
        nonlocal failed_once
        if not failed_once and isinstance(path, str) and path.startswith(".pops-quarantine-"):
            failed_once = True
            raise OSError("injected recovery rmdir failure")
        return real_rmdir(path, *args, **kwargs)

    monkeypatch.setattr(os, "rmdir", fail_first_quarantine_rmdir)
    with pytest.raises(RuntimeError, match="recovery cleanup failed"):
        session.cleanup_recoveries()
    assert session.recoveries == (recovery,)
    assert not recovery.quarantine_path.exists()

    session.cleanup_recoveries()
    assert session.recoveries == ()
    assert temporary.read_bytes() == b"third-party-temporary"
    assert _recovery_paths(tmp_path) == ()


def test_quarantine_rmdir_failure_never_masks_primary_rename_error(tmp_path, monkeypatch):
    common = import_module("pops.output._writers.common")
    snapshot, request, _ = _snapshot()
    target = deterministic_target(tmp_path, "fields", request, snapshot, ".npz")
    session = _stage_writer(NPZWriter(), snapshot, request, target)
    real_rename = common._rename_no_replace
    real_rmdir = os.rmdir

    def fail_rename(source, destination, *args, **kwargs):
        if source == session.temporary.name and kwargs.get("src_dir_fd") is not None:
            raise OSError("injected primary rename failure")
        return real_rename(source, destination, *args, **kwargs)

    def fail_rmdir(path, *args, **kwargs):
        if isinstance(path, str) and path.startswith(".pops-quarantine-"):
            raise OSError("injected quarantine rmdir failure")
        return real_rmdir(path, *args, **kwargs)

    monkeypatch.setattr(common, "_rename_no_replace", fail_rename)
    monkeypatch.setattr(os, "rmdir", fail_rmdir)
    with pytest.raises(RuntimeError, match="injected primary rename failure") as failure:
        session.abort_prepare()
    assert "quarantine cleanup also failed" in str(failure.value)


def test_quarantine_atomic_rename_never_overwrites_a_concurrent_owned_entry(
    tmp_path, monkeypatch,
):
    common = import_module("pops.output._writers.common")
    snapshot, request, _ = _snapshot()
    target = deterministic_target(tmp_path, "fields", request, snapshot, ".npz")
    session = _stage_writer(NPZWriter(), snapshot, request, target)
    temporary = session.temporary
    real_rename = common._rename_no_replace

    def occupy_destination_before_atomic_rename(source, destination, *args, **kwargs):
        descriptor = os.open(
            destination,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
            dir_fd=kwargs["dst_dir_fd"],
        )
        try:
            os.write(descriptor, b"third-party-quarantine-entry")
        finally:
            os.close(descriptor)
        return real_rename(source, destination, *args, **kwargs)

    monkeypatch.setattr(
        common, "_rename_no_replace", occupy_destination_before_atomic_rename)
    with pytest.raises(RuntimeError, match="destination appeared concurrently"):
        session.abort_prepare()

    assert temporary.is_file()
    (occupied,) = _recovery_paths(tmp_path)
    assert occupied.read_bytes() == b"third-party-quarantine-entry"


def test_missing_exact_level_state_selection_fails_before_a_writer(tmp_path):
    snapshot, request, _ = _snapshot()
    missing_values = (
        replace(request.selection[0], level=4),
        replace(request.selection[0], state_id="trial"),
        replace(request.selection[0], layout_identity=_identity("layout-plan", "foreign")),
    )
    for missing in missing_values:
        bad = OutputRequest("density-output", (missing,), ParallelMode.SERIAL)
        with pytest.raises(KeyError, match="owner/layout/level/state"):
            _stage_writer(NPZWriter(), snapshot, bad, tmp_path / "missing.npz")
    assert not (tmp_path / "missing.npz").exists()


def test_diagnostic_only_output_keeps_explicit_layout_and_balance_terms(tmp_path):
    snapshot, request, _ = _snapshot()
    diagnostics_only = replace(snapshot, fields=())
    diagnostic_request = OutputRequest(
        "balance-output", (), ParallelMode.SERIAL, diagnostics=request.diagnostics)
    target = tmp_path / "balance.npz"
    prepared = _stage_writer(NPZWriter(), diagnostics_only, diagnostic_request, target)
    reopened = read_npz(prepared.temporary).require_selection(diagnostic_request)
    assert reopened.manifest["datasets"]["fields"] == {}
    assert len(reopened.manifest["snapshot"]["geometries"]) == 2
    assert len(reopened.manifest["snapshot"]["diagnostics"]) == 1
    prepared.abort_prepare()


def test_paraview_geometry_is_byte_exact_and_keeps_row_major_cell_order(tmp_path):
    snapshot, request, foreign = _snapshot()
    target = deterministic_target(tmp_path, "fields", request, snapshot, ".vtu")
    prepared = _stage_writer(ParaViewWriter(), snapshot, request, target)
    duplicate = _stage_writer(
        ParaViewWriter(), snapshot, request, tmp_path / "duplicate.vtu")
    assert prepared.temporary.read_bytes() == duplicate.temporary.read_bytes()
    duplicate.abort_prepare()
    reopened = read_paraview(prepared.temporary).require_selection(request)
    assert set(reopened.manifest["datasets"]["fields"]) == {
        key.identity.token for key in request.selection}
    assert foreign.identity.token not in reopened.manifest["datasets"]["fields"]
    assert np.array_equal(reopened.arrays["pops_level"], [0, 0, 0, 0] + [1] * 4)
    assert int(np.sum(reopened.arrays["pops_coverage"])) == 1
    assert np.array_equal(reopened.arrays["vtkGhostType"], [0, 8, 0, 0] + [0] * 4)
    assert np.array_equal(reopened.arrays["TimeValue"], [0.125])
    coarse = next(item for item in request.selection if item.level == 0)
    fine = next(item for item in request.selection if item.level == 1)
    coarse_record = reopened.manifest["datasets"]["fields"][coarse.identity.token]
    fine_record = reopened.manifest["datasets"]["fields"][fine.identity.token]
    assert coarse_record["name"] == fine_record["name"] == "field_0000"
    assert np.array_equal(
        reopened.arrays["field_0000"], [1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0])
    assert np.array_equal(reopened.arrays["connectivity"], np.arange(32))
    assert np.array_equal(reopened.arrays["offsets"], np.arange(4, 33, 4))
    assert np.array_equal(
        reopened.arrays["Points"][::4, :2],
        np.asarray([
            [0.0, 0.0], [0.5, 0.0], [0.0, 0.5], [0.5, 0.5],
            [10.25, 20.25], [10.5, 20.25], [10.25, 20.5], [10.5, 20.5],
        ]),
    )
    prepared.abort_prepare()


def test_paraview_per_rank_keeps_one_multilevel_family_and_empty_local_level(tmp_path):
    snapshot, request, _ = _snapshot()
    selected = snapshot.select(request)
    coarse = next(item for item in selected if item.key.level == 0)
    fine = next(item for item in selected if item.key.level == 1)
    cases = (
        (
            0,
            coarse,
            replace(fine, pieces=()),
            [0] * 4,
            [0, 8, 0, 0],
            [1.0] * 4,
            (0, 4),
            (4, 4),
        ),
        (
            1,
            replace(coarse, pieces=()),
            replace(
                fine,
                pieces=(replace(fine.pieces[0], owner_rank=1),),
            ),
            [1] * 4,
            [0] * 4,
            [2.0] * 4,
            (0, 0),
            (0, 4),
        ),
    )

    for (
        rank,
        local_coarse,
        local_fine,
        levels,
        ghosts,
        values,
        coarse_range,
        fine_range,
    ) in cases:
        local_snapshot = replace(snapshot, fields=(local_coarse, local_fine))
        local_request = replace(
            request,
            parallel_mode=ParallelMode.PER_RANK,
            rank=rank,
            size=2,
            diagnostics=(),
        )
        prepared = _stage_writer(
            ParaViewWriter(ParallelMode.PER_RANK),
            local_snapshot,
            local_request,
            tmp_path / ("rank-%d.vtu" % rank),
            communicator=object(),
        )
        reopened = read_paraview(prepared.temporary).require_selection(local_request)
        coarse_record = reopened.manifest["datasets"]["fields"][coarse.key.identity.token]
        fine_record = reopened.manifest["datasets"]["fields"][fine.key.identity.token]

        assert coarse_record["name"] == fine_record["name"] == "field_0000"
        assert tuple(coarse_record["cell_range"]) == coarse_range
        assert tuple(fine_record["cell_range"]) == fine_range
        assert np.array_equal(reopened.arrays["pops_level"], levels)
        assert np.array_equal(reopened.arrays["vtkGhostType"], ghosts)
        assert np.array_equal(reopened.arrays["TimeValue"], [0.125])
        assert np.array_equal(reopened.arrays["field_0000"], values)
        prepared.abort_prepare()


def test_paraview_single_file_is_native_reopenable_and_keeps_amr_metadata(tmp_path):
    snapshot, request, foreign = _snapshot()
    target = deterministic_target(tmp_path, "fields", request, snapshot, ".vtu")
    prepared = _stage_writer(ParaViewWriter(), snapshot, request, target)
    reopened = read_paraview(prepared.temporary).require_selection(request)
    assert set(reopened.manifest["datasets"]["fields"]) == {
        key.identity.token for key in request.selection}
    assert foreign.identity.token not in reopened.manifest["datasets"]["fields"]
    assert np.array_equal(reopened.arrays["pops_level"], [0, 0, 0, 0] + [1] * 4)
    assert int(np.sum(reopened.arrays["pops_coverage"])) == 1
    assert len(reopened.manifest["snapshot"]["geometries"]) == 2
    vtk = pytest.importorskip("vtkmodules.vtkIOXML")
    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(str(prepared.temporary))
    reader.UpdateInformation()
    pipeline = pytest.importorskip("vtkmodules.vtkCommonExecutionModel")
    time_steps = pipeline.vtkStreamingDemandDrivenPipeline.TIME_STEPS()
    information = reader.GetOutputInformation(0)
    assert information.Has(time_steps)
    assert information.Length(time_steps) == 1
    assert information.Get(time_steps, 0) == 0.125
    reader.Update()
    grid = reader.GetOutput()
    assert grid.GetNumberOfCells() == 8 and grid.GetNumberOfPoints() == 32
    assert grid.GetPoint(16) == (10.25, 20.25, 0.0)
    assert {grid.GetCellData().GetArrayName(index)
            for index in range(grid.GetCellData().GetNumberOfArrays())} >= {
                "pops_layout", "pops_level", "pops_coverage", "vtkGhostType", "field_0000"}
    assert grid.GetCellData().GetArray("field_0001") is None
    assert [grid.GetCellData().GetArray("field_0000").GetTuple1(index)
            for index in range(8)] == [1.0] * 4 + [2.0] * 4
    assert [grid.GetCellData().GetArray("vtkGhostType").GetTuple1(index)
            for index in range(8)] == [0.0, 8.0, 0.0, 0.0] + [0.0] * 4
    assert grid.GetFieldData().GetArray("TimeValue").GetTuple1(0) == 0.125
    prepared.publish()
    assert read_paraview(target).output_identity == reopened.output_identity


def test_paraview_rejects_inconsistent_logical_field_family_levels(tmp_path):
    snapshot, request, _ = _snapshot()
    selected_fields = snapshot.select(request)
    coarse = next(item for item in selected_fields if item.key.level == 0)
    fine = next(item for item in selected_fields if item.key.level == 1)
    fine_piece = fine.pieces[0]
    variants = {
        "units": replace(fine, units="different-units"),
        "component_names": replace(
            fine,
            component_names=("rho",),
            pieces=(_piece(np.full((1, 2, 2), 2.0), lower=(1, 1)),),
        ),
        "dtype": replace(
            fine,
            pieces=(replace(
                fine_piece,
                values=np.full((2, 2), 2.0, dtype=np.float32),
            ),),
            dtype="<f4",
        ),
        "centering": replace(fine, centering="face_x", global_shape=(4, 5)),
    }
    for attribute, incompatible in variants.items():
        selected = replace(snapshot, fields=(coarse, incompatible))
        target = tmp_path / ("inconsistent-%s.vtu" % attribute)
        with pytest.raises(
                ValueError,
                match="logical field family levels disagree on %s" % attribute):
            _stage_writer(ParaViewWriter(), selected, request, target)
        assert not target.exists()


def test_paraview_preserves_supported_field_dtype(tmp_path):
    snapshot, request, _ = _snapshot()
    key = next(item for item in request.selection if item.level == 0)
    integer = FieldPayload(
        key, "cell", "count", (), (2, 2),
        (ArrayPiece(
            (0, 0), (2, 2), np.asarray([[1, 2], [3, 4]], dtype=np.int32),
            0, 0, False,
        ),))
    integer_snapshot = replace(snapshot, fields=(integer,))
    integer_request = OutputRequest("integer-output", (key,), ParallelMode.SERIAL)
    prepared = _stage_writer(
        ParaViewWriter(), integer_snapshot, integer_request, tmp_path / "integer.vtu")
    reopened = read_paraview(prepared.temporary).require_selection(integer_request)
    record = reopened.manifest["datasets"]["fields"][key.identity.token]
    start, end = record["cell_range"]
    assert reopened.arrays[record["name"]][start:end].dtype.str == "<i4"
    prepared.abort_prepare()


@pytest.mark.parametrize(
    ("component_names", "values", "expected_shape"),
    (
        (("rho",), np.arange(4, dtype=np.float64).reshape(1, 2, 2), (4, 1)),
        (("rho", "energy"), np.arange(8, dtype=np.float64).reshape(2, 2, 2), (4, 2)),
    ),
)
def test_paraview_preserves_declared_component_axis(
        tmp_path, component_names, values, expected_shape):
    snapshot, request, _ = _snapshot()
    key = next(item for item in request.selection if item.level == 0)
    field = FieldPayload(
        key, "cell", "unspecified", component_names, (2, 2),
        (_piece(values),),
    )
    selected_snapshot = replace(snapshot, fields=(field,))
    selected_request = OutputRequest(
        "component-output", (key,), ParallelMode.SERIAL)
    prepared = _stage_writer(
        ParaViewWriter(), selected_snapshot, selected_request,
        tmp_path / ("components-%d.vtu" % len(component_names)),
    )

    reopened = read_paraview(prepared.temporary).require_selection(selected_request)
    record = reopened.manifest["datasets"]["fields"][key.identity.token]
    start, end = record["cell_range"]
    published = reopened.arrays[record["name"]]
    assert published.shape == expected_shape
    assert published[start:end].shape == expected_shape
    assert np.array_equal(published[start:end], values.reshape(len(component_names), -1).T)
    prepared.abort_prepare()


def test_hdf5_is_reopened_with_native_reader_and_exact_selection(tmp_path):
    pytest.importorskip("h5py")
    snapshot, request, foreign = _snapshot()
    missing = replace(request, selection=(replace(request.selection[0], level=9),))
    with pytest.raises(KeyError, match="owner/layout/level/state"):
        _stage_writer(HDF5Writer(), snapshot, missing, tmp_path / "missing.h5")
    assert not list(tmp_path.iterdir())
    target = deterministic_target(tmp_path, "fields", request, snapshot, ".h5")
    prepared = _stage_writer(HDF5Writer(), snapshot, request, target)
    reopened = read_hdf5(prepared.temporary).require_selection(request)
    assert len(reopened.manifest["datasets"]["fields"]) == 2
    assert foreign.identity.token not in reopened.manifest["datasets"]["fields"]
    prepared.publish()
    assert read_hdf5(target).output_identity == reopened.output_identity


def test_hdf5_reader_rejects_hidden_nested_dataset(tmp_path):
    h5py = pytest.importorskip("h5py")
    snapshot, request, _ = _snapshot()
    target = deterministic_target(tmp_path, "hidden", request, snapshot, ".h5")
    prepared = _stage_writer(HDF5Writer(), snapshot, request, target)
    with h5py.File(prepared.temporary, "r+") as output:
        output.create_dataset("fields/evil", data=np.asarray([1.0]))
    with pytest.raises(ValueError, match="datasets/groups differ"):
        read_hdf5(prepared.temporary)
    prepared.abort_prepare()


def test_hdf5_reader_rejects_field_shape_and_sparse_gap_tampering(tmp_path):
    h5py = pytest.importorskip("h5py")
    snapshot, request, _ = _snapshot()

    shape_target = deterministic_target(tmp_path, "shape", request, snapshot, ".h5")
    shape_prepared = _stage_writer(HDF5Writer(), snapshot, request, shape_target)
    with h5py.File(shape_prepared.temporary, "r+") as output:
        del output["fields/0000/values"]
        output.create_dataset("fields/0000/values", shape=(3, 2), dtype=np.float64)
    with pytest.raises(ValueError, match="shape/dtype"):
        read_hdf5(shape_prepared.temporary)
    shape_prepared.abort_prepare()

    gap_target = deterministic_target(tmp_path, "gap", request, snapshot, ".h5")
    gap_prepared = _stage_writer(HDF5Writer(), snapshot, request, gap_target)
    with h5py.File(gap_prepared.temporary, "r+") as output:
        manifest = json.loads(str(output.attrs["pops_output_manifest"]))
        sparse_path = next(
            path
            for path, evidence in manifest["arrays"].items()
            if evidence.get("fill") == "zero-outside-pieces"
            and evidence["shape"] == [4, 4]
        )
        output[sparse_path][0, 0] = 17.0
    with pytest.raises(ValueError, match="outside declared pieces"):
        read_hdf5(gap_prepared.temporary)
    gap_prepared.abort_prepare()


def test_parallel_contract_fails_before_run_without_supported_backend(tmp_path, monkeypatch):
    snapshot, request, _ = _snapshot()
    parallel = replace(
        request, parallel_mode=ParallelMode.COLLECTIVE, rank=0, size=2)
    with pytest.raises((RuntimeError, TypeError), match="h5py|communicator|MPI"):
        _stage_writer(HDF5Writer(ParallelMode.COLLECTIVE),
            snapshot, parallel, tmp_path / "parallel.h5")
    with pytest.raises(ValueError, match="no COLLECTIVE writer"):
        _stage_writer(NPZWriter(ParallelMode.COLLECTIVE),
            snapshot, parallel, tmp_path / "parallel.npz")
    assert not list(tmp_path.iterdir())


def test_collective_hdf5_snapshot_metadata_requires_exact_rank_consensus():
    snapshot, request, _ = _snapshot()
    request = replace(
        request, parallel_mode=ParallelMode.COLLECTIVE, rank=0, size=2)
    selected = snapshot.select(request)
    data = snapshot.to_data(request)
    canonical = dict(data, fields=[dict(field, pieces=[]) for field in data["fields"]])
    local_pieces = {
        field.key.identity.token: [piece.to_data() for piece in field.pieces]
        for field in selected
    }
    envelope = {
        "rank": 0,
        "snapshot": canonical,
        "pieces": local_pieces,
        "preflight": {"target": "/tmp/collective.h5"},
        "error": None,
    }
    peer = deepcopy(envelope)
    peer["rank"] = 1
    peer["snapshot"]["metadata"]["rank-local"] = 1
    with pytest.raises(ValueError, match="metadata differs across ranks"):
        _rebuild_parallel_snapshot_data(snapshot, (envelope, peer), selected)


def test_composite_integrals_requires_exact_native_level_authority():
    snapshot, request, _ = _snapshot()
    from pops.output.data import _NativeCompositeIntegral, _field_family_identity

    with pytest.raises(RuntimeError, match="exact selected level tuple"):
        composite_integrals(snapshot, request)
    family = _field_family_identity(request.selection[0])
    coarse_only = replace(request, selection=(request.selection[0],))
    exact = replace(
        snapshot,
        _native_composite_integrals=(_NativeCompositeIntegral(family, (0, 1), 5.0),),
    )
    with pytest.raises(RuntimeError, match="under-selected, or over-selected"):
        composite_integrals(exact, coarse_only)
    for mismatched_levels in ((0,), (0, 1, 2)):
        mismatched = replace(
            snapshot,
            _native_composite_integrals=(
                _NativeCompositeIntegral(family, mismatched_levels, 7.0),
            ),
        )
        with pytest.raises(RuntimeError, match="exact selected level tuple"):
            composite_integrals(mismatched, request)

    totals = composite_integrals(exact, request)
    assert len(totals) == 1
    # Python exposes the scalar authenticated for exactly levels (0, 1); it never folds arrays.
    assert next(iter(totals.values())) == 5.0


def test_composite_integrals_refuses_non_cartesian_cell_measure():
    from pops.mesh._layout_plan_contracts import POLAR_ANNULUS_CELL_AREA

    snapshot, request, _ = _snapshot()
    snapshot = replace(snapshot, geometries=tuple(
        replace(geometry, cell_measure=POLAR_ANNULUS_CELL_AREA)
        for geometry in snapshot.geometries
    ))
    with pytest.raises(NotImplementedError, match="only the native Cartesian cell-area metric"):
        composite_integrals(snapshot, request)


def test_balance_terms_keep_the_explicit_open_domain_sign_convention():
    balance = BalanceTerms(11.0, 2.0, 5.0, 3.0, 1.0)
    assert balance.residual == 4.0
    assert set(balance.to_data()) == {
        "storage_change", "outward_boundary_flux", "sources", "reflux", "projection", "residual"}


def test_consumer_publisher_adapter_prepares_only_a_resolved_effect(tmp_path):
    snapshot, request, _ = _snapshot()
    target = deterministic_target(tmp_path, "fields", request, snapshot, ".npz")
    effect = _accepted_output_effect(request, target)
    publisher = ConsumerOutputPublisher(lambda accepted: OutputPreparation(
        NPZ(), snapshot, request, target) if accepted is effect else None)
    prepared = publisher.prepare(effect)
    assert isinstance(prepared, PreparedPublication)
    assert prepared.effect_identity == effect.identity
    assert prepared.payload_identity == effect.payload.identity
    assert prepared.temporary.exists() and not target.exists()
    receipt = prepared.publish()
    assert type(receipt) is PublicationReceipt
    assert receipt.effect_identity == effect.identity
    assert receipt.payload_identity == effect.payload.identity
    assert target.exists()
    prepared.discard()
    prepared.discard()
    with pytest.raises(TypeError, match="AcceptedSideEffect"):
        publisher.prepare(object())


def test_rank_local_base_exception_is_consensus_error_before_publish_returns(
    tmp_path, monkeypatch,
):
    module = import_module("pops.runtime._output_publisher")
    snapshot, request, _ = _snapshot()
    request = replace(request, parallel_mode=ParallelMode.ROOT, size=2)
    target = tmp_path / "rank-cancelled.npz"
    effect = _accepted_output_effect(request, target)
    authority = writer_session_authority("rank-cancelled", request, target)

    class RankLocalCancellation(BaseException):
        pass

    class Session:
        def __init__(self):
            self.identity = Identity.from_token(authority["session_identity"])
            self.temporary = None
            self.target = target

        @property
        def authority(self):
            return dict(authority)

        def stage(self):
            return None

        def abort_prepare(self):
            return None

        def publish(self):
            raise RankLocalCancellation("injected rank-local cancellation")

        def rollback(self):
            return None

        def finalize(self):
            return None

    collectives = []

    def gather(_communicator, envelope):
        collectives.append(envelope)
        return envelope, dict(envelope, rank=1, result=None, error=None)

    monkeypatch.setattr(module, "allgather_value", gather)
    prepared = PreparedConsumerOutput(
        effect, Session(), "pops.test.rank-cancelled", request, object())

    with pytest.raises(RuntimeError, match="RankLocalCancellation"):
        prepared.publish()
    assert len(collectives) == 1


def test_rank_without_pieces_keeps_exact_dtype_but_cannot_materialize():
    snapshot, request, _ = _snapshot()
    field = snapshot.select(request)[0]
    empty_rank = replace(field, pieces=(), dtype=field.array_dtype)
    assert empty_rank.array_dtype == field.array_dtype
    with pytest.raises(ValueError, match="owns no pieces"):
        empty_rank.materialize()


def test_native_writer_projection_never_densifies_field_pieces(monkeypatch):
    from pops.runtime._runtime_consumers import _writer_snapshot_data

    snapshot, request, _ = _snapshot()

    def forbidden_materialize(self):
        raise AssertionError("native Writer projection must preserve exact pieces")

    monkeypatch.setattr(FieldPayload, "materialize", forbidden_materialize)
    projected = _writer_snapshot_data(snapshot, request)
    assert len(projected["fields"]) == 2
    assert all(row["pieces"] for row in projected["fields"])
    assert projected["geometries"][0]["coverage"] is snapshot.geometries[0].coverage
    assert projected["geometries"][0]["cell_volumes"] is snapshot.geometries[0].cell_volumes


def test_external_writer_discard_quarantines_a_replaced_temporary_before_callback(tmp_path):
    prepared, native = _external_writer_preparation(tmp_path)
    temporary = prepared.temporary
    temporary.unlink()
    temporary.write_bytes(b"third-party-temporary")

    with pytest.raises(RuntimeError, match="no longer names its runtime-owned inode"):
        prepared.discard()

    assert native.operations == ["verify"]
    assert temporary.read_bytes() == b"third-party-temporary"
    assert not prepared._staging.is_open
    (recovery,) = prepared.recoveries
    assert os.path.samefile(recovery.quarantine_path, temporary)
    prepared.cleanup_recoveries()
    assert _recovery_paths(tmp_path) == ()


def test_external_writer_cleanup_callback_cannot_delete_names_recreated_after_detach(tmp_path):
    prepared, native = _external_writer_preparation(
        tmp_path, recreate_after_detach=True)
    temporary = prepared.temporary
    component = prepared._component_published
    target = prepared.target

    prepared.discard()

    assert temporary.read_bytes() == b"third-party-temporary"
    assert component.read_bytes() == b"third-party-component"
    assert target.read_bytes() == b"third-party-target"
    assert native.operations == ["verify", "discard"]
    (cleanup_request,) = native.cleanup_requests
    cleanup_text = repr(cleanup_request)
    assert str(temporary) not in cleanup_text
    assert str(component) not in cleanup_text
    assert str(target) not in cleanup_text
    cleanup_temporary = Path(cleanup_request["temporary_path"])
    cleanup_component = Path(cleanup_request["published_path"])
    assert cleanup_temporary.parent == cleanup_component.parent
    assert not cleanup_temporary.parent.exists()


def test_external_writer_publish_quarantines_a_replaced_component_path(tmp_path):
    prepared, native = _external_writer_preparation(tmp_path, replace_component=True)
    component = prepared._component_published

    with pytest.raises(RuntimeError, match="did not move its verified inode"):
        prepared.publish()

    assert native.operations == ["verify", "publish"]
    assert component.read_bytes() == b"third-party-component"
    assert not prepared._staging.is_open
    (recovery,) = prepared.recoveries
    assert os.path.samefile(recovery.quarantine_path, component)
    prepared.cleanup_recoveries()
    assert _recovery_paths(tmp_path) == ()


def test_external_writer_publish_quarantines_a_target_replaced_after_link(
    tmp_path, monkeypatch,
):
    prepared, native = _external_writer_preparation(tmp_path)
    target = prepared.target
    real_link = os.link
    raced = False

    def replace_target_after_link(source, destination, *args, **kwargs):
        nonlocal raced
        result = real_link(source, destination, *args, **kwargs)
        if not raced and Path(destination) == target and kwargs.get("dst_dir_fd") is None:
            raced = True
            target.unlink()
            target.write_bytes(b"third-party-target")
        return result

    monkeypatch.setattr(os, "link", replace_target_after_link)
    with pytest.raises(RuntimeError, match="public target does not name"):
        prepared.publish()

    assert native.operations == ["verify", "publish"]
    assert target.read_bytes() == b"third-party-target"
    assert not prepared._staging.is_open
    (recovery,) = prepared.recoveries
    assert os.path.samefile(recovery.quarantine_path, target)
    prepared.cleanup_recoveries()
    assert _recovery_paths(tmp_path) == ()


def test_external_writer_finalize_releases_fd_without_removing_accepted_artifact(tmp_path):
    prepared, native = _external_writer_preparation(tmp_path)
    staging = prepared._staging
    receipt = prepared.publish()

    assert receipt.artifact_id
    assert staging.is_open
    assert prepared.finalize() is None
    assert not staging.is_open
    assert prepared.target.read_bytes() == b"native-writer-output"
    with pytest.raises(RuntimeError, match="finalized native Writer"):
        prepared.rollback()
    assert native.operations == ["verify", "publish"]
    assert _recovery_paths(tmp_path) == ()


def test_format_interface_selects_exact_writer():
    assert type(NPZ().writer()) is NPZWriter
    assert type(HDF5().writer()) is HDF5Writer


def test_format_writers_publish_structural_preflight_capabilities():
    context = ExecutionContext(
        backend=proven_serial_manifest(
            backend="production", target="system", abi="test|c++|c++23", runtime=True),
        communicator=ExecutionResource("communicator", "serial"),
        datatype=ExecutionResource("datatype", "float64"),
        device=ExecutionResource("device", "host"),
    )
    for writer, provider in (
        (NPZ().writer(), "pops.output.npz.v1"),
        (ParaView().writer(), "pops.output.paraview-vtu.v1"),
    ):
        capability = writer.preflight(context)
        assert capability == {
            "schema_version": 1,
            "provider_id": provider,
            "parallel_mode": "serial",
            "communicator": "serial",
            "size": 1,
        }


def test_custom_format_writer_must_implement_structural_preflight():
    class Provider:
        __pops_ir_immutable__ = True

        @staticmethod
        def consumer_data():
            return {
                "schema_version": 1,
                "provider_id": "pops.test.output.v1",
                "extension": ".test",
                "parallel_mode": "serial",
            }

        @staticmethod
        def writer():
            return type("Writer", (), {"prepare_session": lambda self: None})()

    snapshot, request, _ = _snapshot()
    provider = Provider()
    effect = _accepted_output_effect(request, "broken.test")
    effect = replace(effect, target=PublicationTarget(
        "broken.test", provider.consumer_data(), None, ParallelMode.SERIAL))
    publisher = ConsumerOutputPublisher(lambda accepted: OutputPreparation(
        provider, snapshot, request, accepted.target.uri))
    with pytest.raises(
        RuntimeError,
        match=r"must implement preflight\(\) and prepare_session\(\)",
    ):
        publisher.prepare(effect)


def test_provider_missing_required_key_is_rejected_even_with_extra_state():
    class Provider:
        __pops_ir_immutable__ = True

        @staticmethod
        def consumer_data():
            return {
                "schema_version": 1,
                "provider_id": "pops.test.missing-extension.v1",
                "parallel_mode": "serial",
                "extra": "cannot-mask-a-missing-required-key",
            }

        @staticmethod
        def writer():
            return object()

    with pytest.raises(ValueError, match=r"lacks required keys \['extension'\]"):
        consumer_format_data(Provider())


@pytest.mark.parametrize(
    ("selection_contract", "error_type", "message"),
    (
        ({"schema_version": 1}, TypeError, "selection_contract has an unknown schema"),
        (
            {"schema_version": 2, "layout_cardinality": "single"},
            ValueError,
            "selection_contract schema_version must be 1",
        ),
        (
            {"schema_version": 1, "layout_cardinality": "sometimes"},
            ValueError,
            "layout_cardinality must be single or multiple",
        ),
    ),
)
def test_format_selection_contract_is_closed_and_typed(
    selection_contract, error_type, message,
):
    class Provider:
        __pops_ir_immutable__ = True

        @staticmethod
        def consumer_data():
            return {
                "schema_version": 1,
                "provider_id": "pops.test.selection-contract.v1",
                "extension": ".test",
                "parallel_mode": "serial",
                "selection_contract": selection_contract,
            }

        @staticmethod
        def writer():
            return object()

    with pytest.raises(error_type, match=message):
        consumer_format_data(Provider())


def test_custom_writer_session_uses_only_the_public_structural_protocol(tmp_path):
    snapshot, request, _ = _snapshot()
    target = tmp_path / "custom.test"

    class Session:
        def __init__(self):
            self._authority = writer_session_authority("custom-test", request, target)
            self._identity = Identity.from_token(self._authority["session_identity"])
            self.temporary = target.with_name(".custom.test.prepared")
            self.target = target
            self._created = False

        @property
        def authority(self):
            return dict(self._authority)

        @property
        def identity(self):
            return self._identity

        def stage(self):
            self.temporary.write_bytes(b"public-structural-session")

        def abort_prepare(self):
            self.temporary.unlink(missing_ok=True)

        def publish(self):
            os.link(self.temporary, self.target)
            self._created = True
            self.temporary.unlink()
            return OutputPublicationReceipt(
                self.target,
                "custom-test",
                make_identity("scientific-output", {"bytes": "public-structural-session"}),
                request.publication_identity,
            )

        def rollback(self):
            self.temporary.unlink(missing_ok=True)
            if self._created:
                self.target.unlink(missing_ok=True)

        def finalize(self):
            return None

    class Writer:
        @staticmethod
        def preflight(_context):
            return {"schema_version": 1, "writer": "custom-test"}

        @staticmethod
        def prepare_session(_snapshot, _request, _target, *, communicator=None):
            assert communicator is None
            assert not target.exists()
            return Session()

    class Provider:
        __pops_ir_immutable__ = True

        @staticmethod
        def consumer_data():
            return {
                "schema_version": 1,
                "provider_id": "pops.test.public-session.v1",
                "extension": ".test",
                "parallel_mode": "serial",
            }

        @staticmethod
        def writer():
            return Writer()

    provider = Provider()
    effect = replace(
        _accepted_output_effect(request, target),
        target=PublicationTarget(
            str(target), provider.consumer_data(), None, ParallelMode.SERIAL),
    )
    prepared = ConsumerOutputPublisher(
        lambda _accepted: OutputPreparation(provider, snapshot, request, target)
    ).prepare(effect)
    assert prepared.temporary.is_file() and not target.exists()
    receipt = prepared.publish()
    assert receipt.parallel_mode is ParallelMode.SERIAL
    assert target.read_bytes() == b"public-structural-session"
    prepared.rollback()
    assert not target.exists()


def test_stage_abort_recovery_is_registered_on_runtime_instance(tmp_path):
    snapshot, request, _ = _snapshot()
    target = tmp_path / "stage-recovery.test"

    class Session:
        def __init__(self):
            self._authority = writer_session_authority(
                "stage-recovery", request, target)
            self.identity = Identity.from_token(self._authority["session_identity"])
            self.temporary = tmp_path / ".stage-recovery.prepared"
            self.target = target
            self._owner = None
            self._recoveries = ()

        @property
        def authority(self):
            return dict(self._authority)

        @property
        def recoveries(self):
            return self._recoveries

        def stage(self):
            self.temporary.write_bytes(b"runtime-owned")
            staged = self.temporary.lstat()
            self._owner = (int(staged.st_dev), int(staged.st_ino))
            replacement = self.temporary.with_name(".stage-recovery.third-party")
            replacement.write_bytes(b"third-party")
            os.replace(replacement, self.temporary)
            raise RuntimeError("injected stage failure")

        def abort_prepare(self):
            try:
                _StagedOutputFile._quarantine_owned_path(
                    self.temporary,
                    self._owner,
                    replaced_message="stage abort refused replacement",
                )
            except _OutputRecoveryRequired as error:
                self._recoveries = (error.recovery,)
                raise RuntimeError("stage abort retained recovery") from error

        def publish(self):
            raise AssertionError("failed stage cannot publish")

        def rollback(self):
            return None

        def finalize(self):
            return None

    session = Session()

    class Writer:
        @staticmethod
        def preflight(_context):
            return {"schema_version": 1, "writer": "stage-recovery"}

        @staticmethod
        def prepare_session(_snapshot, _request, _target, *, communicator=None):
            assert communicator is None
            return session

    class Provider:
        __pops_ir_immutable__ = True

        @staticmethod
        def consumer_data():
            return {
                "schema_version": 1,
                "provider_id": "pops.test.stage-recovery.v1",
                "extension": ".test",
                "parallel_mode": "serial",
            }

        @staticmethod
        def writer():
            return Writer()

    provider = Provider()
    effect = replace(
        _accepted_output_effect(request, target),
        target=PublicationTarget(
            str(target), provider.consumer_data(), None, ParallelMode.SERIAL),
    )
    runtime = object.__new__(RuntimeInstance)
    runtime._consumer_recoveries = {}

    with pytest.raises(RuntimeError, match="stage abort retained recovery"):
        ConsumerOutputPublisher(
            lambda _effect: OutputPreparation(provider, snapshot, request, target),
            retain_recoveries=runtime._retain_output_recoveries,
        ).prepare(effect)

    (record,) = runtime.consumer_recoveries
    assert record.public_path == session.temporary
    assert record.quarantine_path.is_file()
    runtime.restore_consumer_recovery(record.recovery_id)
    assert session.temporary.read_bytes() == b"third-party"
    runtime.cleanup_consumer_recovery(record.recovery_id)
    assert runtime.consumer_recoveries == ()
    assert not record.quarantine_path.exists()








def test_parallel_modes_bind_request_target_and_receipt_identities(tmp_path):
    snapshot, serial, _ = _snapshot()
    root0 = replace(
        serial, parallel_mode=ParallelMode.ROOT, rank=0, size=2)
    root1 = replace(root0, rank=1)
    assert root0.identity != root1.identity
    assert root0.publication_identity == root1.publication_identity
    assert root0.publication_data()["ranks"] == [0, 1]
    assert "rank" not in root0.publication_data()

    rank0 = replace(
        serial, parallel_mode=ParallelMode.PER_RANK, rank=0, size=2)
    rank1 = replace(rank0, rank=1)
    assert rank0.publication_identity != rank1.publication_identity
    target0 = deterministic_target(tmp_path, "pieces", rank0, snapshot, ".npz")
    target1 = deterministic_target(tmp_path, "pieces", rank1, snapshot, ".npz")
    assert target0 != target1
    assert "r000000" in target0.name and "r000001" in target1.name

    with pytest.raises(ValueError, match="rank 0 / size 1"):
        replace(serial, size=2)
    with pytest.raises(TypeError, match="exact pops.output.ParallelMode"):
        HDF5(mode="collective")
    with pytest.raises(ValueError, match="does not support collective"):
        NPZ(mode=ParallelMode.COLLECTIVE)
    assert ParaView(mode=ParallelMode.PER_RANK).mode is ParallelMode.PER_RANK


def test_per_rank_publication_receipt_is_one_deterministic_rank_set():
    effect = _identity("accepted-side-effect", "per-rank")
    payload = _identity("consumer-payload", "per-rank")
    rows = ((0, "artifact-r0"), (1, "artifact-r1"))
    receipt = PublicationReceipt(
        effect,
        payload,
        "test-per-rank",
        _identity("scientific-output-artifact-set", {"rows": rows}).token,
        ParallelMode.PER_RANK,
        rows,
    )
    assert receipt.rank_artifacts == rows
    assert receipt.to_data()["parallel_mode"] == "per_rank"
    with pytest.raises(ValueError, match="every contiguous rank"):
        replace(receipt, rank_artifacts=((0, "artifact-r0"), (2, "artifact-r2")))
