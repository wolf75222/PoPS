"""ADC-686 exact scientific formats, selection, metadata and transaction protocol."""
from __future__ import annotations

import json
import os
from copy import deepcopy
from dataclasses import replace

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
from pops.output._writers.hdf5 import _rebuild_parallel_snapshot_data
from pops.output.provider import consumer_format_data
from pops.runtime._consumer import (
    AcceptedSideEffect, ConsumerPayload, PreparedPublication, PublicationReceipt, PublicationTarget,
)
from pops.runtime._output_publisher import (
    ConsumerOutputPublisher,
    OutputPreparation,
)


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
    reader.Update()
    grid = reader.GetOutput()
    assert grid.GetNumberOfCells() == 8 and grid.GetNumberOfPoints() == 32
    assert grid.GetPoint(16) == (10.25, 20.25, 0.0)
    assert {grid.GetCellData().GetArrayName(index)
            for index in range(grid.GetCellData().GetNumberOfArrays())} >= {
        "pops_layout", "pops_level", "pops_coverage", "field_0000", "field_0001"}
    prepared.publish()
    assert read_paraview(target).output_identity == reopened.output_identity


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


def test_composite_reduction_uses_metrics_and_excludes_covered_coarse_cells():
    snapshot, request, _ = _snapshot()
    totals = composite_integrals(snapshot, request)
    assert len(totals) == 1
    # coarse: 3 uncovered cells * 1; fine: 4 valid box cells * value 2 * volume 0.25
    assert next(iter(totals.values())) == 5.0
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
