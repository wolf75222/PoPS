"""ADC-686 exact scientific formats, selection, metadata and transaction protocol."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest

from pops.identity import make_identity
from pops.model import Handle, OwnerKind, OwnerPath
from pops.output import (
    ArrayPiece, BalanceTerms, DiagnosticKey, DiagnosticPayload,
    FieldKey, FieldPayload,
    HDF5, HDF5Writer, LevelGeometry, NPZ, NPZWriter, OutputClock,
    OutputProvenance, OutputRequest, OutputSnapshot,
    ParaViewWriter, composite_integrals, deterministic_target, read_hdf5,
    read_npz, read_paraview,
)
from pops.output._consumer_contracts import FailRun, ParallelMode, ScheduleCursor
from pops.output._writers import hdf5 as hdf5_backend
from pops.output._writers.hdf5 import _parallel_snapshot_data
from pops.runtime._consumer import (
    AcceptedSideEffect, ConsumerPayload, PreparedPublication, PublicationReceipt, PublicationTarget,
)
from pops.runtime._output_publisher import ConsumerOutputPublisher, OutputPreparation


def _identity(domain, name):
    return make_identity(domain, {"name": name})


def _handle(block, name="rho"):
    owner = OwnerPath.case("case").child(OwnerKind.BLOCK, block)
    return Handle(name, kind="state", owner=owner)


def _piece(values):
    values = np.asarray(values, dtype=np.float64)
    return ArrayPiece((0, 0), values.shape[-2:], values)


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
                     (_piece(np.full((4, 4), fine_value)),)),
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
        "density-output", (coarse_key, fine_key), False, diagnostics=(diagnostic_key,))
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


def test_npz_prepare_reopen_publish_is_exact_and_deterministic(tmp_path):
    snapshot, request, foreign = _snapshot()
    target = deterministic_target(tmp_path, "fields", request, snapshot, ".npz")
    prepared = NPZWriter().prepare(snapshot, request, target)
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
    prepared.discard()
    prepared.discard()
    assert target.is_file()
    assert deterministic_target(tmp_path, "fields", request, snapshot, ".npz") == target
    read_npz(target).require_selection(request)


def test_npz_collision_and_discard_never_publish_partial_content(tmp_path):
    snapshot, request, _ = _snapshot()
    target = deterministic_target(tmp_path, "fields", request, snapshot, ".npz")
    NPZWriter().prepare(snapshot, request, target).publish()
    changed, _, _ = _snapshot(fine_value=3.0)
    competing = NPZWriter().prepare(changed, request, target)
    with pytest.raises(FileExistsError, match="collision"):
        competing.publish()
    competing.discard()
    competing.discard()
    assert not competing.temporary.exists()
    first = read_npz(target)
    fine = next(key for key in request.selection if key.level == 1)
    fine_name = first.manifest["datasets"]["fields"][fine.identity.token]
    assert np.all(first.arrays[fine_name] == 2.0)


def test_npz_publication_rollback_removes_only_the_artifact_it_created(tmp_path):
    snapshot, request, _ = _snapshot()
    target = deterministic_target(tmp_path, "fields", request, snapshot, ".npz")
    created = NPZWriter().prepare(snapshot, request, target)
    created.publish()
    created.rollback()
    created.rollback()
    assert not target.exists()

    NPZWriter().prepare(snapshot, request, target).publish()
    idempotent = NPZWriter().prepare(snapshot, request, target)
    idempotent.publish()
    idempotent.rollback()
    assert target.is_file()
    read_npz(target).require_selection(request)


def test_missing_exact_level_state_selection_fails_before_a_writer(tmp_path):
    snapshot, request, _ = _snapshot()
    missing_values = (
        replace(request.selection[0], level=4),
        replace(request.selection[0], state_id="trial"),
        replace(request.selection[0], layout_identity=_identity("layout-plan", "foreign")),
    )
    for missing in missing_values:
        bad = OutputRequest("density-output", (missing,), False)
        with pytest.raises(KeyError, match="owner/layout/level/state"):
            NPZWriter().prepare(snapshot, bad, tmp_path / "missing.npz")
    assert not (tmp_path / "missing.npz").exists()


def test_diagnostic_only_output_keeps_explicit_layout_and_balance_terms(tmp_path):
    snapshot, request, _ = _snapshot()
    diagnostics_only = replace(snapshot, fields=())
    diagnostic_request = OutputRequest(
        "balance-output", (), False, diagnostics=request.diagnostics)
    target = tmp_path / "balance.npz"
    prepared = NPZWriter().prepare(diagnostics_only, diagnostic_request, target)
    reopened = read_npz(prepared.temporary).require_selection(diagnostic_request)
    assert reopened.manifest["datasets"]["fields"] == {}
    assert len(reopened.manifest["snapshot"]["geometries"]) == 2
    assert len(reopened.manifest["snapshot"]["diagnostics"]) == 1
    prepared.discard()


def test_paraview_geometry_is_byte_exact_and_keeps_row_major_cell_order(tmp_path):
    snapshot, request, foreign = _snapshot()
    target = deterministic_target(tmp_path, "fields", request, snapshot, ".vtu")
    prepared = ParaViewWriter().prepare(snapshot, request, target)
    duplicate = ParaViewWriter().prepare(snapshot, request, tmp_path / "duplicate.vtu")
    assert prepared.temporary.read_bytes() == duplicate.temporary.read_bytes()
    duplicate.discard()
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
    prepared.discard()


def test_paraview_single_file_is_native_reopenable_and_keeps_amr_metadata(tmp_path):
    snapshot, request, foreign = _snapshot()
    target = deterministic_target(tmp_path, "fields", request, snapshot, ".vtu")
    prepared = ParaViewWriter().prepare(snapshot, request, target)
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
        (ArrayPiece((0, 0), (2, 2), np.asarray([[1, 2], [3, 4]], dtype=np.int32)),))
    integer_snapshot = replace(snapshot, fields=(integer,))
    integer_request = OutputRequest("integer-output", (key,), False)
    prepared = ParaViewWriter().prepare(
        integer_snapshot, integer_request, tmp_path / "integer.vtu")
    reopened = read_paraview(prepared.temporary).require_selection(integer_request)
    record = reopened.manifest["datasets"]["fields"][key.identity.token]
    start, end = record["cell_range"]
    assert reopened.arrays[record["name"]][start:end].dtype.str == "<i4"
    prepared.discard()


def test_hdf5_is_reopened_with_native_reader_and_exact_selection(tmp_path):
    pytest.importorskip("h5py")
    snapshot, request, foreign = _snapshot()
    missing = replace(request, selection=(replace(request.selection[0], level=9),))
    with pytest.raises(KeyError, match="owner/layout/level/state"):
        HDF5Writer().prepare(snapshot, missing, tmp_path / "missing.h5")
    assert not list(tmp_path.iterdir())
    target = deterministic_target(tmp_path, "fields", request, snapshot, ".h5")
    prepared = HDF5Writer().prepare(snapshot, request, target)
    reopened = read_hdf5(prepared.temporary).require_selection(request)
    assert len(reopened.manifest["datasets"]["fields"]) == 2
    assert foreign.identity.token not in reopened.manifest["datasets"]["fields"]
    prepared.publish()
    assert read_hdf5(target).output_identity == reopened.output_identity


def test_parallel_contract_fails_before_run_without_supported_backend(tmp_path, monkeypatch):
    snapshot, request, _ = _snapshot()
    parallel = replace(request, parallel=True)
    with pytest.raises((RuntimeError, TypeError), match="h5py|communicator|MPI"):
        HDF5Writer().prepare(snapshot, parallel, tmp_path / "parallel.h5")
    with pytest.raises(ValueError, match="no collective writer"):
        NPZWriter().prepare(snapshot, parallel, tmp_path / "parallel.npz")
    assert not list(tmp_path.iterdir())


def _fake_parallel_h5py(monkeypatch):
    class Version:
        hdf5_version = "test-hdf5"

    class H5py:
        __version__ = "test-h5py"
        version = Version()

    monkeypatch.setattr(hdf5_backend, "_require_h5py", lambda parallel: H5py())


def test_collective_hdf5_snapshot_metadata_requires_exact_rank_consensus(monkeypatch):
    _fake_parallel_h5py(monkeypatch)
    snapshot, request, _ = _snapshot()
    request = replace(request, parallel=True)

    class MetadataMismatch:
        @staticmethod
        def Get_rank():
            return 0

        @staticmethod
        def bcast(value, root=0):
            return value

        @staticmethod
        def Barrier():
            return None

        @staticmethod
        def allgather(envelope):
            peer = deepcopy(envelope)
            peer["snapshot"]["metadata"]["rank-local"] = 1
            return [envelope, peer]

    with pytest.raises(ValueError, match="metadata differs across ranks"):
        _parallel_snapshot_data(snapshot, request, "collective.h5", MetadataMismatch())


def test_collective_hdf5_rank_local_selection_error_enters_consensus(monkeypatch):
    _fake_parallel_h5py(monkeypatch)
    snapshot, request, _ = _snapshot()
    request = replace(request, parallel=True)

    class BrokenLocalSnapshot:
        @staticmethod
        def to_data(output_request):
            return snapshot.to_data(output_request)

        @staticmethod
        def select(_output_request):
            raise ValueError("rank-local field selection is malformed")

    class ErrorConsensus:
        called = False

        @staticmethod
        def Get_rank():
            return 0

        @staticmethod
        def bcast(value, root=0):
            return value

        @staticmethod
        def Barrier():
            return None

        @classmethod
        def allgather(cls, envelope):
            cls.called = True
            assert envelope["snapshot"] is None
            assert "rank-local field selection is malformed" in envelope["error"]
            return [envelope, deepcopy(envelope)]

    with pytest.raises(ValueError, match="snapshot preparation failed across ranks"):
        _parallel_snapshot_data(
            BrokenLocalSnapshot(), request, "collective.h5", ErrorConsensus()
        )
    assert ErrorConsensus.called, "the local error escaped before the collective envelope"


def test_collective_hdf5_refuses_divergent_target_identity(monkeypatch, tmp_path):
    _fake_parallel_h5py(monkeypatch)
    snapshot, request, _ = _snapshot()
    request = replace(request, parallel=True)

    class TargetMismatch:
        @staticmethod
        def Get_rank():
            return 0

        @staticmethod
        def bcast(value, root=0):
            return value

        @staticmethod
        def Barrier():
            return None

        @staticmethod
        def allgather(envelope):
            peer = deepcopy(envelope)
            peer["preflight"]["target"] += ".rank-one"
            return [envelope, peer]

    with pytest.raises(ValueError, match="preflight differs across ranks"):
        _parallel_snapshot_data(
            snapshot, request, tmp_path / "collective.h5", TargetMismatch()
        )


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
