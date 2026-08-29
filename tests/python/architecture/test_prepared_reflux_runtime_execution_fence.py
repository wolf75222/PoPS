"""Exact-ND AMR reflux has one prepared spatial/runtime authority."""

from pathlib import Path

from tests.python.architecture.test_final_nd_amr_consumers import (
    ROOTS as AMR_CONSUMER_ROOTS,
    _semantic_closure as _amr_semantic_closure,
    _source as _amr_semantic_source,
)


ROOT = Path(__file__).resolve().parents[3]
PATCH_RANGE = (
    ROOT / "include" / "pops" / "numerics" / "time" / "amr" / "levels" / "amr_patch_range.hpp"
)
LEDGER = ROOT / "include" / "pops" / "amr" / "reflux" / "face_flux_ledger.hpp"
METRIC_REFLUX = LEDGER.with_name("metric_reflux.hpp")
AMR_RUNTIME = ROOT / "include" / "pops" / "runtime" / "amr" / "amr_runtime.hpp"
PROGRAM_EXECUTION_SERVICES = (
    ROOT / "include" / "pops" / "runtime" / "program" / "program_execution_services.hpp"
)
RETIRED_PROGRAM_REFLUX = ROOT / "include" / "pops" / "runtime" / "amr" / "amr_program_reflux.hpp"
HEADERS_MANIFEST = ROOT / "include" / "pops_headers.manifest"


def test_edge_strip_program_reflux_facade_is_retired() -> None:
    assert not RETIRED_PROGRAM_REFLUX.exists()
    manifest = HEADERS_MANIFEST.read_text(encoding="utf-8")
    assert "pops/runtime/amr/amr_program_reflux.hpp" not in manifest


def test_subcycle_transition_routes_only_through_the_live_ranked_runtime() -> None:
    source = _amr_semantic_source(_amr_semantic_closure(AMR_CONSUMER_ROOTS["subcycling"]))
    assert "class PreparedAmrSubcycleTransition" in source
    assert "class PreparedAmrSubcyclePlan" in source
    assert "template <int Dim" in source
    assert "CoarseFineInterface<Dim>" in source
    assert "runtime.reconcile_reflux(" in source
    assert "TransactionalFaceFluxLedger<Dim, Payload>" in source
    assert "EdgeStrip" not in source
    assert "Box2D" not in source
    assert "Fx" not in source
    assert "Fy" not in source


def test_patch_range_is_one_rank_generic_algorithm() -> None:
    source = PATCH_RANGE.read_text(encoding="utf-8")
    assert "class PatchRange" in source
    assert "PatchRange(Box<Dim> fine" in source
    assert "RefinementRatio<Dim> ratio" in source
    assert "for (const Box<Dim>& fine" in source
    assert "CoarseFineInterfaceIdentity<Dim>" in source
    assert "Box2D" not in source


def test_face_ledger_is_transactional_axis_qualified_and_bounded() -> None:
    source = LEDGER.read_text(encoding="utf-8")
    assert "class TransactionalFaceFluxLedger" in source
    assert "std::array<std::vector<Entry>, Dim>" in source
    assert "void begin(std::uint64_t attempt)" in source
    assert "void commit()" in source
    assert "void rollback()" in source
    assert "FaceFluxLedgerBudget" in source
    assert "attempt <= *last_closed_attempt_" in source
    assert "max_pending_entries" in source
    assert "max_published_entries" in source


def test_metric_reflux_authenticates_exact_time_and_tangential_faces() -> None:
    source = METRIC_REFLUX.read_text(encoding="utf-8")
    assert "AuthenticatedWindow" in source
    assert "authenticated_window(" in source
    assert "validate_temporal_coverage(" in source
    assert "expected_fine_face_set" in source
    assert "for (int direction = 0; direction < Dim; ++direction)" in source
    assert "metric_reflux(" in source


def test_program_execution_services_and_runtime_share_the_same_prepared_authority() -> None:
    assert PROGRAM_EXECUTION_SERVICES.is_file()
    context = _amr_semantic_source(_amr_semantic_closure(AMR_CONSUMER_ROOTS["program"]))
    runtime = AMR_RUNTIME.read_text(encoding="utf-8")
    # The public root delegates the AMR implementation to its authenticated private backend; the
    # semantic closure is the source-of-truth for the prepared hierarchy type.
    assert "PreparedAmrSubcyclePlan<Dim" in context
    assert "prepare_subcycling(" in context
    assert "runtime_->reconcile_reflux(" in context
    assert "TransactionalFaceFluxLedger<Dim, Payload>" in context
    assert "MetricFaceReflux<Payload> reconcile_reflux(" in runtime
    assert "metric_reflux(ledger, key, ratio, mapping, budget" in runtime
