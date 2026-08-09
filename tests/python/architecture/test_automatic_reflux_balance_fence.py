"""Accepted AMR reflux evidence is exact, transactional, and restartable."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
LEDGER = ROOT / "include" / "pops" / "amr" / "reflux" / "face_flux_ledger.hpp"
METRIC_REFLUX = LEDGER.with_name("metric_reflux.hpp")
CHECKPOINT = ROOT / "include" / "pops" / "runtime" / "program" / "amr_program_checkpoint.hpp"
SUBCYCLING = (
    ROOT / "include" / "pops" / "numerics" / "time" / "amr" / "levels" / "amr_subcycling.hpp"
)
RETIRED_PROGRAM_REFLUX = ROOT / "include" / "pops" / "runtime" / "amr" / "amr_program_reflux.hpp"


def test_pending_fragments_never_cross_the_accepted_boundary() -> None:
    ledger = LEDGER.read_text(encoding="utf-8")
    checkpoint = CHECKPOINT.read_text(encoding="utf-8")
    assert "active_attempt_" in ledger
    assert "savepoints_" in ledger
    assert "pending_" in ledger
    assert "published_" in ledger
    assert "if (ledger.in_transaction())" in checkpoint
    assert "cannot observe an active face-flux transaction" in checkpoint
    assert "ledger.published_entries(axis)" in checkpoint
    assert "pending_entries" not in checkpoint


def test_checkpoint_persists_exact_ranked_face_fragments_not_2d_strips() -> None:
    source = CHECKPOINT.read_text(encoding="utf-8")
    assert "template <int Dim>\nstruct AmrProgramAcceptedState" in source
    assert "FaceFluxFragment<Dim, AmrProgramFacePayload>" in source
    assert "std::array<std::vector" in source
    assert "write_face_fragment" in source
    assert "read_face_fragment" in source
    assert "write_rational" in source
    assert "write_clock" in source
    for retired in ("EdgeStrip", "EdgeFlux", "Box2D", "ConstArray4", "ring_flux"):
        assert retired not in source


def test_checkpoint_is_canonical_dimension_qualified_and_budgeted() -> None:
    source = CHECKPOINT.read_text(encoding="utf-8")
    assert "out.i32(Dim);" in source
    assert "native dimension does not match the artifact" in source
    assert "std::sort(destination.begin(), destination.end()" in source
    assert "face fragments must be uniquely ordered" in source
    assert "FaceFluxLedgerBudget budget" in source
    assert "restore_amr_program_face_flux_ledger" in source
    assert "ledger.rollback();" in source


def test_restart_and_collective_preflight_fail_closed_before_publication() -> None:
    source = CHECKPOINT.read_text(encoding="utf-8")
    assert "require_live_amr_program_checkpoint" in source
    assert "runtime.spatial_contract()" in source
    assert "runtime.topology_epoch()" in source
    assert "runtime.materialization_generation()" in source
    assert "require_collective_amr_program_checkpoint_consensus" in source
    assert "all_ranks_agree_exact_ordered_byte_pairs" in source
    assert "differs between communicator ranks" in source


def test_metric_reconciliation_authenticates_the_complete_substep_window() -> None:
    metric = METRIC_REFLUX.read_text(encoding="utf-8")
    transition = SUBCYCLING.read_text(encoding="utf-8")
    assert "substeps do not form a contiguous exact clock partition" in metric
    assert "stage weights do not close one accepted substep" in metric
    assert "coarse and fine physical clocks do not cover the same window" in metric
    assert "has an incomplete " in metric
    assert "runtime.reconcile_reflux(" in transition


def test_no_parallel_program_reflux_authority_remains() -> None:
    assert not RETIRED_PROGRAM_REFLUX.exists()
