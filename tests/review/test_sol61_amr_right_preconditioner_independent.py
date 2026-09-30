"""Source/host acceptance and mutants; no native field/MPI reception."""

import importlib.util
from pathlib import Path
import re
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "independent_right_probe", Path(__file__).with_name("sol61_amr_right_preconditioner_probe.py")
)
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)
WORKSPACE = ROOT / "include/pops/numerics/elliptic/interface/amr_field_newton_krylov.hpp"
PREPARED = ROOT / "include/pops/runtime/program/prepared_amr_field_residual.hpp"


def test_actual_right_gmres_diagonal_and_lane_authority(tmp_path):
    result = probe.run_probe(ROOT, tmp_path)
    assert result["checks"] == 36
    assert result["foreign_lane_refused"]
    assert result["foreign_validation_lane"] == 1
    assert "no Kokkos/MPI/AMR" in result["scope"]


@pytest.mark.parametrize(
    "attack",
    [
        "arnoldi_not_preconditioned",
        "correction_not_preconditioned",
        "wrong_true_jvp",
        "forgot_affine_zero",
        "force_positive_diagonal",
        "ignore_inactive_coverage",
    ],
)
def test_source_phase_mutants_cannot_satisfy_independent_host_equations(tmp_path, attack):
    workspace, prepared = WORKSPACE.read_text(), PREPARED.read_text()
    old, new = {
        "arnoldi_not_preconditioned": (
            "apply_jvp(iterate, image_, work_, nonlinear_iteration)",
            "apply_jvp(iterate, basis_[static_cast<std::size_t>(column)], work_, nonlinear_iteration)",
        ),
        "correction_not_preconditioned": (
            "saxpy_(correction_, coefficients_[static_cast<std::size_t>(index)], image_)",
            "saxpy_(correction_, coefficients_[static_cast<std::size_t>(index)], basis_[static_cast<std::size_t>(index)])",
        ),
        "wrong_true_jvp": (
            "apply_jvp(iterate, correction_, image_, nonlinear_iteration)",
            "copy_(correction_, image_)",
        ),
        "forgot_affine_zero": (
            "value(cell, component) - zero(cell, component)",
            "value(cell, component)",
        ),
        "force_positive_diagonal": (
            "const Real reciprocal = Real(1) / diagonal",
            "const Real reciprocal = Real(1) / std::abs(diagonal)",
        ),
        "ignore_inactive_coverage": (
            "if (active(cell, 0) < Real(.5)) return Real(0);",
            "if (false) return Real(0);",
        ),
    }[attack]
    assert (old in workspace) != (old in prepared)
    workspace, prepared = workspace.replace(old, new), prepared.replace(old, new)
    with pytest.raises(subprocess.CalledProcessError):
        probe.run_probe(ROOT, tmp_path, workspace_source=workspace, prepared_source=prepared)


def test_foreign_lane_mutant_fails_the_explicit_authority_countercase(tmp_path):
    prepared = PREPARED.read_text()
    old = "&lane != &op_->original_field_execution_lane() ||"
    assert old in prepared
    result = probe.run_probe(ROOT, tmp_path, prepared_source=prepared.replace(old, "", 1))
    assert not result["foreign_lane_refused"]


def test_argument_lane_vote_mutant_cannot_preserve_prepared_collective_context(tmp_path):
    prepared = PREPARED.read_text()
    assert "local_phase_(authority_lane," in prepared
    changed = prepared.replace("local_phase_(authority_lane,", "local_phase_(lane,", 1)
    result = probe.run_probe(ROOT, tmp_path, prepared_source=changed)
    assert result["foreign_lane_refused"]
    assert result["foreign_validation_lane"] == 2


def test_full_global_stored_dof_order_and_collective_phases_are_not_local_rank_shortcuts():
    text = PREPARED.read_text()
    phase = probe.method(text, "void prepare_spatial_jacobi_(")
    assert "candidate_[level].layout().size()" in phase
    assert "candidate_[level].local_size()" not in phase
    assert phase.index("require_authority(authority_, lane);") < phase.index(
        "apply_original_field_operator(candidate_, minus_)"
    )
    assert phase.index("apply_original_field_operator(candidate_, plus_)") < phase.index(
        "if (!candidate_[level].contains_local(global)) return;"
    )
    assert "points * static_cast<std::size_t>(field.ncomp())" in phase
    assert "std::numeric_limits<std::size_t>::max() - applications" in phase
    assert "local_phase_(lane, [&]" in phase and "jacobi_applications_ = 1" in phase
    assert "spacing(" not in phase and "add_local" not in phase


def test_unchanged_physics_and_explicit_realization_in_positive_and_legacy_negative():
    source = (ROOT / "tests/cpp/unit/elliptic/amr_original_field_residual.inc").read_text()
    controls = probe.method(source, "FieldNewtonOptions original_controls()")
    assert ".restart = 80" in controls
    assert ".linear_max_iterations = 240" in controls
    assert ".tolerance = Real(2e-9)" in controls
    assert ".linear_tolerance = Real(1e-5)" in controls
    profile = probe.method(source, "void original_nonlinear_native_profile(")
    assert "for (int cells : {16, 32})" in profile
    assert "Real(1e-5), lane, realization" in profile
    assert "original_field_dot(defect, defect)" in profile
    assert "OriginalAmrNonlinearResidualTwoResolutionsAndPermutation" in source
    assert "OriginalAmrNonlinearIdentityBudgetRejectsWithoutPublication" in source
    assert "OriginalFieldOutcomeStagesAllLevelsAndRevalidatesBeforeAccept" in source
    assert (
        "original_nonlinear_native_profile(AmrFieldRightPreconditioner::kIdentity, true)" in source
    )
    assert (
        "original_nonlinear_native_profile(AmrFieldRightPreconditioner::kSpatialBasisJacobi)"
        in source
    )
    reason = re.search(
        r'"amr_field_newton_gmres_breakdown:iteration_limit:newton=0:columns=240:"', profile
    )
    assert reason and "EXPECT_THROW(prepared->candidate(authority, lane)" in profile
