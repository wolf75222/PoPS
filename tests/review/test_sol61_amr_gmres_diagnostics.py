"""Failure diagnostics only: preserve every original GMRES algebra statement."""
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[2]
HEADER = "include/pops/numerics/elliptic/interface/amr_field_newton_krylov.hpp"
BASE = "9fb7e8fb64c3e3de289d9c139ba310e9e47e43a6"
DIAGNOSTIC_FREEZE = "75c1d2c5f59fcda88a394d6caecccfea5fe64380"


def algebra(source):
    source = source.split("LinearResult solve_linear_", 1)[1].split("// Covered/EB", 1)[0]
    statements = re.findall(
        r"(?:copy_|scale_|saxpy_|lincomb_|apply_jvp|project_unknowns_|local_phase_)\([^;]*;"
        r"|^\s*(?:h_|beta|coefficients_|rotated_rhs_|cosine_|sine_)[^\n=;]*=\s*[^;]*;",
        source, re.MULTILINE)
    return [re.sub(r"\s+", "", row) for row in statements]


def test_original_gmres_numeric_operations_are_byte_equivalent_after_whitespace():
    old = subprocess.check_output(["git", "show", f"{BASE}:{HEADER}"], cwd=ROOT, text=True)
    current = subprocess.check_output(["git", "show", f"{DIAGNOSTIC_FREEZE}:{HEADER}"], cwd=ROOT, text=True)
    assert algebra(current) == algebra(old)


def test_diagnostics_keep_budget_tolerance_and_failure_disposition():
    source = subprocess.check_output(["git", "show", f"{DIAGNOSTIC_FREEZE}:{HEADER}"], cwd=ROOT, text=True)
    assert "options_.linear_tolerance * report.residual_norm" in source
    assert "completed < options_.linear_max_iterations" in source
    assert "SolveStatus::kBreakdown, SolveAction::kRejectAttempt" in source
    assert "<< std::hexfloat" in source
    for field in ("newton", "columns", "evaluations", "stop", "beta", "projected", "pivot"):
        assert f'":{field}="' in source
    for cause in ("iteration_limit", "nonfinite_initial_norm", "zero_arnoldi_column",
                  "nonfinite_arnoldi_column", "zero_triangular_pivot",
                  "nonfinite_triangular_pivot", "nonfinite_triangular_coefficient",
                  "nonfinite_recomputed_norm"):
        assert f'"{cause}"' in source
