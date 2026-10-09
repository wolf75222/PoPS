"""Independent host-provider C17 check; compiles one small standalone C++ TU.

Set POPS_C17_INCLUDE_DIR to examine a separate, frozen checkout before this
test is integrated. No PoPS native extension or backend is built or loaded.
"""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[4]
CPP = r"""
#include <cmath>
#include <cstdio>
#include <limits>
#include <pops/numerics/nonlinear/prepared_local_nonlinear.hpp>
namespace Kokkos { using std::isfinite; }

using pops::Real;
struct Affine {
  POPS_HD void operator()(const Real (&x)[1], Real (&r)[1]) const { r[0] = x[0] - Real(1); }
};
struct MisleadingJacobian {
  POPS_HD bool operator()(const Real (&)[1], Real (&j)[1][1]) const {
    j[0][0] = Real(1e12);
    return true;
  }
};
struct ExactJacobian {
  POPS_HD bool operator()(const Real (&)[1], Real (&j)[1][1]) const {
    j[0][0] = Real(1);
    return true;
  }
};
int main() {
  pops::PreparedLocalNonlinearControls control;
  control.absolute_tolerance = Real(1e-12);
  control.step_tolerance = Real(1e-6);
  const Real stale_seed[1] = {Real(0)};
  auto false_problem = pops::prepare_local_nonlinear_problem<1>(
      Affine{}, pops::AnalyticLocalJacobian<1, MisleadingJacobian>{MisleadingJacobian{}},
      pops::AcceptAllLocalCandidates<1>{}, control);
  const auto false_result = pops::solve_prepared_local_nonlinear(false_problem, stale_seed);
  const auto false_report = pops::local_nonlinear_solve_report(
      pops::local_nonlinear_status_code(false_result.status), false_result.iterations,
      false_result.evaluations, false_result.reference_residual_norm,
      false_result.residual_norm, false_result.step_norm,
      false_result.condition_evidence, false_result.safeguard_steps);
  std::printf("false %d %.17g %.17g %d\n", int(false_result.status),
              double(false_result.residual_norm), double(false_result.step_norm),
              int(false_report.solved_value_available()));

  const Real almost_root[1] = {Real(1) + Real(1e-7)};
  auto exact_problem = pops::prepare_local_nonlinear_problem<1>(
      Affine{}, pops::AnalyticLocalJacobian<1, ExactJacobian>{ExactJacobian{}},
      pops::AcceptAllLocalCandidates<1>{}, control);
  const auto exact_result = pops::solve_prepared_local_nonlinear(exact_problem, almost_root);
  std::printf("true %d %.17g %.17g\n", int(exact_result.status),
              double(exact_result.residual_norm), double(exact_result.step_norm));

  control.step_tolerance = std::numeric_limits<Real>::quiet_NaN();
  auto invalid_problem = pops::prepare_local_nonlinear_problem<1>(
      Affine{}, pops::AnalyticLocalJacobian<1, ExactJacobian>{ExactJacobian{}},
      pops::AcceptAllLocalCandidates<1>{}, control);
  const auto invalid_result = pops::solve_prepared_local_nonlinear(invalid_problem, stale_seed);
  std::printf("invalid %d\n", int(invalid_result.status));
}
"""


def test_affine_false_progress_cannot_publish_and_true_root_precedes_stagnation(tmp_path):
    compiler = shutil.which("clang++") or shutil.which("c++")
    if compiler is None:
        pytest.skip("host C++ compiler unavailable")
    include = Path(os.environ.get("POPS_C17_INCLUDE_DIR", str(ROOT / "include")))
    source, binary = tmp_path / "c17.cpp", tmp_path / "c17"
    source.write_text(CPP)
    subprocess.run([compiler, "-std=c++20", "-O0", "-fno-fast-math",
                    "-I" + str(include), str(source), "-o", str(binary)],
                   check=True, capture_output=True, text=True)
    rows = [line.split() for line in subprocess.run([str(binary)], check=True,
            capture_output=True, text=True).stdout.splitlines()]
    assert [row[0] for row in rows] == ["false", "true", "invalid"]
    assert rows[0][1] == "4"  # safeguard failure, never converged
    assert float(rows[0][2]) > .9  # original residual still order one
    assert float(rows[0][3]) < 1e-6
    assert rows[0][4] == "0"  # no solved value may be published
    assert rows[1][1] == "0"  # actual residual convergence precedes step guard
    assert abs(float(rows[1][2])) <= 1e-12
    assert float(rows[1][3]) < 1e-6
    assert rows[2][1] == "6"  # invalid control -> unsupported capability
