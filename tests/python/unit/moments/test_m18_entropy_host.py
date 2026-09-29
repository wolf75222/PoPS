"""Small host C++ probe of the *emitted* public M18 residual, not a runtime receipt."""

import ctypes
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pops
import pytest

from examples.migration.scientific.api040_m18_entropy import make_case, moderate_multipliers, target_moments
from pops.codegen.program_emit_local_product import product_residual_lines


@pytest.fixture(scope="module")
def emitted_solver(tmp_path_factory):
    compiler = shutil.which("clang++") or shutil.which("c++")
    if compiler is None:
        pytest.skip("host C++ compiler unavailable")
    case, layout, _ = make_case()
    program = pops.resolve(pops.validate(case), layout=layout).time
    token = next(value for value in program._values if value.op == "solve_coupled_implicit")
    assert token.attrs["output_count"] == 2 and len(token.inputs) == 3
    variables = {value.id: "input%d" % i for i, value in enumerate(token.inputs)}
    body = "\n".join(product_residual_lines(token, variables))
    assert body.count("std::exp(") >= 5
    source = """#include <cmath>
#include <limits>
#include <pops/numerics/nonlinear/prepared_local_nonlinear.hpp>
namespace Kokkos { using std::isfinite; }
struct Capture { const double* data; double operator()(int, int c) const { return data[c]; } };
extern "C" int solve(const double* target, double* output, double* residual_norm) {
  const int index = 0;
  const Capture input2A{target};
  auto residual = [&](const pops::Real (&Ueval)[6], pops::Real (&rout)[6]) {
""" + body + """
  };
  pops::PreparedLocalNonlinearControls controls;
  controls.absolute_tolerance = 2e-11;
  controls.max_iterations = 12;
  controls.max_backtracks = 16;
  controls.minimum_step = 1.0/65536.0;
  controls.safeguard = pops::LocalSafeguardKind::kBacktrackingLineSearch;
  const auto prepared = pops::prepare_local_nonlinear_problem<6>(
      residual, pops::FiniteDifferenceLocalJacobian<6>{},
      pops::AcceptAllLocalCandidates<6>{}, controls);
  pops::Real seed[6] = {0, 0, 0, target[0], target[1], target[2]};
  const auto result = pops::solve_prepared_local_nonlinear(prepared, seed);
  for (int i=0; i<6; ++i) output[i] = result.value[i];
  *residual_norm = result.residual_norm;
  return static_cast<int>(result.status);
}
"""
    folder = tmp_path_factory.mktemp("m18_emitted_host")
    cpp, library = folder/"probe.cpp", folder/"probe.so"
    cpp.write_text(source)
    subprocess.run([compiler, "-std=c++20", "-O2", "-fno-fast-math", "-shared", "-fPIC",
                    "-I" + str(Path(__file__).resolve().parents[4]/"include"),
                    str(cpp), "-o", str(library)], check=True)
    function = ctypes.CDLL(str(library)).solve
    function.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
                         ctypes.POINTER(ctypes.c_double)]
    function.restype = ctypes.c_int
    return function


def _solve(function, target):
    target = np.ascontiguousarray(target, dtype=np.float64)
    output = np.empty(6, dtype=np.float64)
    residual = ctypes.c_double()
    status = function(target.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                      output.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                      ctypes.byref(residual))
    return status, output, residual.value


def test_all_twenty_emitted_native_cell_solves_match_original_residual(emitted_solver):
    expected = moderate_multipliers()
    targets = target_moments(expected)
    for x, y in np.ndindex(4, 5):
        status, result, norm = _solve(emitted_solver, targets[:, x, y])
        assert status == 0, (x, y, status, norm)
        np.testing.assert_allclose(result[:3], expected[:, x, y], rtol=0, atol=1.e-8)
        np.testing.assert_allclose(result[3:], targets[:, x, y], rtol=0, atol=2.e-11)


def test_outside_cone_emitted_native_cell_cannot_publish(emitted_solver):
    status, _, _ = _solve(emitted_solver, np.asarray((1., 0., 1.1)))
    assert status != 0
