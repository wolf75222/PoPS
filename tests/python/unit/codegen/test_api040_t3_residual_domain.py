"""T3/C16-C19: named physical calls must retain the residual's numeric domain.

Build public equations, compile their exact emitted lambdas, and execute the
actual header-only local nonlinear provider on CPU. The IEEE oracle distinguishes
a masked invalid intermediate from a valid zero residual. This does not exercise
a linked distributed runtime, Kokkos kernels, MPI or GPU execution.
"""
import math
import re

import numpy as np
import pytest

import pops
from pops.codegen.module_lowering import lower_and_validate
from pops.codegen.program_codegen import emit_cpp_program
from pops.frames import Cartesian2D
from pops.math import minimum, rounded, sqrt, where
from pops.solvers.nonlinear import LocalNewton
from pops.time import FailRun, LocalResidual


def build_residual(*, named_source=True, kind="masked", argument="unknown", apply=False, rate_only=False):
    model = pops.Model("domain_witness", frame=Cartesian2D())
    state = model.state("state", components=("quantity",))
    u = state[0]
    if kind == "masked":
        physical_body = minimum(sqrt(-u), 0 * u)
    elif kind == "primitive":
        root = model.primitive("recipe", sqrt(-u))
        physical_body = minimum(root, 0 * u)
    elif kind == "lazy":
        root = model.primitive("recipe", sqrt(-u))
        physical_body = where(u > 0, lambda: 0 * u, lambda: root)
    elif kind == "lazy_invalid":
        root = model.primitive("recipe", sqrt(-u))
        physical_body = where(u > 0, lambda: minimum(root, 0 * u), lambda: 0 * u)
    elif kind == "rounded":
        physical_body = rounded(u + 1.) - u
    elif kind == "leaf":
        physical_body = minimum(u, 0.)
    elif kind == "finite":
        physical_body = 2 * u
    else:
        raise AssertionError(kind)
    if apply:
        coefficient = (minimum(sqrt(-1.), 0.) if kind == "masked" else
                       where(sqrt(4.) > 0, lambda: 2., lambda: sqrt(-1.)) if kind == "lazy" else
                       rounded(sqrt(1.e32) + 1.) - 1.e16 if kind == "rounded" else 2.)
        # Matrix coefficients are independent of the iterate by the public contract.
        operator = model.local_linear_operator("constitutive", on=state, matrix=[[coefficient]])
        source = model.operator("constitutive_source", returns=operator)
    else:
        model.source("constitutive_source", on=state, value=(physical_body,))
        source = model.module.operator_handle("constitutive_source")
    case = pops.Case("domain_case")
    block = case.block("material", model)
    program = pops.Program("local_implicit_domain")
    q = program.state(block[state])
    old = q.n
    seed = program.value("seed", old, at=q.next.point)

    def residual(p, unknown, fixed):
        if named_source:
            rate = (p.apply(source, fixed if argument == "capture" else unknown) if apply
                    else p.source(source, fixed if argument == "capture" else unknown))
            if rate_only:
                return p.value("rate_residual", 0 * unknown + p.dt * rate, at=unknown.point)
            return p.value("original_residual", unknown - fixed - p.dt * rate,
                           at=unknown.point)
        return (unknown[0] - fixed[0]
                - p.dt * minimum(sqrt(-unknown[0]), 0 * unknown[0]),)

    value = program.solve(LocalResidual(residual, seed, captures={"fixed": old}),
                          solver=LocalNewton(tolerance=1e-12)).consume(action=FailRun())
    program.commit(q.next, value)
    assert program.validate()
    model_ir, _ = lower_and_validate(model, facade=model)
    return emit_cpp_program(program, model=model_ir)


def test_ieee_masking_is_not_an_original_residual_domain_check():
    # Same original implicit equation and same seed; no changed PDE/tolerance.
    with np.errstate(invalid="ignore"):
        intermediate = np.sqrt(np.float64(-1))
        source = np.fmin(intermediate, np.float64(0))
    assert math.isnan(intermediate)
    assert source == 0.0
    assert 1.0 - 1.0 - 0.1 * source == 0.0


@pytest.mark.parametrize("named_source", [False, True], ids=["inline-body-control", "named-physical-source"])
def test_original_residual_observes_invalid_sqrt_before_minimum_can_mask_it(named_source):
    emitted = build_residual(named_source=named_source)
    # The residual body must retain an invalid observation for the evaluated sqrt,
    # rather than only test its final residual (which is exactly zero here).
    # The original named-source lowering nested sqrt directly inside fmin;
    # both physical and inline bodies must now retain the domain observation.
    unchecked_mask = re.search(r"Kokkos::fmin\(\s*(?:Kokkos|std)::sqrt\(", emitted)
    assert unchecked_mask is None, (
        "named physical source hides an invalid original-residual intermediate: "
        + emitted[unchecked_mask.start():unchecked_mask.start() + 180]
    )
    assert "quiet_NaN" in emitted and "Kokkos::isfinite" in emitted


def residual_lambda(emitted):
    """Extract the actual generated residual, with its nested lazy branch scopes."""
    start = emitted.index("auto residual_eval = ")
    opening = emitted.index("{", start)
    depth = 1
    end = opening + 1
    while depth:
        depth += (emitted[end] == "{") - (emitted[end] == "}")
        end += 1
    assert emitted[end] == ";"
    return emitted[start:end + 1]


@pytest.fixture(scope="module")
def compiled_residuals(tmp_path_factory):
    """Compile emitted bodies and the actual native cell solver; no mesh runtime."""
    import ctypes
    import shutil
    import subprocess
    from pathlib import Path
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("a C++ compiler is required")
    cases = {
        "masked": {}, "primitive": {"kind": "primitive"},
        "lazy": {"kind": "lazy"}, "lazy_invalid": {"kind": "lazy_invalid"},
        "rounded": {"kind": "rounded"},
        "leaf": {"kind": "leaf", "argument": "capture", "rate_only": True},
        "finite": {"kind": "finite"},
        "captured": {"kind": "finite", "argument": "capture"},
        "captured_primitive": {"kind": "primitive", "argument": "capture"},
        "apply_masked": {"apply": True},
        "apply_finite": {"apply": True, "kind": "finite"},
        "apply_captured": {"apply": True, "kind": "finite", "argument": "capture"},
        "apply_lazy": {"apply": True, "kind": "lazy"},
        "apply_rounded": {"apply": True, "kind": "rounded"},
    }
    code = ('#include <cmath>\n#include <limits>\n#include <cfenv>\n'
            '#pragma STDC FENV_ACCESS ON\n'
            '#include <pops/numerics/nonlinear/prepared_local_nonlinear.hpp>\n'
            'namespace Kokkos { using std::fmin; using std::isfinite; }\n')
    for name, options in cases.items():
        code += ('extern "C" double residual_' + name
                 + '(double unknown, double seed, double capture, double dt) {\n'
                 + 'double Ueval[1] = {unknown}, Gval[1] = {seed}, Cval0[1] = {capture};\n'
                 + 'double rout[1]; std::feclearexcept(FE_ALL_EXCEPT);\n'
                 + residual_lambda(build_residual(**options))
                 + '\nresidual_eval(Ueval, rout); return rout[0];\n}\n')
        code += ('extern "C" int solve_' + name
                 + '(double seed, double capture, double dt, double* out) {\n'
                 + 'double Gval[1] = {seed}, Cval0[1] = {capture};\n'
                 + residual_lambda(build_residual(**options))
                 + '\npops::PreparedLocalNonlinearControls controls;\n'
                 + 'auto prepared = pops::prepare_local_nonlinear_problem<1>(residual_eval, '
                 + 'pops::FiniteDifferenceLocalJacobian<1>{}, '
                 + 'pops::AcceptAllLocalCandidates<1>{}, controls);\n'
                 + 'auto result = pops::solve_prepared_local_nonlinear(prepared, Gval);\n'
                 + '*out = result.value[0]; return static_cast<int>(result.status);\n}\n')
    code += ('extern "C" int invalid_flags() { '
             'return std::fetestexcept(FE_INVALID | FE_DIVBYZERO); }\n')
    folder = tmp_path_factory.mktemp("local_residual_domain")
    source, library = folder / "residuals.cpp", folder / "residuals.so"
    source.write_text(code)
    completed = subprocess.run([compiler, "-std=c++20", "-shared", "-fPIC", "-O3",
                                "-fno-fast-math", "-ffp-contract=off",
                                "-I", str(Path(__file__).resolve().parents[4] / "include"), str(source),
                                "-o", str(library)], capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
    compiled = ctypes.CDLL(str(library))
    for name in cases:
        function = getattr(compiled, "residual_" + name)
        function.argtypes = [ctypes.c_double] * 4
        function.restype = ctypes.c_double
        solver = getattr(compiled, "solve_" + name)
        solver.argtypes = [ctypes.c_double] * 3 + [ctypes.POINTER(ctypes.c_double)]
        solver.restype = ctypes.c_int
    return compiled


@pytest.mark.parametrize("name", ["masked", "primitive", "lazy_invalid", "apply_masked"])
def test_compiled_original_residual_rejects_masked_domain_errors(compiled_residuals, name):
    residual = getattr(compiled_residuals, "residual_" + name)
    assert math.isnan(residual(1., 1., 1., .1))


@pytest.mark.parametrize("name", ["lazy", "apply_lazy"])
def test_compiled_residual_does_not_evaluate_inactive_invalid_branch(compiled_residuals, name):
    residual = getattr(compiled_residuals, "residual_" + name)
    actual = residual(1., 1., 1., .1)
    flags = compiled_residuals.invalid_flags()
    assert actual == pytest.approx(0. if name == "lazy" else -.2)
    assert flags == 0


def test_compiled_primitive_uses_frozen_capture_not_iterate_or_seed(compiled_residuals):
    # sqrt(-capture) is finite; sqrt(-iterate) and sqrt(-seed) are invalid.
    assert compiled_residuals.residual_captured_primitive(2., 9., -3., .1) == 5.
    assert compiled_residuals.invalid_flags() == 0


@pytest.mark.parametrize("name", ["finite", "apply_finite"])
def test_compiled_residual_retains_exact_unknown_and_distinct_capture(compiled_residuals, name):
    residual = getattr(compiled_residuals, "residual_" + name)
    assert residual(2., 19., 3., .1) == pytest.approx(-1.4)


@pytest.mark.parametrize("name", ["captured", "apply_captured"])
def test_compiled_call_evaluates_exact_frozen_argument(compiled_residuals, name):
    residual = getattr(compiled_residuals, "residual_" + name)
    assert residual(2., 19., 3., .1) == pytest.approx(-1.6)


@pytest.mark.parametrize("name", ["rounded", "apply_rounded"])
def test_compiled_residual_preserves_rounded_barrier(compiled_residuals, name):
    residual = getattr(compiled_residuals, "residual_" + name)
    assert residual(1.e16, 99., 1.e16, 1.) == 0.


def test_compiled_residual_observes_leaf_before_minimum_masks_nan(compiled_residuals):
    assert math.isnan(compiled_residuals.residual_leaf(2., 0., float("nan"), .1))


@pytest.mark.parametrize("name", ["masked", "primitive", "lazy_invalid", "apply_masked"])
def test_actual_native_local_provider_refuses_invalid_original_equation(compiled_residuals, name):
    import ctypes
    result = ctypes.c_double()
    status = getattr(compiled_residuals, "solve_" + name)(1., 1., .1, ctypes.byref(result))
    assert status == 5  # LocalNonlinearStatus::kInvalidEvaluation
    assert result.value == 1.  # The provider returns the unmodified candidate on failure.


@pytest.mark.parametrize("name,expected", [("finite", 1.25), ("apply_finite", 1.25),
                                           ("captured", 1.2), ("apply_captured", 1.2),
                                           ("lazy", 1.), ("apply_lazy", 1.25)])
def test_actual_native_local_provider_solves_original_equation(compiled_residuals, name, expected):
    import ctypes
    result = ctypes.c_double()
    status = getattr(compiled_residuals, "solve_" + name)(1., 1., .1, ctypes.byref(result))
    assert status == 0  # LocalNonlinearStatus::kConverged
    assert result.value == pytest.approx(expected, abs=1.e-12)
    original = getattr(compiled_residuals, "residual_" + name)(result.value, 1., 1., .1)
    assert abs(original) <= 1.e-12
