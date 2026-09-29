"""Public residual and control lowering executed by the actual native provider."""
import ctypes
from pathlib import Path
import shutil
import subprocess

import pops
import pytest
from pops.codegen.module_lowering import lower_and_validate
from pops.codegen.program_codegen import emit_cpp_program
from pops.domain import Rectangle
from pops.frames import Cartesian2D
from pops.layouts import Uniform
from pops.mesh import CartesianGrid, PeriodicAxes
from pops.solvers.nonlinear import LocalNewton
from pops.time import FailRun, FixedDt, LocalResidual


def _emitted_solver(*, step_tolerance, width=1, safeguard="exact", damping=1., seed_factor=1):
    frame = Rectangle("box", (0, 0), (1, 1)).frame(Cartesian2D())
    model = pops.Model("original_equation", frame=frame)
    state = model.state("U", components=tuple("component_%d" % i for i in range(width)))
    case = pops.Case("stagnation")
    block = case.block("material", model)
    program = pops.Program("original_residual_acceptance")
    q = program.state(block[state])
    seed = program.value("algorithmic_seed", seed_factor*q.n, at=q.next.point)

    def residual(builder, unknown, fixed):
        if width == 1:
            return (unknown[0]*unknown[0] - fixed[0],)
        return (unknown[0]*unknown[0] + unknown[1] - fixed[0],
                unknown[0] + unknown[1]*unknown[1] - fixed[1])

    solved = program.solve(LocalResidual(residual, seed, captures={"fixed": q.n}),
        solver=LocalNewton(tolerance=1e-12, step_tolerance=step_tolerance,
                           safeguard=safeguard, damping=damping,
                           minimum_step=min(damping, 1./4096.))).consume(action=FailRun())
    program.commit(q.next, solved)
    program.step_strategy(FixedDt(.01))
    case.program(program)
    resolved = pops.resolve(pops.validate(case), layout=Uniform(CartesianGrid(
        frame=frame, cells=(4, 4), periodic=PeriodicAxes(frame.axes))))
    selected = resolved.blocks[0]
    emitted, _ = lower_and_validate(selected.model, state_space=selected.state_spaces[0],
        resolved_operations=selected.resolved_operations, numerics=selected.numerics)
    source = emit_cpp_program(resolved.time, model=emitted)
    begin = source.index("auto residual_eval = ")
    end = source.index("outA(index, 0)", begin)
    return source[begin:end]


@pytest.fixture(scope="module")
def native_original_residual(tmp_path_factory):
    compiler = shutil.which("clang++") or shutil.which("c++")
    if compiler is None:
        pytest.skip("host C++ compiler unavailable")
    cases = {
        "stagnation": dict(step_tolerance=1.),
        "normal": dict(step_tolerance=0.),
        "changed_seed": dict(step_tolerance=0., seed_factor=2),
        "damped": dict(step_tolerance=1e-4, safeguard="damped", damping=1e-6),
        "coupled": dict(step_tolerance=1., width=2),
    }
    code = """#include <cmath>
#include <pops/numerics/nonlinear/prepared_local_nonlinear.hpp>
namespace Kokkos { using std::isfinite; }
"""
    for name, options in cases.items():
        width = options.get("width", 1)
        code += "extern \"C\" int " + name + "(const double* guess, const double* capture, double* out) {\n"
        code += "pops::Real Gval[%d], Cval0[%d];\n" % (width, width)
        code += "for (int i=0; i<%d; ++i) { Gval[i]=guess[i]; Cval0[i]=capture[i]; }\n" % width
        code += _emitted_solver(**options)
        code += "pops::Real original[%d]; residual_eval(solved_.value, original);\n" % width
        code += "for (int i=0; i<%d; ++i) { out[i]=solved_.value[i]; out[%d+i]=original[i]; }\n" % (width, width)
        code += "out[%d]=solved_.residual_norm; out[%d]=solved_.step_norm;\n" % (2*width, 2*width+1)
        code += "return static_cast<int>(solved_.status);\n}\n"
    folder = tmp_path_factory.mktemp("original_residual")
    source, library = folder / "probe.cpp", folder / "probe.so"
    source.write_text(code)
    include = Path(__file__).resolve().parents[4] / "include"
    subprocess.run([compiler, "-std=c++20", "-shared", "-fPIC", "-O2", "-fno-fast-math",
                    "-I" + str(include), str(source), "-o", str(library)], check=True)
    loaded = ctypes.CDLL(str(library))
    for name in cases:
        function = getattr(loaded, name)
        function.argtypes = [ctypes.POINTER(ctypes.c_double)]*3
        function.restype = ctypes.c_int

    def run(name, guess, capture):
        width = len(guess)
        output = (ctypes.c_double*(2*width+2))()
        status = getattr(loaded, name)((ctypes.c_double*width)(*guess),
                                      (ctypes.c_double*width)(*capture), output)
        return status, tuple(output)
    return run


def test_small_step_cannot_accept_large_original_residual(native_original_residual):
    status, output = native_original_residual("stagnation", (2.,), (2.,))
    assert output[0] == pytest.approx(1.5, abs=1e-7)
    assert output[1] > .24
    assert output[3] < 1.
    assert status == 4, "small Newton step published a candidate with original residual about .25"


def test_damping_does_not_turn_step_size_into_an_equation_tolerance(native_original_residual):
    status, output = native_original_residual("damped", (2.,), (2.,))
    assert output[1] > 1.9
    assert output[3] < 1e-4
    assert status == 4


def test_coupled_product_checks_both_original_equations(native_original_residual):
    status, output = native_original_residual("coupled", (2., 1.), (2., 1.))
    assert output[0] == pytest.approx(10./7., abs=1e-7)
    assert output[1] == pytest.approx(2./7., abs=1e-7)
    assert output[2] > .3 and output[3] > .5
    assert output[4] == max(abs(output[2]), abs(output[3]))
    assert status == 4


@pytest.mark.parametrize("name,guess", [("normal", 2.), ("changed_seed", 4.)])
def test_success_requires_original_equation_with_frozen_capture(native_original_residual, name, guess):
    status, output = native_original_residual(name, (guess,), (2.,))
    assert status == 0
    assert output[0] == pytest.approx(2.**.5, abs=1e-12)
    assert abs(output[1]) <= 1e-12
    assert output[2] == abs(output[1])


def test_residual_convergence_precedes_stagnation_stop(native_original_residual):
    status, output = native_original_residual("stagnation", (1.,), (1.,))
    assert status == 0
    assert output[1] == 0.


def test_invalid_capture_cannot_be_accepted_by_step_tolerance(native_original_residual):
    status, _ = native_original_residual("stagnation", (float("nan"),), (float("nan"),))
    assert status == 5
