"""C09 whole-expression observations at existing numerical consumers."""
import ctypes
import math
import shutil
import subprocess

import pytest


def guarded_coefficient(q):
    from pops.math import minimum, where
    return minimum(where(q > 0, lambda: 1 / (q - q), lambda: 2.), 3.)


def test_condensed_matrix_entry_cannot_mask_active_invalid_branch(tmp_path):
    from pops.math import Var
    from pops.codegen.program_emit_condensed import _matrix_entry
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("a C++ compiler is required")
    expression = guarded_coefficient(Var("q", "aux"))
    value = _matrix_entry([[expression]], 0, 0)
    source = ("#include <cmath>\n#include <limits>\n"
              "namespace pops { using Real = double; }\n"
              "namespace Kokkos { using std::fmin; }\n"
              'extern "C" double evaluate(double q) { return ' + value + "; }\n")
    cpp, library = tmp_path / "matrix.cpp", tmp_path / "matrix.so"
    cpp.write_text(source)
    result = subprocess.run([compiler, "-std=c++17", "-shared", "-fPIC", "-O3",
                             "-fno-fast-math", "-ffp-contract=off", str(cpp),
                             "-o", str(library)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    function = ctypes.CDLL(str(library)).evaluate
    function.argtypes, function.restype = [ctypes.c_double], ctypes.c_double
    assert function(-1.) == 2.
    assert math.isnan(function(1.)), "outer minimum masked an invalid selected branch"


def test_affine_moment_coefficient_uses_whole_expression_observation():
    import pops
    from pops.codegen.module_lowering import lower_and_validate
    from pops.codegen.program_codegen import emit_cpp_program
    from pops.domain import Rectangle
    from pops.frames import Cartesian2D
    from pops.moments import moment_names
    from pops.params import RuntimeParam
    frame = Rectangle("moment_square", (0, 0), (1, 1)).frame(Cartesian2D())
    model = pops.Model("conditional_moments", frame=frame)
    state = model.state("U", components=tuple(moment_names(2)))
    p = model.value(model.param(RuntimeParam("omega", default=-1.)))
    omega = guarded_coefficient(p)
    matrix = [[0.] * 6 for _ in range(6)]
    matrix[1][3], matrix[3][1] = omega, -omega
    rotation = model.operator("rotation", returns=model.local_linear_operator(
        "rotation", on=state, matrix=matrix))
    case = pops.Case("conditional_moments")
    block = case.block("moments", model=model)
    program = pops.Program("conditional_moments")
    q = program.state(block[state])
    mapped = program.affine_moment_update(q.n, q.n, linear_operator=rotation,
                                         theta_dt=program.dt / 2, order=2)
    program.commit(q.next, program.value("accepted", mapped, at=q.next.point))
    emitted, _ = lower_and_validate(model, facade=model)
    source = emit_cpp_program(program, model=emitted)
    assert "const pops::Real jxy = ([&]()" in source


def test_diffusion_inlines_guarded_named_primitive_before_cell_emission():
    import pops
    from pops import math as pmath
    from pops.codegen import Production
    from pops.codegen.module_lowering import lower_and_validate
    from pops.codegen.program_codegen import emit_cpp_program
    from pops.domain import Rectangle
    from pops.frames import Cartesian2D
    from pops.layouts import Uniform
    from pops.lib.time import ForwardEuler
    from pops.mesh import CartesianGrid, PeriodicAxes
    from pops.numerics import Diffusion, DiscretizationPlan
    from pops.time import FixedDt
    frame = Rectangle("guarded_heat", (0., 0.), (1., 1.)).frame(Cartesian2D())
    model = pops.Model("guarded_heat", frame=frame)
    state = model.state("U", components=("u",))
    q = state[0]
    danger = model.scalar("danger", 1 / (q - q))
    variable = pmath.where(q > 0, lambda: danger, lambda: q)
    flux = model.diffusive_flux("heat", state=state, value=.1 * pmath.grad(variable))
    rate = model.rate("heat_rate", equation=pmath.ddt(state) == pmath.div(flux))
    case = pops.Case("guarded_heat")
    block = case.block("heat", model)
    plan = DiscretizationPlan()
    plan.rates.add(rate, Diffusion(flux=flux))
    case.numerics(plan, block=block)
    program = ForwardEuler(block[state], rate=rate)
    program.step_strategy(FixedDt(1e-4))
    case.program(program)
    resolved = pops.resolve(pops.validate(case), layout=Uniform(CartesianGrid(
        frame=frame, cells=(8, 8), periodic=PeriodicAxes(frame.axes))), backend=Production())
    emitted, _ = lower_and_validate(model)
    source = emit_cpp_program(resolved.time, model=emitted)
    assert "const pops::Real danger =" not in source
    assert "? ([&]()" in source
