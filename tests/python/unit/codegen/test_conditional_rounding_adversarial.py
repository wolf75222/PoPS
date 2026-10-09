"""A non-finite active physical predicate input cannot turn into a finite branch."""

import ctypes
import math
import shutil
import subprocess

import pytest


def test_common_emitter_observes_nan_input_to_conditional_predicate(tmp_path):
    from pops.codegen.cpp_writer import _cse_emit
    from pops.math import Var, where

    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("a C++ compiler is required")
    q = Var("q", "cons")
    expression = where(q > 0., lambda: 1., lambda: 2.)
    lines, (value,) = _cse_emit([expression], "double", "")
    source = ("#include <cmath>\n#include <limits>\n"
              "namespace pops { using Real = double; }\n"
              "extern \"C\" double evaluate(double q) {\n"
              + "\n".join(lines) + "\nreturn " + value + ";\n}\n")
    cpp, library = tmp_path / "predicate.cpp", tmp_path / "predicate.so"
    cpp.write_text(source)
    result = subprocess.run([compiler, "-std=c++17", "-shared", "-fPIC", "-O3",
                             "-fno-fast-math", "-ffp-contract=off", str(cpp),
                             "-o", str(library)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    function = ctypes.CDLL(str(library)).evaluate
    function.argtypes = [ctypes.c_double]
    function.restype = ctypes.c_double
    assert function(1.) == 1.
    assert function(-1.) == 2.
    assert math.isnan(function(float("nan")))


@pytest.mark.parametrize("cse", [True, False])
def test_physical_primitive_in_inactive_where_branch_is_not_evaluated_early(cse):
    """The real model emitter must keep a primitive recipe behind its guard."""
    import pops
    from pops.codegen._native_model_provider_plan import native_model_provider_plan
    from pops.codegen.module_emit_brick import emit_cpp_brick
    from pops.codegen.module_lowering import lower_and_validate
    from pops.domain import Rectangle
    from pops.frames import Cartesian2D
    from pops.layouts import Uniform
    from pops.lib.time import ForwardEuler
    from pops.math import ddt, div, where
    from pops.mesh import CartesianGrid, PeriodicAxes
    from pops.numerics import DiscretizationPlan, FiniteVolume, reconstruction, riemann, variables
    from pops.time import AdaptiveCFL

    frame = Rectangle("guarded-primitive", lower=(0., 0.), upper=(1., 1.)).frame(Cartesian2D())
    x, y = frame.axes
    model = pops.Model("guarded_primitive", frame=frame)
    state = model.state("U", components=("q",))
    (q,) = state
    dangerous = model.scalar("danger", 1 / (q - q))
    flux = model.flux("transport", state=state, frame=frame,
                      components={x: (where(q > 0, lambda: dangerous, lambda: 7.),),
                                  y: (0.,)},
                      waves={x: (1.,), y: (1.,)})
    rate = model.rate("balance", equation=ddt(state) == -div(flux))
    plan = DiscretizationPlan()
    plan.rates.add(rate, FiniteVolume(
        flux=flux, variables=variables.Conservative(state),
        reconstruction=reconstruction.FirstOrder(), riemann=riemann.Rusanov()))
    case = pops.Case("guarded_primitive_case")
    block = case.block("material", model)
    case.numerics(plan, block=block)
    program = ForwardEuler(block[state], rate=rate)
    program.step_strategy(AdaptiveCFL(cfl=0.25, max_dt=1.e-3))
    case.program(program)
    layout = Uniform(CartesianGrid(frame=frame, cells=(8, 8),
                                   periodic=PeriodicAxes(frame.axes)))
    resolved_block = pops.resolve(pops.validate(case), layout=layout).blocks[0]
    emitter, _ = lower_and_validate(
        resolved_block.model, state_space=resolved_block.state_spaces[0],
        resolved_operations=resolved_block.resolved_operations,
        numerics=resolved_block.numerics)

    source = emit_cpp_brick(emitter._m, cse=cse,
                            native_input_plan=native_model_provider_plan(emitter._m))
    flux_start = source.index("flux_evaluation(const State& U")
    flux_end = source.index("State flux(", flux_start)
    flux_source = source[flux_start:flux_end if flux_end != -1 else None]
    assert "const pops::Real danger =" not in flux_source, flux_source
    assert " / " in flux_source, flux_source
    assert "?" in flux_source, flux_source
    assert flux_source.index("?") < flux_source.index(" / "), flux_source
