"""The public exponential must survive an original local residual unchanged."""

import math
import ctypes
import shutil
import subprocess

import pytest

import pops
from pops import math as pmath, model, time
from pops._ir.expr import Const, Var
from pops._ir.lowering import diff
from pops._ir.visitors import _dag_key_data, _key
from pops.codegen.module_lowering import lower_and_validate
from pops.codegen.program_codegen import emit_cpp_program
from pops.codegen.program_emit_expressions import checked_expression_dag
from pops.solvers.nonlinear import LocalNewton
from pops.params import RuntimeParam
from pops.model._bind_expression import (
    eval_expression_key, expression_reference_keys, qualified_expression_key,
)
from pops.time.expressions import encode_expressions
from typed_program_support import typed_state


def test_exponential_has_distinct_symbolic_identity_and_derivative():
    x = Var("x", "cons")
    exponential = pmath.exp(x)
    assert _key(exponential) == ("exp", _key(x))
    assert exponential.eval({"x": .3}) == pytest.approx(math.exp(.3))
    assert diff(exponential, x).eval({"x": .3}) == pytest.approx(math.exp(.3))
    assert _dag_key_data(exponential) != _dag_key_data(x)


def test_exponential_bind_key_keeps_its_distinct_opcode():
    key = qualified_expression_key(pmath.exp(Const(.3)), where="exp test")
    assert key[0] == "exp" and expression_reference_keys(key, where="exp test") == frozenset()
    assert eval_expression_key(key, {}, where="exp test") == pytest.approx(math.exp(.3))


def test_exponential_in_model_source_keeps_exact_runtime_parameter_dependency():
    physics = pops.Model("exp_source")
    u = physics.state("U", components=("u",))
    parameter = physics.param(RuntimeParam("gain", default=.25))
    gain = physics.value(parameter)
    physics.source("exponential", on=u, value=(pmath.exp(gain*u[0]),))
    emitted = lower_and_validate(physics)[0]
    body = next(iter(emitted.module.operator_registry())).body[0]
    assert _key(body)[0] == "exp"
    assert parameter in body.declaration_references()


def test_exponential_in_original_local_residual_emits_native_math():
    program = time.Program("exponential_residual")
    module = model.Module("exponential_residual_model")
    space = module.state_space("U", ("u",))
    state = typed_state(program, "u", state_name="U", space=space,
                        model=module, state=module.state_handle(space))
    seed = program.value("seed", state.n, at=state.next.point)

    def residual(_program, iterate, *, old):
        return (pmath.exp(iterate[0]) - old[0],)

    outcome = program.solve(
        time.LocalResidual(residual, seed, captures={"old": state.n}),
        solver=LocalNewton(tolerance=1e-11),
    )
    program.commit(state.next, outcome.consume(action=time.FailRun()))
    assert program.validate()
    source = emit_cpp_program(program, model=lower_and_validate(module)[0])
    assert "std::exp(" in source
    assert "prepare_local_nonlinear_problem<1>" in source


def test_inactive_overflow_is_lazy_and_active_overflow_is_observable(tmp_path):
    compiler = shutil.which("clang++") or shutil.which("c++")
    if compiler is None:
        pytest.skip("host C++ compiler unavailable")
    program = time.Program("exp_control")
    x = typed_state(program, "u")[0]
    value = pmath.where(x > 2, lambda: x, lambda: pmath.exp(1000*x))
    roots, nodes, _ = encode_expressions((value,), program)
    lines, values, invalid = checked_expression_dag(roots, nodes, [["x"]])
    source = """#include <cmath>
#include <cfenv>
namespace pops { using Real = double; }
namespace Kokkos { using std::isfinite; }
extern "C" double evaluate(double x, int* invalid, int* overflow) {
  std::feclearexcept(FE_OVERFLOW);
""" + "\n".join(lines) + "\n" + """
  *invalid = (""" + invalid + """ ) ? 1 : 0;
  *overflow = std::fetestexcept(FE_OVERFLOW) ? 1 : 0;
  return """ + values[0] + ";\n}\n"
    cpp, library = tmp_path/"exp_where.cpp", tmp_path/"exp_where.so"
    cpp.write_text(source)
    subprocess.run([compiler, "-std=c++20", "-O2", "-fno-fast-math", "-shared", "-fPIC",
                    str(cpp), "-o", str(library)], check=True)
    function = ctypes.CDLL(str(library)).evaluate
    function.argtypes = [ctypes.c_double, ctypes.POINTER(ctypes.c_int),
                         ctypes.POINTER(ctypes.c_int)]
    function.restype = ctypes.c_double
    flag, overflow = ctypes.c_int(), ctypes.c_int()
    assert function(3., ctypes.byref(flag), ctypes.byref(overflow)) == 3.
    assert (flag.value, overflow.value) == (0, 0)
    assert math.isinf(function(1., ctypes.byref(flag), ctypes.byref(overflow)))
    assert (flag.value, overflow.value) == (1, 1)
