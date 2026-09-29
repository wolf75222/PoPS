"""Public SymbolicPath Programs: stability belongs to their actual state consumer.

Source checks and compilation of the emitted scalar guard only. These do not execute a
mesh, MPI, Kokkos kernels, or the installed native path runtime.
"""
from fractions import Fraction
import re
import shutil
import subprocess

import pops
import pytest
from pops.codegen.module_lowering import lower_and_validate
from pops.codegen.program_codegen import emit_cpp_program
from pops.layouts import Uniform
from pops.math import ddt, div
from pops.mesh import CartesianGrid, PeriodicAxes
from pops.numerics import DiscretizationPlan, PathConservativeFiniteVolume, reconstruction, riemann, variables
from pops.time import ExternalTimeGrid, FixedDt, StagePoint, TimePoint
from tests.python.support.symbolic_path_case import declarations


def _source(*, strategy, coefficient=Fraction(1, 100), diagnostic=False, alpha=Fraction(1),
            target="system", unused_prediction=False):
    model, state, flux, product, path = declarations()
    rate = model.rate("balance", equation=ddt(state) == -div(flux) - product)
    numerics = DiscretizationPlan()
    numerics.rates.add(rate, PathConservativeFiniteVolume(
        flux=flux, path=path, variables=variables.Conservative(state),
        reconstruction=reconstruction.FirstOrder(), riemann=riemann.Rusanov()))
    case = pops.Case("path-consumer-evidence")
    block = case.block("transport", model)
    case.numerics(numerics, block=block)
    program = pops.Program("path-consumer")
    q = program.state(block[state])
    point = StagePoint("observation", {"main": TimePoint(program.clock, 0)})
    sampled = q.n if alpha == 1 else program.value("sampled", 2 * q.n, at=point)
    rhs = program.value("physical_rhs", rate(sampled), at=point)
    if unused_prediction:
        program.value("unused_prediction", q.n + 100 * program.dt * rhs, at=q.next.point)
    # The diagnostic must remain in the authored Program even though it does not advance q.
    candidate = (q.n if diagnostic else
                 alpha * sampled + (1 - alpha) * q.n + coefficient * program.dt * rhs)
    endpoint = program.value("accepted", candidate, at=q.next.point)
    program.commit(q.next, endpoint)
    program.step_strategy(strategy)
    case.program(program)
    resolved = pops.resolve(pops.validate(case), layout=Uniform(CartesianGrid(
        frame=path.frame, cells=(8, 8), periodic=PeriodicAxes(path.frame.axes))))
    selected = resolved.blocks[0]
    emitter, _ = lower_and_validate(selected.model, state_space=selected.state_spaces[0],
        resolved_operations=selected.resolved_operations, numerics=selected.numerics)
    return emit_cpp_program(resolved.time, model=emitter, target=target)


def _guard(source):
    lines = source.splitlines()
    definitions = [line.strip() for line in lines
                   if "const pops::Real user_face_update_frequency_" in line]
    checks = [line.strip() for line in lines if '"user_face_numerical_stability"' in line]
    assert len(definitions) == len(checks) == 1, "consumer stability guard missing or duplicated"
    return definitions[0], checks[0]


@pytest.mark.parametrize("strategy", [FixedDt(.1), ExternalTimeGrid("sample_times")])
def test_real_public_path_has_deferred_consumer_budget_without_cfl_proposal(strategy):
    source = _source(strategy=strategy)
    assert "ctx.path_rhs_into(" in source
    definition, guard = _guard(source)
    assert "pops::Real(1) / pops::Real(100)" in definition
    assert "ctx.numerical_face_courant()" in guard
    # Source authoring must not install an AdaptiveCFL call behind either strategy.
    assert "step_cfl(" not in source


def test_diagnostic_path_rhs_has_no_explicit_state_stability_budget():
    source = _source(strategy=FixedDt(10.), diagnostic=True)
    assert "ctx.path_rhs_into(" in source, "fixture must exercise an emitted physical RHS"
    assert re.search(r"ctx\.path_rhs_into\([^;]+, &path_frequency_\d+\);", source), (
        "diagnostic RHS must use the deferred numerical-frequency output")
    assert "user_face_update_frequency_" not in source
    assert '"user_face_numerical_stability"' not in source


@pytest.mark.parametrize("alpha", [Fraction(1), Fraction(1, 4)])
def test_emitted_path_consumer_guard_uses_alpha_and_one_percent_of_dt(tmp_path, alpha):
    source = _source(strategy=FixedDt(.1), alpha=alpha)
    definition, guard = _guard(source)
    lhs, rhs = definition.removeprefix("const pops::Real ").rstrip(";").split(" = ", 1)
    symbols = set(re.findall(r"\b[A-Za-z_]\w*\b", rhs)) - {"pops", "Real"}
    assert len(symbols) == 1, definition
    actual_frequency = symbols.pop()
    compiler = shutil.which("clang++") or shutil.which("c++")
    if compiler is None:
        pytest.skip("host C++ compiler unavailable for emitted scalar guard")
    cpp = tmp_path / "guard.cpp"
    cpp.write_text("""#include <cmath>
#include <limits>
namespace pops { using Real=double; }
struct Context {
  int status=0;
  double numerical_face_courant() const { return 1.; }
  void consume_pointwise_evaluation_status(int,int,int value,const char*,int) { status=value; }
};
int guard(double dt, double frequency) {
  Context ctx;
  const double """ + actual_frequency + " = frequency;\n" + definition + "\n" + guard + """
  return ctx.status;
}
int main() {
  // Full dt*nu=10 is unstable, but beta=.01 uses only .1 of the unit budget.
  if (guard(1., 10.) != 0) return 1;
  if (guard(1., """ + str(float(alpha) * 100 + 1) + """) == 0) return 2;
  if (guard(0., 1e8) != 0) return 3;
  if (guard(1., std::numeric_limits<double>::infinity()) == 0) return 4;
}
""")
    binary = tmp_path / "guard"
    subprocess.run([compiler, "-std=c++20", "-O2", str(cpp), "-o", str(binary)], check=True)
    subprocess.run([str(binary)], check=True)


def test_unused_large_prediction_does_not_spend_the_accepted_consumer_budget():
    source = _source(strategy=FixedDt(1.), unused_prediction=True)
    assert 'ProfileScope "node:unused_prediction"' in source
    definition, _ = _guard(source)
    assert "pops::Real(1) / pops::Real(100)" in definition


def test_amr_emission_reads_published_frequency_after_its_hierarchy_barrier():
    # Exercise the real AMR emitter using the same resolved constitutive plan.
    # This is neither an AMR layout-resolution test nor native hierarchy execution.
    source = _source(strategy=ExternalTimeGrid("samples"), target="amr_system",
                     alpha=Fraction(1, 4))
    assert re.search(r"const auto path_frequency_\d+ = ctx\.stage_path_rhs\([^;]+true\);", source)
    definition, guard = _guard(source)
    assert "(*path_frequency_" in definition
    assert source.index("ctx.publish_staged_path_rhs(") < source.index(definition)
    assert "pops::Real(1) / pops::Real(4)" in guard
