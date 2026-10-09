"""Exact row captures of one principal spatial evaluation, independent of storage names."""
import pops
import pytest

from pops.codegen.program_codegen import emit_cpp_program
from pops.codegen.program_emit_params import program_param_entries
from pops.codegen.program_models import ProgramModelGraph
from pops.domain import Rectangle
from pops.frames import Cartesian2D
from pops.layouts import Uniform
from pops.math import ddt, div
from pops.mesh import CartesianGrid, PeriodicAxes
from pops.numerics import DiscretizationPlan, FiniteVolume, reconstruction, riemann, variables
from pops.params import RuntimeParam
from pops.time import FixedDt


def captured_group(*, widths=(1, 1), reverse=False, face=False, foreign=False,
                   step_weight=1, diagnostic_only=False):
    frame = Rectangle("capture_domain", (0., 0.), (1., 1.)).frame(Cartesian2D())
    model = pops.Model("capture_law", frame=frame)
    states = tuple(model.species("row%d" % i, state=tuple("q%d" % j for j in range(n)))
                   for i, n in enumerate(widths))
    parameters = tuple(model.param(RuntimeParam("alpha%d" % i, default=.1 + .1 * i))
                       for i in range(len(states)))
    coefficients = tuple(model.value(parameter) for parameter in parameters)
    if foreign:
        other = pops.Model("capture_law", frame=frame)
        coefficients = (other.value(other.param(RuntimeParam("alpha0", default=.1))), *coefficients[1:])
    total = sum(q for state in states for q in state)
    fluxes = tuple(model.flux("F%d" % i, state=state, frame=frame,
        components={frame.x: tuple(total + q for q in state), frame.y: (0.,) * len(state.components)},
        waves=({frame.x: (float(sum(widths) + 1),) * widths[0], frame.y: (0.,) * widths[0]}
               if i == 0 else None))
        for i, state in enumerate(states))
    rates = tuple(model.rate("R%d" % i, equation=ddt(state) == -div(flux))
                  for i, (state, flux) in enumerate(zip(states, fluxes)))
    case = pops.Case("captured_principal")
    order = tuple(reversed(range(len(states)))) if reverse else tuple(range(len(states)))
    blocks = {i: case.block("block%d" % i, model, states=(states[i],)) for i in order}
    for i in order:
        coefficient = coefficients[i]
        reconstruction_policy = reconstruction.User(
            lambda sample, coefficient=coefficient:
                sample(0) + coefficient * (sample(1) - sample(-1)), formal_order=1)
        numerical = (riemann.User(
            body=lambda left, right, fl, fr, speed, coefficient=coefficient:
                .5 * (fl + fr) - coefficient * speed * (right - left),
            stability=lambda left, right, fl, fr, speed, coefficient=coefficient:
                2 * coefficient * speed,
            state=states[i]) if face else riemann.Rusanov())
        plan = DiscretizationPlan()
        plan.rates.add(rates[i], FiniteVolume(flux=fluxes[i],
            variables=variables.Conservative(states[i]), reconstruction=reconstruction_policy,
            riemann=numerical, sampling=tuple(states[j] for j in order if j != i)))
        case.numerics(plan, block=blocks[i])
    program = pops.Program("captured_principal")
    temporal = {i: program.state(blocks[i][states[i]]) for i in order}
    bindings = {states[i]: temporal[i].n for i in order}
    for i in order:
        rhs = rates[i](temporal[i].n, bindings=bindings)
        update = (temporal[i].n if diagnostic_only else
                  temporal[i].n + step_weight * program.dt * rhs)
        program.commit(temporal[i].next, program.value("accepted", update,
                                                       at=temporal[i].next.point))
    program.step_strategy(FixedDt(.001))
    case.program(program)
    layout = Uniform(CartesianGrid(frame=frame, cells=(8, 8), periodic=PeriodicAxes(frame.axes)))
    resolved = pops.resolve(pops.validate(case), layout=layout)
    return resolved, ProgramModelGraph.from_resolved_blocks(resolved.blocks)


@pytest.mark.parametrize("reverse", (False, True))
def test_principal_reconstruction_captures_are_selected_by_component_owner(reverse):
    resolved, graph = captured_group(widths=(2, 3), reverse=reverse)
    entries = program_param_entries(resolved.time, graph)
    assert {(block, name, slot) for block, name, slot, default in entries} == {
        (block, "alpha%d" % slot, slot) for block in range(2) for slot in range(2)}
    source = emit_cpp_program(resolved.time, model=graph)
    assert "sample.component < 2" in source and "sample.component < 5" in source
    assert "policy.params = parameter_sets[0]" in source
    assert "policy.params = parameter_sets[1]" in source
    assert "params.get(0)" in source and "params.get(1)" in source


def test_principal_reconstruction_refuses_same_named_foreign_model_capture():
    with pytest.raises(ValueError, match="another model|belongs|owner"):
        captured_group(foreign=True)


@pytest.mark.parametrize("reverse", (False, True))
def test_principal_face_keeps_vector_rows_and_each_rows_own_bound(reverse):
    resolved, graph = captured_group(widths=(2, 3), reverse=reverse, face=True)
    source = emit_cpp_program(resolved.time, model=graph)
    assert "RowPhysical<2>" in source and "RowPhysical<3>" in source
    assert "policy.params = parameter_sets[0]" in source
    assert "policy.params = parameter_sets[1]" in source
    assert "row_evaluation.stability" in source


def test_generated_reconstruction_selects_row_table_and_slot_in_compiled_scalar_code(tmp_path):
    """C++ emission proof only; the integrated SDK/kernel execution is a separate test."""
    import ctypes
    import shutil
    import subprocess
    from pops.codegen.program_emit_principal_numerics import emit_principal_reconstruction
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("a C++ compiler is required")
    _, graph = captured_group(widths=(2, 3))
    model = graph.model_for_block("block0")
    entry = model._m._principal_groups[0]
    definition, _ = emit_principal_reconstruction(entry)
    wrapper = entry["cpp_name"] + "Reconstruction"
    # Minimal scalar ABI for this emitter test; no native loader/kernel proof is claimed.
    source = """#include <array>
#include <cmath>
#include <limits>
#define POPS_HD
namespace pops {
using Real=double;
struct RuntimeParams { double values[2]{}; double get(int i) const { return values[i]; } };
template<class T> concept ReconstructionPolicy=true;
template<class T> constexpr bool stencil_envelope_fits_storage=true;
}
struct Sample { int component; double operator()(int i) const { return i==0 ? 4. : i>0 ? 7. : 1.; } };
""" + definition + """
extern "C" double evaluate(int component,double alpha0,double alpha1) {
  %s policy{};
  policy.parameter_sets[0].values[0]=alpha0;
  policy.parameter_sets[0].values[1]=99.;
  policy.parameter_sets[1].values[0]=-99.;
  policy.parameter_sets[1].values[1]=alpha1;
  return policy.stencil_face_value(Sample{component});
}
""" % wrapper
    cpp, library = tmp_path / "capture.cpp", tmp_path / "capture.so"
    cpp.write_text(source)
    result = subprocess.run([compiler, "-std=c++20", "-shared", "-fPIC", "-O2",
        "-fno-fast-math", "-ffp-contract=off", str(cpp), "-o", str(library)],
        capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    function = ctypes.CDLL(str(library)).evaluate
    function.argtypes, function.restype = [ctypes.c_int, ctypes.c_double, ctypes.c_double], ctypes.c_double
    for alpha0, alpha1 in ((.25, .75), (.5, -.125)):
        assert [function(i, alpha0, alpha1) for i in range(5)] == [
            4 + 6 * alpha0, 4 + 6 * alpha0, 4 + 6 * alpha1, 4 + 6 * alpha1, 4 + 6 * alpha1]


def test_diagnostic_principal_rhs_has_no_time_step_restriction():
    resolved, graph = captured_group(diagnostic_only=True)
    source = emit_cpp_program(resolved.time, model=graph)
    assert "_resource.publish(" in source
    assert "dt*principal_" not in source
    assert "user_face_numerical_stability" not in source


@pytest.mark.parametrize("weight,spelling", (
    (1, "(pops::Real(1) / pops::Real(1))"), (.01, "pops::Real(0.01)")))
def test_principal_stability_uses_the_consumers_effective_dt(weight, spelling):
    resolved, graph = captured_group(step_weight=weight)
    source = emit_cpp_program(resolved.time, model=graph)
    assert "dt*principal_" not in source
    assert "user_face_numerical_stability" in source
    assert spelling + " * principal_frequency_" in source
