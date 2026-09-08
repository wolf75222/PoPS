"""Imported heterogeneous law evaluated at actual prepared-solver iterates."""
from pathlib import Path
import ctypes
import time

import numpy as np
import pytest
import pops
from pops.codegen._orchestration_compile import build_program_model_graph
from pops.codegen.program_codegen import emit_cpp_program
from pops.domain import Rectangle
from pops.frames import Cartesian2D
from pops.layouts import Uniform
from pops.math import ddt
from pops.model import Signature
from pops.model.bundles import ProductSpace
from pops.native_calls import NativeDerivative, NativeFunction, NativeInputDomain
from pops.native_components import PreparedNativeComponent
from pops.numerics import DiscretizationPlan, JointEvaluation
from pops.solvers.nonlinear import LocalNewton
from pops.time import CoupledImplicitEuler, DerivativeStrategy, RejectAttempt
from tests.python.support.layout_plan import cartesian_grid
from tests.python.support.native_execution_context import artifact_execution_context
from test_native_call_compiled import HEADER


def implicit_case(directory, *, route="exact"):
    directory.mkdir()
    header = HEADER.replace("static inline std::atomic<long> calls{0};",
        "static inline std::atomic<long> calls{0}; static inline std::atomic<long> jac_calls{0}; static inline std::atomic<long> approximate_calls{0};")
    header = header.replace("static pops::NativeCallResult<25> jacobian(double p, double e, double q, double f, double m) {",
        "static pops::NativeCallResult<25> jacobian(double p, double e, double q, double f, double m) { ++jac_calls;")
    header = header.replace("\n};\n}", "\n  static pops::NativeCallResult<25> approximate(double p, double e, double q, double f, double m) { ++approximate_calls; auto result=jacobian(p,e,q,f,m); for(int i=0;i<25;++i) result.values[i]*=.95; return result; }\n};\n}")
    header += 'extern "C" long interaction_calls() { return imported::Exchange::calls.load(); }\n'
    header += 'extern "C" long jacobian_calls() { return imported::Exchange::jac_calls.load(); }\n'
    header += 'extern "C" long approximate_calls() { return imported::Exchange::approximate_calls.load(); }\n'
    (directory / "law.hpp").write_text(header)
    component = PreparedNativeComponent.header_only("implicit.exchange", include_root=directory,
                                                   entry_headers=("law.hpp",))
    frame = Rectangle("domain", lower=(0, 0), upper=(1, 1)).frame(Cartesian2D())
    model = pops.Model("pair", frame=frame)
    left = model.species("a", state=("p", "E"))
    right = model.species("b", state=("p", "E", "m"))
    function = NativeFunction(component, "imported::Exchange::evaluate",
        Signature((left.space, right.space), ProductSpace({"left": left.space, "right": right.space})),
        reads=((0, 0), (1, 0), (1, 2)),
        domains=(NativeInputDomain(1, 2, lower=0, lower_open=True),),
        derivatives=(NativeDerivative("exact", "imported::Exchange::jacobian"),
                     NativeDerivative("approximate", "imported::Exchange::approximate")),
        effects=("fallible", "diagnostic_counter"))
    call = function(left, right, occurrence="implicit-exchange")
    application = model.interaction("exchange", outputs={left: call.left, right: call.right})
    rates = [model.rate("a_rate", equation=ddt(left) == application[left]),
             model.rate("b_rate", equation=ddt(right) == application[right])]
    case = pops.Case("implicit_pair")
    blocks = [case.block("left", model, states=(left,)), case.block("right", model, states=(right,))]
    for block, state, rate in zip(blocks, (left, right), rates, strict=True):
        numerics = DiscretizationPlan()
        numerics.rates.add(rate, JointEvaluation(state))
        case.numerics(numerics, block=block)
    program = pops.Program("implicit_step")._bind_operators(model.module)
    a, b = (program.state(block[state]) for block, state in zip(blocks, (left, right), strict=True))
    strategy = None if route is None else DerivativeStrategy(route)
    outcome = program.solve(CoupledImplicitEuler(application.operator, (a.n, b.n), derivative=strategy),
                            solver=LocalNewton(tolerance=1e-12, max_iterations=20,
                                               finite_difference_step=1e-7))
    solved = outcome.consume(action=RejectAttempt())
    program.commit_many({a.next: solved[a.n.block], b.next: solved[b.n.block]})
    program.step_strategy(pops.time.FixedDt(.001))
    case.program(program)
    return case, program


def test_imported_coupled_solve_requires_explicit_and_authenticated_derivative(tmp_path):
    with pytest.raises(TypeError, match="explicit DerivativeStrategy"):
        implicit_case(tmp_path / "unspecified", route=None)
    case, program = implicit_case(tmp_path / "exact")
    resolved = pops.resolve(pops.validate(case), layout=Uniform(cartesian_grid(n=16, periodic=True)))
    graph = build_program_model_graph(resolved)
    source = emit_cpp_program(resolved.time, model_graph=graph)
    assert source.count("imported::Exchange::evaluate(") == 1
    assert source.count("imported::Exchange::jacobian(") == 1
    assert "pops::AnalyticLocalJacobian<5" in source
    assert "LocalNonlinearEvaluationResult::reject(evaluation_reason_)" in source
    assert source.index("LocalNonlinearEvaluationResult::reject") < source.index("rout[0] =")
    token = next(v for v in program._values if v.op == "solve_coupled_implicit")
    assert token.attrs["derivative_contract"]["route"] == "exact"
    assert tuple(block.local_id for block in token.attrs["output_bindings"].values()) == ("left", "right")
    from pops.time._program.detach import detach_compiled_program
    detached = detach_compiled_program(resolved.time)
    assert detached._ir_hash() == resolved.time._ir_hash()
    assert emit_cpp_program(detached, model_graph=graph) == source
    token = next(v for v in resolved.time._values if v.op == "solve_coupled_implicit")
    bad = dict(token.attrs["derivative_contract"])
    bad["providers"] = ()
    with pytest.raises(RuntimeError, match="frozen"):
        resolved.time._replace_value(token, attrs={**token.attrs, "derivative_contract": bad})
    _case, forged = implicit_case(tmp_path / "forged")
    token = next(v for v in forged._values if v.op == "solve_coupled_implicit")
    forged._replace_value(token, attrs={**token.attrs, "derivative_contract": bad})
    with pytest.raises(ValueError, match="authenticated residual providers"):
        emit_cpp_program(forged)


@pytest.mark.parametrize("route,target", [("approximate", "approximate("), ("finite_difference", None)])
def test_imported_coupled_solve_selects_actual_jacobian_route(tmp_path, route, target):
    case, _program = implicit_case(tmp_path / route, route=route)
    resolved = pops.resolve(pops.validate(case), layout=Uniform(cartesian_grid(n=16, periodic=True)))
    source = emit_cpp_program(resolved.time, model_graph=build_program_model_graph(resolved))
    if target is None:
        assert "pops::FiniteDifferenceLocalJacobian<5>" in source
        assert "imported::Exchange::jacobian(" not in source
    else:
        assert "imported::Exchange::" + target in source
        assert "imported::Exchange::jacobian(" not in source


@pytest.fixture(scope="module", params=("exact", "approximate", "finite_difference"))
def compiled(request, tmp_path_factory):
    from pops._native_selector import select_native_dimension
    select_native_dimension(2)
    case, _program = implicit_case(tmp_path_factory.mktemp(request.param) / "law", route=request.param)
    resolved = pops.resolve(pops.validate(case), layout=Uniform(cartesian_grid(n=16, periodic=True)))
    started = time.perf_counter()
    artifact = pops.compile(resolved)
    native = ctypes.CDLL(str(artifact.program.so_path))
    for name in ("interaction_calls", "jacobian_calls", "approximate_calls"):
        getattr(native, name).restype = ctypes.c_long
    return request.param, artifact, native, time.perf_counter() - started


def initial():
    return {"left": np.stack([np.full((16, 16), value) for value in (2., 5.)]),
            "right": np.stack([np.full((16, 16), value) for value in (1., 3., 2.)])}


@pytest.mark.compiler
@pytest.mark.native_loader
@pytest.mark.integration
def test_native_full_coupled_solve_iterates_and_conserves(compiled, record_property):
    route, artifact, native, elapsed = compiled
    before = native.interaction_calls()
    simulation = pops.bind(artifact, initial_state=initial(),
        resources={"execution_context": artifact_execution_context(artifact)})
    report = pops.run(simulation, t_end=.1, max_steps=100)
    assert report.accepted_steps == 100 and report.rejected_steps == 0
    a, b = np.asarray((2., 5.)), np.asarray((1., 3., 2.))
    for _ in range(100):
        p, q = np.linalg.solve(np.asarray(((1.001, -.0005), (-.001, 1.0005))), (a[0], b[0]))
        power = .5 * (p + q/2) * (q/2-p)
        a, b = np.asarray((p, a[1]+.001*power)), np.asarray((q, b[1]-.001*power, 2.))
    for name, expected in (("left", a), ("right", b)):
        np.testing.assert_allclose(simulation.state_global(name),
            expected[:, None, None] * np.ones((1, 16, 16)), rtol=0, atol=1e-10)
    for component, invariant in ((0, 3), (1, 8)):
        assert abs(simulation.integral("left", component) + simulation.integral("right", component) - invariant) < 1e-10
    local_cells = sum(np.prod(np.asarray(upper)-np.asarray(lower)) for lower, upper in simulation.local_boxes("left"))
    calls = native.interaction_calls() - before
    assert calls >= 2*100*local_cells
    if route == "finite_difference":
        assert native.jacobian_calls() == 0 and native.approximate_calls() == 0
    else:
        assert native.jacobian_calls() > 0
        assert (native.approximate_calls() > 0) == (route == "approximate")
    record_property("selected_derivative", route)
    record_property("actual_residual_native_calls_local", calls)
    record_property("actual_jacobian_calls_local", native.jacobian_calls())
    record_property("compile_seconds", elapsed)
    record_property("program_binary_bytes", Path(artifact.program.so_path).stat().st_size)


@pytest.mark.compiler
@pytest.mark.native_loader
@pytest.mark.integration
def test_failed_imported_iterate_cannot_publish_any_recipient(compiled):
    _route, artifact, _native, _elapsed = compiled
    state = initial()
    state["right"][2, -1, -1] = -1
    simulation = pops.bind(artifact, initial_state=state,
        resources={"execution_context": artifact_execution_context(artifact)})
    from pops._bootstrap import StepAttemptRejected
    with pytest.raises(StepAttemptRejected, match="[Ee]valuation|invalid|reject"):
        pops.run(simulation, t_end=.001, max_steps=1)
    for name in ("left", "right"):
        np.testing.assert_array_equal(simulation.state_global(name), state[name])
    assert simulation._executor_for_block("left")._program_exchange_records() == []
