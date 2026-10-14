"""Original-equation solve contracts in genuine Program/closed-function IR.

These are authoring/codegen checks. Native numerical acceptance is in the separate
integration receiver and must be run against a newly built installed identity.
"""
from dataclasses import replace
import json

import pops
import pytest

from pops.codegen.program_codegen import emit_cpp_program
from pops.codegen.program_models import ProgramModelGraph
from pops.graph_functions import GraphFunction
from pops.relations import (Accuracy, Algorithm, Approximation, Check, Evidence, Problem,
                            RelationErrors, SolveFailure, solve)
from pops.time._program.detach import detach_compiled_program
from pops.time.variant_types import Failure, Float64, Product, Result, type_of
from examples.api_v02.relation_newton_krylov import (
    _approximation, _function, _make_problem, _ok, _parameters, _vec,
    build_nested_relation,
)


def _resolved(**kwargs):
    spec = build_nested_relation(**kwargs)
    return pops.resolve(pops.validate(spec.case), layout=spec.layout)


def test_newton_cg_line_search_and_original_checks_survive_real_codegen_and_detach():
    plan = _resolved()
    source = emit_cpp_program(plan.time, model_graph=ProgramModelGraph.from_resolved_blocks(plan.blocks))
    # Three independent native iteration regions, plus checked relation wrappers.
    assert source.count("iteration_stop_") >= 3
    for reason in SolveFailure:
        assert str(int(reason)) in source
    assert "std::isfinite" in source
    assert "typed" in source or "variant" in source
    detached = detach_compiled_program(plan.time)
    assert detached.to_graph().to_data() == plan.time.to_graph().to_data()
    assert detached._ir_hash() == plan.time._ir_hash()


def test_fabricated_algorithm_residual_does_not_remove_original_equation_guard():
    plan = _resolved(mode="fabricated_residual")
    source = emit_cpp_program(plan.time, model_graph=ProgramModelGraph.from_resolved_blocks(plan.blocks))
    assert str(int(SolveFailure.ORIGINAL_RESIDUAL)) in source
    assert str(int(SolveFailure.REQUEST_REJECTED)) in source
    # The dishonest algorithm contains no Newton/CG solve, yet the mandatory
    # original residual guard still survives as executable C++.
    assert "iteration_stop_" not in source


@pytest.mark.parametrize("options", (
    {"mode": "negative_root"}, {"mode": "request_payload_failure"},
    {"inner_budget": 0, "recover": False},
))
def test_failure_reception_programs_really_resolve_and_emit(options):
    plan = _resolved(**options)
    source = emit_cpp_program(plan.time, model_graph=ProgramModelGraph.from_resolved_blocks(plan.blocks))
    assert str(int(SolveFailure.ORIGINAL_RESIDUAL)) in source
    assert str(int(SolveFailure.UNSELECTED)) in source
    assert source and plan.time.to_graph().to_data()


def _toy(p):
    unknown = _vec(p, p.float64(1.), p.float64(2.))
    data = p.product(a=p.float64(1.), b=p.float64(4.))
    parameters = _parameters(p, p.float64(1.e-12))
    budget = p.product(steps=p.int64(2))
    def residual(q, args):
        u, d = args["unknown"], args["data"]
        return _vec(q, u["x"]*u["x"]-d["a"], u["y"]*u["y"]-d["b"])
    problem, norm, r = _make_problem(p, unknown, data, parameters, residual,
                                    positive_selection=True)
    prototype = _approximation(p, unknown, r, p.int64(0), p.int64(0))
    def exact(q, args):
        values = args["guess"]
        residual = problem.residual(q, unknown=values, data=args["data"])
        return _ok(q, _approximation(q, values, residual, q.int64(0), q.int64(0)))
    algorithm = Algorithm(_function(dict(data=data, guess=unknown, accuracy=parameters, budget=budget),
                                    _ok(p, prototype), exact))
    return problem, data, unknown, Accuracy(parameters, norm), budget, algorithm


def test_request_test_is_not_an_unchecked_plain_bool_and_failure_is_atomic():
    p = pops.Program("request_type")
    problem, data, unknown, accuracy, budget, algorithm = _toy(p)
    residual = problem.residual(p, unknown=unknown, data=data)
    estimates = p.product(residual_norm=p.float64(0.), iterations=p.int64(0),
                         line_reductions=p.int64(0))
    evidence = Evidence(p, estimates=estimates, valid=p.boolean(True))
    bad = _function(dict(unknown=unknown, residual=residual, data=data,
                         accuracy=accuracy.parameters, evidence=evidence),
                    p.boolean(True), lambda q, args: q.boolean(True))
    problem = replace(problem, request_test=bad)
    before = p._ir_hash()
    with pytest.raises(TypeError, match="typed Result"):
        solve(p, problem, data=data, algorithm=algorithm, guess=unknown,
              accuracy=accuracy, budget=budget)
    assert p._ir_hash() == before


def test_request_failure_payload_is_preserved_in_the_outer_error_union():
    p = pops.Program("request_payload")
    problem, data, unknown, accuracy, budget, algorithm = _toy(p)
    residual = problem.residual(p, unknown=unknown, data=data)
    evidence = _approximation(p, unknown, residual, p.int64(0), p.int64(0))["evidence"]
    error_type = Failure(observation=Product(measured=Float64, limit=Float64))
    check = Check(p, accepted=p.boolean(True), observations=p.product())
    result_type = Result(type_of(check), error_type)
    prototype = p.variant(result_type, "ok", check)
    def reject(q, args):
        payload = q.variant(error_type, "observation", q.product(
            measured=args["unknown"]["x"]+args["unknown"]["y"], limit=q.float64(2.5)))
        return q.variant(Result(type_of(Check(q, accepted=q.boolean(False), observations=q.product())),
                                error_type), "err", payload)
    request = _function(dict(unknown=unknown, residual=residual, data=data,
                             accuracy=accuracy.parameters, evidence=evidence), prototype, reject)
    result = solve(p, replace(problem, request_test=request), data=data, algorithm=algorithm,
                   guess=unknown, accuracy=accuracy, budget=budget)
    error = dict(type_of(result).signature["members"])["err"]
    assert set(dict(error["members"])) == {"observation", "relation", "runtime"}
    assert set(dict(dict(error["members"])["observation"]["members"])) == {"measured", "limit"}
    # Closed functions reload from serialized data without rerunning the builder.
    restored = GraphFunction.from_data(json.loads(json.dumps(request.to_data())))
    assert restored.identity == request.identity


def test_problem_rejects_host_callbacks_and_accuracy_rejects_untyped_parameters():
    with pytest.raises(TypeError, match="closed GraphFunction"):
        Problem(*(lambda *args: True for _ in range(5)))
    p = pops.Program("bad_precision")
    problem, data, unknown, accuracy, budget, algorithm = _toy(p)
    class HostCallback(GraphFunction):
        def __call__(self, program, **bindings):
            return program.boolean(True)
    fake = HostCallback(inputs=accuracy.norm.inputs, output=accuracy.norm.output,
                        snapshot=accuracy.norm.to_data())
    with pytest.raises(TypeError, match="exact type"):
        Accuracy(accuracy.parameters, fake)
    with pytest.raises(TypeError, match="atol"):
        Accuracy(p.product(atol=p.int64(1), rtol=p.float64(0), scale=p.float64(1),
                           request=p.product()), accuracy.norm)
    with pytest.raises(ValueError, match="expected a product"):
        Approximation(p, values=p.float64(0), evidence=p.product(), termination=p.product())


def test_algorithm_must_return_the_declared_unknown_product_and_authoring_is_atomic():
    p = pops.Program("algorithm_shape")
    problem, data, unknown, accuracy, budget, algorithm = _toy(p)
    bad = _function(dict(data=data, guess=unknown, accuracy=accuracy.parameters, budget=budget),
                    _ok(p, p.float64(0.)), lambda q, args: _ok(q, q.float64(0.)))
    before = p._ir_hash()
    with pytest.raises(TypeError, match="native Product"):
        solve(p, problem, data=data, algorithm=Algorithm(bad), guess=unknown,
              accuracy=accuracy, budget=budget)
    assert p._ir_hash() == before
