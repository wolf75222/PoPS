"""A coupled finite relation with Newton, inner CG and Armijo search in Python.

The two scalar unknowns are replicated.  This is a crossing of three native bounded
regions, not a claim of a distributed PDE solve.  The positive selected solution of
the default problem is independently known to be (1,2).  Its Jacobian is SPD:
diag(1+3*x*x, 2+3*y*y), with off-diagonal 1/4.
"""
from __future__ import annotations

from pops.graph_functions import graph
from pops.time.variant_types import Bool, Failure, Float64, Int64, Product, Result, Variant, type_of
from pops.relations import (Accuracy, Algorithm, Approximation, Check, Evidence,
                            Problem, RelationErrors, solve)


def _replicated_type(value):
    """This example declares scalar arguments polymorphic in the caller's clock."""
    def visit(signature):
        kind = signature["vtype"]
        if kind in ("scalar", "int64", "bool"):
            return {"scalar": Float64, "int64": Int64, "bool": Bool}[kind]
        if kind in ("product", "variant"):
            factory = Product if kind == "product" else Variant
            return factory(**{name: visit(child) for name, child in signature["members"]})
        raise TypeError("this replicated example has no field/array authority erasure")
    return visit(type_of(value).signature)


def _ok(p, value):
    return p.variant(Result(type_of(value), RelationErrors), "ok", value)


def _function(inputs, output, body):
    return graph(inputs=Product(**{name: _replicated_type(value) for name, value in inputs.items()}),
                 output=_replicated_type(output))(body)


def _vec(p, x, y):
    return p.product(x=x, y=y)


def _dot(a, b):
    return a["x"] * b["x"] + a["y"] * b["y"]


def _scale_add(p, a, scale, b):
    return _vec(p, a["x"] + scale*b["x"], a["y"] + scale*b["y"])


def _absolute(p, x):
    return p.branch(x >= 0, lambda q: x, lambda q: -x)


def _norm(p, residual):
    x, y = _absolute(p, residual["x"]), _absolute(p, residual["y"])
    return p.branch(x >= y, lambda q: x, lambda q: y)


def _both(p, first, second):
    return p.branch(first, lambda q: second, lambda q: q.boolean(False))


def _parameters(p, atol, *, max_sum=3.1):
    return p.product(atol=atol, rtol=p.float64(0.), scale=p.float64(1.),
                     request=p.product(max_sum=p.float64(max_sum)))


def _evidence(p, values, residual, iterations, reductions, *, invalid=False):
    estimate = _norm(p, residual)
    return Evidence(p, estimates=p.product(residual_norm=estimate, iterations=iterations,
                                           line_reductions=reductions),
                    valid=p.boolean(False) if invalid else p.isfinite(estimate))


def _approximation(p, values, residual, iterations, reductions, *, invalid=False):
    return Approximation(p, values=values,
                         evidence=_evidence(p, values, residual, iterations, reductions,
                                            invalid=invalid),
                         termination=p.product(iterations=iterations,
                                               line_reductions=reductions))


def _request_check(p, args):
    accepted = args["unknown"]["x"] + args["unknown"]["y"] <= args["accuracy"]["request"]["max_sum"]
    return _ok(p, Check(p, accepted=accepted, observations=p.product(
        sum=args["unknown"]["x"] + args["unknown"]["y"])))


def _make_problem(p, unknown, data, parameters, residual_body, *, positive_selection):
    residual_template = _vec(p, p.float64(0.), p.float64(0.))
    args = {"unknown": unknown, "data": data}
    residual = _function(args, residual_template, residual_body)
    domain = _function(args, p.boolean(True), lambda q, a: _both(q,
        _norm(q, a["unknown"]) < 64., q.isfinite(_norm(q, a["unknown"]))))
    selected = _function(args, p.boolean(True), lambda q, a:
        _both(q, a["unknown"]["x"] > 0., a["unknown"]["y"] > 0.)
        if positive_selection else q.boolean(True))
    boundary = _function(args, p.boolean(True), lambda q, a: q.boolean(True))
    evidence = _evidence(p, unknown, residual_template, p.int64(0), p.int64(0))
    check = _ok(p, Check(p, accepted=p.boolean(True), observations=p.product(sum=p.float64(0.))))
    request = _function(dict(args, residual=residual_template, accuracy=parameters,
                             evidence=evidence), check, _request_check)
    norm = _function({"residual": residual_template}, p.float64(0.),
                     lambda q, a: _norm(q, a["residual"]))
    return Problem(residual, domain, selected, boundary, request), norm, residual_template


def _linear_action(p, data, vector):
    return _vec(p, data["a11"]*vector["x"] + data["a12"]*vector["y"],
                   data["a12"]*vector["x"] + data["a22"]*vector["y"])


def _linear_residual(p, args):
    image = _linear_action(p, args["data"], args["unknown"])
    return _vec(p, image["x"]-args["data"]["b1"], image["y"]-args["data"]["b2"])


def _cg_body(problem, p, args):
    data, guess, accuracy = args["data"], args["guess"], args["accuracy"]
    original = problem.residual(p, unknown=guess, data=data)
    residual = _vec(p, -original["x"], -original["y"])
    initial = p.product(x=guess, r=residual, d=residual)
    bound = accuracy["atol"] + accuracy["rtol"]*accuracy["scale"]

    def step(q, state):
        x, r, d = state["x"], state["r"], state["d"]
        image = _linear_action(q, data, d)
        rr = _dot(r, r)
        alpha = rr/_dot(d, image)
        new_x = _scale_add(q, x, alpha, d)
        new_r = _scale_add(q, r, -alpha, image)
        beta = _dot(new_r, new_r)/rr
        return q.product(x=new_x, r=new_r, d=_scale_add(q, new_r, beta, d))

    outcome = p.require_success(p.iterate(initial, step,
        lambda q, state: _norm(q, state["r"]) <= bound,
        max_steps=args["budget"]["krylov"]))
    values = outcome["state"]["x"]
    residual = problem.residual(p, unknown=values, data=data)
    return _ok(p, _approximation(p, values, residual, outcome["iterations"], p.int64(0)))


def _nonlinear_residual(p, args):
    x, y, data = args["unknown"]["x"], args["unknown"]["y"], args["data"]
    return _vec(p, x*x*x + x + .25*y - data["b1"],
                   y*y*y + 2.*y + .25*x - data["b2"])


def _multiple_roots_residual(p, args):
    u, data = args["unknown"], args["data"]
    return _vec(p, u["x"]*u["x"]-data["b1"], u["y"]*u["y"]-data["b2"])


def relation_graph(program, *, inner_budget=4, line_budget=20, mode="solve", target=(1., 2.)):
    """Return a checked Result and its explicit initial guess, without committing state.

    mode='fabricated_residual' supplies a lying algorithm; 'bad_request' tightens an
    independent output observation; 'bad_evidence' invalidates estimator domain.
    A zero inner or line-search budget supplies an observable interior failure.
    """
    if mode not in ("solve", "fabricated_residual", "bad_request", "bad_evidence",
                    "negative_root", "request_payload_failure", "bad_accuracy", "bad_boundary",
                    "inadmissible", "bad_norm"):
        raise ValueError("unknown relation reception mode")
    p = program
    guess = _vec(p, p.float64(.2), p.float64(.2))
    tx, ty = map(float, target)
    data = p.product(b1=p.float64(tx*tx if mode == "negative_root" else tx**3+tx+.25*ty),
                     b2=p.float64(ty*ty if mode == "negative_root" else ty**3+2*ty+.25*tx))
    parameters = _parameters(p, p.float64(-1. if mode == "bad_accuracy" else 1.e-10),
                             max_sum=2.5 if mode == "bad_request" else tx+ty+.1)
    budget = p.product(newton=p.int64(12), krylov=p.int64(inner_budget),
                       line_search=p.int64(line_budget))
    outer_problem, norm, residual_template = _make_problem(
        p, guess, data, parameters,
        _multiple_roots_residual if mode == "negative_root" else _nonlinear_residual,
        positive_selection=True)
    if mode == "bad_norm":
        norm = _function({"residual": residual_template}, p.float64(-1.),
                         lambda q, args: q.float64(-1.))
    if mode == "bad_boundary":
        from dataclasses import replace
        boundary = _function(dict(unknown=guess, data=data), p.boolean(False),
                             lambda q, args: q.boolean(False))
        outer_problem = replace(outer_problem, boundary=boundary)
    if mode == "request_payload_failure":
        from dataclasses import replace
        error_type = Failure(observation=Product(measured=Float64, limit=Float64))
        result_type = Result(Product(accepted=Bool, observations=Product()), error_type)
        check = p.variant(result_type, "ok", Check(p, accepted=p.boolean(True), observations=p.product()))
        evidence = _evidence(p, guess, residual_template, p.int64(0), p.int64(0))
        def reject(q, args):
            payload = q.variant(error_type, "observation", q.product(
                measured=args["unknown"]["x"]+args["unknown"]["y"], limit=q.float64(2.5)))
            return q.variant(result_type, "err", payload)
        request = _function(dict(unknown=guess, residual=residual_template, data=data,
                                 accuracy=parameters, evidence=evidence), check, reject)
        outer_problem = replace(outer_problem, request_test=request)
    linear_data = p.product(a11=p.float64(1.), a12=p.float64(.25), a22=p.float64(2.),
                            b1=p.float64(0.), b2=p.float64(0.))
    # Inner precision is explicitly tighter; the reconstructed nonlinear residual
    # is nevertheless checked against the unchanged outer request by solve.
    inner_parameters = _parameters(p, p.float64(1.e-12), max_sum=1.e6)
    linear_problem, linear_norm, _ = _make_problem(
        p, guess, linear_data, inner_parameters, _linear_residual, positive_selection=False)
    prototype = _approximation(p, guess, residual_template, p.int64(0), p.int64(0))
    algorithm_inputs = dict(data=linear_data, guess=guess, accuracy=inner_parameters, budget=budget)
    linear_algorithm = Algorithm(_function(algorithm_inputs, _ok(p, prototype),
        lambda q, args: _cg_body(linear_problem, q, args)))

    def newton(q, args):
        if mode in ("negative_root", "inadmissible"):
            chosen = _vec(q, q.float64(-tx if mode == "negative_root" else 100.),
                            q.float64(-ty if mode == "negative_root" else 100.))
            actual = outer_problem.residual(q, unknown=chosen, data=args["data"])
            return _ok(q, _approximation(q, chosen, actual, q.int64(0), q.int64(0)))
        if mode == "fabricated_residual":
            invented = _vec(q, q.float64(0.), q.float64(0.))
            return _ok(q, _approximation(q, args["guess"], invented, q.int64(0), q.int64(0)))
        rhs_data, request, limits = args["data"], args["accuracy"], args["budget"]
        initial = q.product(x=args["guess"], reductions=q.int64(0))
        bound = request["atol"] + request["rtol"]*request["scale"]

        def step(t, state):
            x = state["x"]
            residual = outer_problem.residual(t, unknown=x, data=rhs_data)
            local_data = t.product(a11=1.+3.*x["x"]*x["x"], a12=t.float64(.25),
                                  a22=2.+3.*x["y"]*x["y"],
                                  b1=-residual["x"], b2=-residual["y"])
            zero = _vec(t, t.float64(0.), t.float64(0.))
            tight = _parameters(t, bound*.01, max_sum=1.e6)
            direction = t.require_success(solve(t, linear_problem, data=local_data,
                algorithm=linear_algorithm, guess=zero, accuracy=Accuracy(tight, linear_norm),
                budget=limits))["values"]
            slope = _dot(residual, _linear_action(t, local_data, direction))
            merit = .5*_dot(residual, residual)

            def acceptable(s, alpha):
                trial = _scale_add(s, x, alpha, direction)
                domain = outer_problem.admissible(s, unknown=trial, data=rhs_data)
                def in_domain(u):
                    r = outer_problem.residual(u, unknown=trial, data=rhs_data)
                    return .5*_dot(r, r) <= merit + .0001*alpha*slope
                return s.branch(domain, in_domain, lambda u: u.boolean(False))

            search = t.require_success(t.iterate(t.float64(1.),
                lambda s, alpha: alpha*.5, acceptable, max_steps=limits["line_search"]))
            updated = _scale_add(t, x, search["state"], direction)
            return t.product(x=updated, reductions=state["reductions"]+search["iterations"])

        outcome = q.require_success(q.iterate(initial, step,
            lambda t, state: _norm(t, outer_problem.residual(t, unknown=state["x"], data=rhs_data)) <= bound,
            max_steps=limits["newton"]))
        values = outcome["state"]["x"]
        residual = outer_problem.residual(q, unknown=values, data=rhs_data)
        return _ok(q, _approximation(q, values, residual, outcome["iterations"],
            outcome["state"]["reductions"], invalid=mode == "bad_evidence"))

    algorithm = Algorithm(_function(dict(data=data, guess=guess, accuracy=parameters, budget=budget),
                                     _ok(p, prototype), newton))
    return solve(p, outer_problem, data=data, algorithm=algorithm, guess=guess,
                 accuracy=Accuracy(parameters, norm), budget=budget), guess


def build_nested_relation(*, mode="solve", inner_budget=4, line_budget=20, recover=True,
                          target=(1., 2.)):
    """A real PoPS case; six stored components expose successful or handled failure results."""
    import numpy as np
    import pops
    from pops.time import FixedDt
    from pops.time.expressions import ProgramExpression, ProgramScalar
    from examples.api_v02.models_M01_M14 import _domain, _initial
    from examples.api_v02.common import CaseSpec, state_array, non_owner_metrics

    frame, layout = _domain("nested_relation", (4, 4))
    model = pops.Model("nested_relation", frame=frame)
    storage = model.state("solver", components=("x", "y", "code", "origin", "iterations", "reductions"))
    case = pops.Case("nested_relation"); block = case.block("relation", model, states=(storage,))
    p = pops.Program("nested_relation_equations"); state = p.state(block[storage])
    result, guess = relation_graph(p, mode=mode, inner_budget=inner_budget,
                                   line_budget=line_budget, target=target)
    def accepted(q, checked):
        return q.product(values=checked["values"], code=q.int64(0), origin=q.int64(0),
                         iterations=checked["termination"]["iterations"],
                         reductions=checked["termination"]["line_reductions"])
    def failed(q, error):
        def payload(t, code, origin):
            return t.product(values=guess, code=code, origin=origin,
                             iterations=t.int64(0), reductions=t.int64(0))
        handlers = dict(relation=lambda t, e: payload(t, e["code"], t.int64(1)),
                        runtime=lambda t, e: payload(t, e["code"], e["origin"]))
        if "observation" in dict(type_of(error).signature["members"]):
            handlers["observation"] = lambda t, e: t.product(
                values=_vec(t, e["measured"], e["limit"]), code=t.int64(2201),
                origin=t.int64(1), iterations=t.int64(0), reductions=t.int64(0))
        return q.match(error, **handlers)
    outcome = p.match(result, ok=accepted, err=failed) if recover else accepted(p, p.require_success(result))
    scalars = (outcome["values"]["x"], outcome["values"]["y"],
               p.float64(outcome["code"]), p.float64(outcome["origin"]),
               p.float64(outcome["iterations"]), p.float64(outcome["reductions"]))
    candidate = p.value("relation_outcome", ProgramExpression(tuple(ProgramScalar(x) for x in scalars), state.n),
                        at=state.next.point)
    p.commit(state.next, candidate); p.step_strategy(FixedDt(.1)); case.program(p)
    initial = np.broadcast_to(np.array([.2, .2, 0., 0., 0., 0.])[:, None, None], (6, 4, 4)).copy()
    subject = block[storage]
    def invariants(runtime):
        actual = state_array(runtime, "relation", initial.shape)
        if actual is None: return non_owner_metrics()
        x, y = actual[0], actual[1]
        tx, ty = target
        # Independent selected root; no Newton/CG/search iteration is replayed.
        error = float(max(np.max(abs(x-tx)), np.max(abs(y-ty))))
        return dict(passed=bool(error < 2.e-10 and np.all(actual[2:4] == 0)
                               and np.all(actual[4] > 0) and np.all(actual[5] > 0)),
                    metrics_owner=True, selected_root_error=error,
                    iterations=float(actual[4,0,0]), line_reductions=float(actual[5,0,0]))
    return CaseSpec(case, layout, _initial(case, subject, initial), {"relation": subject}, .1, .1,
                    invariants, "Replicated two-unknown nonlinear relation; Python Newton/CG/Armijo; "
                    "mandatory original residual, selection, domain, boundary and requested-observation checks")
