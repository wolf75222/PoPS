"""Finite implicit relations whose original equations govern every successful solve.

This is a Python graph-building library, not a solver registry.  All functions below
elaborate ordinary Program operations, including the mandatory acceptance guards.
Algorithms are closed GraphFunctions; no Python callback executes in a native solve.

Failures are closed native sums; algorithm and request-test payloads propagate
unchanged.  ``data_identity`` is the immutable SSA data binding returned with the result;
it is not a fabricated persistent identity across restarts or remeshing.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Any

from pops.time._authoring import atomic_authoring
from pops.time._program.structured_control import value_signature, same_signature
from pops.time._program.value_validation import require_owned
from pops.time.values import ProgramValue
from pops.time.variant_types import Failure, Int64, Product, Result, type_of


# This tag is reserved by the library. An incompatible application tag is rejected
# when the closed error union is formed, rather than silently converting its payload.
RelationErrors = Failure(relation=Product(code=Int64))


class SolveFailure(IntEnum):
    """Library rejection codes; native Result retains the separate origin field."""

    INVALID_ACCURACY = 2101
    EVIDENCE_OUTSIDE_DOMAIN = 2102
    INADMISSIBLE = 2103
    UNSELECTED = 2104
    BOUNDARY = 2105
    INVALID_NORM = 2106
    ORIGINAL_RESIDUAL = 2107
    REQUEST_REJECTED = 2108


def _function(value, where):
    from pops.graph_functions import GraphFunction

    if type(value) is not GraphFunction:
        raise TypeError(where + " requires a closed GraphFunction of exact type")
    return value


def _product(value, where):
    if not isinstance(value, ProgramValue) or value.vtype != "product":
        raise TypeError(where + " requires a native Product")
    return dict(value_signature(value)["members"])


def _fields(value, expected, where):
    members = _product(value, where)
    if set(members) != set(expected):
        raise TypeError(where + " requires exactly fields " + repr(tuple(expected)))
    for key, kind in expected.items():
        if members[key]["vtype"] != kind:
            raise TypeError(where + "." + key + " requires native " + kind)


@dataclass(frozen=True)
class Problem:
    """The finite problem, including its independently declared root selection.

    residual/admissible/selection/boundary bind ``unknown`` and ``data``.
    request_test binds ``unknown, residual, data, accuracy, evidence`` and returns
    Result(Check).  Residual, unknown and data are native Products.  A boundary-free
    problem explicitly supplies a Bool-true boundary function.
    """

    residual: Any
    admissible: Any
    selection: Any
    boundary: Any
    request_test: Any

    def __post_init__(self):
        for name in ("residual", "admissible", "selection", "boundary", "request_test"):
            _function(getattr(self, name), "Problem." + name)


@dataclass(frozen=True)
class Algorithm:
    """A supplied closed graph: (data, guess, accuracy, budget) -> Result(Approximation).

    Problem functions are specialized into this graph at authoring time.  The
    library does not trust its successful status or its internal residual estimate.
    """

    body: Any

    def __post_init__(self):
        _function(self.body, "Algorithm.body")


@dataclass(frozen=True)
class Accuracy:
    """One native request shared by the algorithm and the final original-equation test.

    parameters = Product(atol:Scalar, rtol:Scalar, scale:Scalar, request:Product).
    norm is GraphFunction(residual:Product)->Scalar, with the mathematical norm and
    units supplied by the case.  The mandatory bound is norm(r) <= atol+rtol*scale.
    request carries additional observation-specific targets, checked by request_test.
    This residual bound is not asserted to be a solution-error bound.
    """

    parameters: ProgramValue
    norm: Any

    def __post_init__(self):
        _fields(self.parameters, {"atol": "scalar", "rtol": "scalar", "scale": "scalar",
                                  "request": "product"}, "Accuracy.parameters")
        _function(self.norm, "Accuracy.norm")


@atomic_authoring
def Evidence(program, *, estimates, valid):
    """Native estimates plus their supplied domain-of-validity predicate."""
    require_owned(program, estimates, "Evidence.estimates", vtype="product")
    require_owned(program, valid, "Evidence.valid", vtype="bool")
    return program.product(estimates=estimates, valid=valid)


@atomic_authoring
def Approximation(program, *, values, evidence, termination):
    """Algorithm output payload; only solve can turn it into a checked solve result."""
    require_owned(program, values, "Approximation.values", vtype="product")
    require_owned(program, evidence, "Approximation.evidence", vtype="product")
    require_owned(program, termination, "Approximation.termination", vtype="product")
    _fields(evidence, {"estimates": "product", "valid": "bool"}, "Evidence")
    return program.product(values=values, evidence=evidence, termination=termination)


@atomic_authoring
def Check(program, *, accepted, observations):
    """A request-specific check whose accepted Bool is still guarded by solve."""
    require_owned(program, accepted, "Check.accepted", vtype="bool")
    require_owned(program, observations, "Check.observations", vtype="product")
    return program.product(accepted=accepted, observations=observations)


def _gate(program, value, predicate, code):
    require_owned(program, predicate, "solve acceptance predicate", vtype="bool")
    # The returned value depends on the checked Result.  Inlining or dead-code
    # removal cannot drop a guard as an unused recorded diagnostic.
    result_type = Result(type_of(value), RelationErrors)
    def rejected(p):
        payload = p.variant(RelationErrors, "relation", p.product(code=p.int64(int(code))))
        return p.variant(result_type, "err", payload)
    return program.require_success(program.branch(
        predicate, lambda p: p.variant(result_type, "ok", value), rejected))


def _require_result(value, where):
    if not isinstance(value, ProgramValue) or value.vtype != "variant":
        raise TypeError(where + " requires a typed Result")
    members = dict(value_signature(value)["members"])
    if set(members) != {"ok", "err"} or members["err"]["vtype"] != "variant":
        raise TypeError(where + " requires Result with a closed Failure sum")


def _accuracy(program, parameters):
    checked = parameters
    for name in ("atol", "rtol", "scale"):
        checked = _gate(program, checked, program.isfinite(checked[name]),
                        SolveFailure.INVALID_ACCURACY)
        predicate = checked[name] > 0 if name == "scale" else checked[name] >= 0
        checked = _gate(program, checked, predicate, SolveFailure.INVALID_ACCURACY)
    return checked


@atomic_authoring
def solve(program, problem, *, data, algorithm, guess, accuracy, budget):
    """Return Result(SolveResult), with no successful path bypassing final checks.

    SolveResult is a native Product containing values, data_identity (the actual SSA
    data binding), accuracy, evidence, checks, defects (the original residual), and
    termination.  An inner algorithm/region failure remains a failed Result and can
    be explicitly matched by the caller.  A plain bool callback is not accepted.
    """
    if not isinstance(problem, Problem) or not isinstance(algorithm, Algorithm):
        raise TypeError("solve requires Problem and Algorithm descriptors")
    if not isinstance(accuracy, Accuracy):
        raise TypeError("solve requires an explicit Accuracy request")
    for label, value in (("data", data), ("guess", guess), ("budget", budget),
                         ("accuracy", accuracy.parameters)):
        require_owned(program, value, "solve " + label, vtype="product")
    unknown_signature = value_signature(guess)

    def checked_body(p):
        request = _accuracy(p, accuracy.parameters)
        proposed = algorithm.body(p, data=data, guess=guess, accuracy=request, budget=budget)
        _require_result(proposed, "Algorithm output")
        candidate = p.require_success(proposed)
        _fields(candidate, {"values": "product", "evidence": "product",
                            "termination": "product"}, "Approximation")
        _fields(candidate["evidence"], {"estimates": "product", "valid": "bool"}, "Evidence")
        if not same_signature(value_signature(candidate["values"]), unknown_signature):
            raise TypeError("Algorithm changed an unknown's type, point, or support authority")
        candidate = _gate(p, candidate, candidate["evidence"]["valid"],
                          SolveFailure.EVIDENCE_OUTSIDE_DOMAIN)

        predicates = {}
        # Domain first: residual functions need not be defined on inadmissible
        # candidates.  Every subsequent graph is downstream of the previous gate.
        for name, code in (("admissible", SolveFailure.INADMISSIBLE),
                           ("selection", SolveFailure.UNSELECTED),
                           ("boundary", SolveFailure.BOUNDARY)):
            predicate = getattr(problem, name)(p, unknown=candidate["values"], data=data)
            candidate = _gate(p, candidate, predicate, code)
            predicates[name] = predicate

        residual = problem.residual(p, unknown=candidate["values"], data=data)
        _product(residual, "Problem.residual output")
        norm = accuracy.norm(p, residual=residual)
        require_owned(p, norm, "Accuracy.norm output", vtype="scalar")
        candidate = _gate(p, candidate, p.isfinite(norm), SolveFailure.INVALID_NORM)
        candidate = _gate(p, candidate, norm >= 0, SolveFailure.INVALID_NORM)
        bound = request["atol"] + request["rtol"] * request["scale"]
        candidate = _gate(p, candidate, p.isfinite(bound), SolveFailure.INVALID_ACCURACY)
        candidate = _gate(p, candidate, norm <= bound, SolveFailure.ORIGINAL_RESIDUAL)

        requested = problem.request_test(p, unknown=candidate["values"], residual=residual,
                                        data=data, accuracy=request, evidence=candidate["evidence"])
        _require_result(requested, "Problem.request_test output")
        checked_request = p.require_success(requested)
        _fields(checked_request, {"accepted": "bool", "observations": "product"}, "Check")
        candidate = _gate(p, candidate, checked_request["accepted"], SolveFailure.REQUEST_REJECTED)
        checks = p.product(**predicates, residual_norm=norm, residual_bound=bound,
                           request=checked_request)
        return p.product(values=candidate["values"], data_identity=data, accuracy=request,
                         evidence=candidate["evidence"], checks=checks, defects=residual,
                         termination=candidate["termination"])

    return program.capture(checked_body)


__all__ = ["Problem", "Algorithm", "Accuracy", "Evidence", "Approximation", "Check",
           "SolveFailure", "RelationErrors", "solve"]
