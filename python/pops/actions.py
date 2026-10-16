"""Checked finite actions assembled entirely from closed Python equation graphs.

An Action is not a catalogue of matrix functions. Its target graph defines a native
packet consumed by the approximation and by an independent final error contract.
The latter is the scientific source of truth for the declared bound: this library
checks its actual inputs, domain predicate and numerical bound, not a theorem
about a name. No Action opcode or host callback survives elaboration.
"""
from __future__ import annotations
from dataclasses import dataclass
from enum import IntEnum
from typing import Any

from pops.time._authoring import atomic_authoring
from pops.time._program.structured_control import new_typed, value_signature
from pops.time._program.value_validation import require_owned
from pops.time.variant_types import Failure, Int64, Product, Result, type_of
from pops.relations import _fields, _function, _require_result

ActionErrors = Failure(action=Product(code=Int64))


class ActionFailure(IntEnum):
    INVALID_ACCURACY = 2401
    EVIDENCE_OUTSIDE_DOMAIN = 2402
    INVALID_ERROR_BOUND = 2403
    ERROR_BOUND_EXCEEDED = 2404
    CONTRACT_REJECTED = 2405


def _gate(p, value, predicate, code):
    require_owned(p, predicate, "Action predicate", vtype="bool")
    result = Result(type_of(value), ActionErrors)
    return p.require_success(p.branch(predicate,
        lambda q: q.variant(result, "ok", value),
        lambda q: q.variant(result, "err", q.variant(ActionErrors, "action",
            q.product(code=q.int64(int(code)))))))


@dataclass(frozen=True)
class Action:
    """Three closed graphs with mandatory final control on the actual result.

    ``target(data) -> Product`` defines the target and its explicit domain data.
    ``body(target,data,accuracy,budget) -> Result(Approximation)`` supplies any
    finite algorithm. Approximation has values/evidence/termination Products;
    evidence has estimates:Product and valid:Bool, as in relations.Approximation.
    ``error_contract(target,data,values,evidence,termination,accuracy,budget)``
    returns Result(Product(accepted:Bool,error_bound:Scalar,observations:Product)).

    accuracy is Product(atol:Scalar,rtol:Scalar,scale:Scalar,request:Product).
    The bound must be finite, nonnegative and <= atol+rtol*scale, and accepted must
    hold. Additional targets and budget semantics belong to the closed contract.
    The library cannot prove a supplied analytical inequality; its equations and
    assumptions are inspectable in to_data() and retained in the Program receipt.
    """
    target: Any
    body: Any
    error_contract: Any

    def __post_init__(self):
        for name in ("target", "body", "error_contract"):
            _function(getattr(self, name), "Action." + name)

    def to_data(self):
        return {"schema": "pops.action.v14", **{
            name: getattr(self, name).to_data() for name in ("target", "body", "error_contract")}}

    def evaluate(self, program, *, data, accuracy, budget):
        """Return a recoverable Result whose ok payload retains the complete receipt."""
        return _evaluate(program, self, data=data, accuracy=accuracy, budget=budget)


@atomic_authoring
def _evaluate(program, action, *, data, accuracy, budget):
    for label, value in (("data", data), ("accuracy", accuracy), ("budget", budget)):
        require_owned(program, value, "Action " + label, vtype="product")
    _fields(accuracy, {"atol": "scalar", "rtol": "scalar", "scale": "scalar",
                       "request": "product"}, "Action accuracy")

    def checked(p):
        request = accuracy
        for name in ("atol", "rtol", "scale"):
            request = _gate(p, request, p.isfinite(request[name]), ActionFailure.INVALID_ACCURACY)
            request = _gate(p, request, request[name] > 0 if name == "scale" else request[name] >= 0,
                            ActionFailure.INVALID_ACCURACY)
        definition = action.target(p, data=data)
        require_owned(p, definition, "Action target definition", vtype="product")
        approximation = action.body(p, target=definition, data=data, accuracy=request, budget=budget)
        _require_result(approximation, "Action body output")
        approximation = p.require_success(approximation)
        _fields(approximation, {"values": "product", "evidence": "product", "termination": "product"},
                "Action Approximation")
        _fields(approximation["evidence"], {"estimates": "product", "valid": "bool"}, "Action Evidence")
        approximation = _gate(p, approximation, approximation["evidence"]["valid"],
                              ActionFailure.EVIDENCE_OUTSIDE_DOMAIN)
        control = action.error_contract(p, target=definition, data=data, values=approximation["values"],
            evidence=approximation["evidence"], termination=approximation["termination"],
            accuracy=request, budget=budget)
        _require_result(control, "Action error_contract output")
        control = p.require_success(control)
        _fields(control, {"accepted": "bool", "error_bound": "scalar", "observations": "product"},
                "Action error_contract check")
        control = _gate(p, control, control["accepted"], ActionFailure.CONTRACT_REJECTED)
        control = _gate(p, control, p.isfinite(control["error_bound"]), ActionFailure.INVALID_ERROR_BOUND)
        control = _gate(p, control, control["error_bound"] >= 0, ActionFailure.INVALID_ERROR_BOUND)
        limit = request["atol"] + request["rtol"] * request["scale"]
        control = _gate(p, control, p.isfinite(limit), ActionFailure.INVALID_ACCURACY)
        control = _gate(p, control, control["error_bound"] <= limit, ActionFailure.ERROR_BOUND_EXCEEDED)
        receipt = p.product(values=approximation["values"], target=definition, data_identity=data,
            accuracy=request, budget=budget, evidence=approximation["evidence"], checks=control,
            termination=approximation["termination"])
        # Ordinary Product node: provenance travels with its actual SSA arguments;
        # there is no new opcode, native action dispatch or alternate equation body.
        return new_typed(p, value_signature(receipt), "product", receipt.inputs,
                         {**receipt.attrs, "action_contract": action.to_data()})
    return program.capture(checked)
