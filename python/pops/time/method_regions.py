"""Common temporal equation regions resolved by the existing solve request authority.

These records describe equations, unknown products, and component windows. They
neither choose a method by name nor provide a second numerical executor.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pops.identity import make_identity
from pops.time.canonical_data import CanonicalData
from pops.time.solve_request import SolveRequestError, TemporalInterval, _bindings


def temporal_value_signature(value: Any) -> dict[str, Any]:
    """Exact owner, quantity, space and component identity used at equation boundaries."""
    from pops.time.solve_request import SolveUnknown
    data = SolveUnknown("signature", value).to_data()
    return {key: data[key] for key in ("value_type", "block", "quantity", "space", "components")}


@dataclass(frozen=True, slots=True)
class TemporalProblemRegion:
    """Named residual equations over explicit unknown and frozen-input captures.

    ``equations`` maps each unknown name to an existing immutable ``Region``.
    ``unknown_captures`` and ``input_captures`` bind graph ValueRef ids, never debug
    names or a use-latest state. Component windows map unknown names to exact
    intervals; timed observations remain explicit Region captures.
    """
    equations: Any
    unknown_captures: Any
    input_captures: Any
    windows: Any = ()
    __pops_ir_immutable__ = True

    def __post_init__(self) -> None:
        from pops.time._graph.control import Region
        from pops.time._graph.base import ValueRef
        for name in ("equations", "unknown_captures", "input_captures", "windows"):
            object.__setattr__(self, name, _bindings(getattr(self, name), name=name))
        if not self.equations or any(type(region) is not Region for region in self.equations.values()):
            raise SolveRequestError("invalid_equation_region", "equations require non-empty exact Region values")
        if set(self.equations) != set(self.unknown_captures):
            raise SolveRequestError("missing_equation", "each unknown capture requires exactly one equation")
        refs = (*self.unknown_captures.values(), *self.input_captures.values())
        if any(type(ref) is not ValueRef for ref in refs):
            raise SolveRequestError("invalid_capture", "capture bindings require exact ValueRef values")
        if len({ref.node_id for ref in refs}) != len(refs):
            raise SolveRequestError("duplicate_capture", "unknowns and frozen inputs require distinct capture ids")
        if not set(self.windows) <= set(self.unknown_captures) or any(
                type(window) is not TemporalInterval for window in self.windows.values()):
            raise SolveRequestError("invalid_window", "component windows must bind declared unknowns to intervals")

    def resolve(self, request: Any, *, program: Any) -> ResolvedTemporalProblem:
        return resolve_temporal_problem(request, program=program)


@dataclass(frozen=True, slots=True)
class ResolvedTemporalProblem:
    """Immutable inspected equation contract, distinct from native realization."""
    contract: CanonicalData
    disposition: CanonicalData
    initialization: CanonicalData

    @property
    def identity(self) -> str:
        return make_identity("temporal-problem", self.contract.to_data()).token

    def to_data(self) -> dict[str, Any]:
        return {"identity": self.identity, "contract": self.contract.to_data(),
                "native": self.disposition.to_data(), "analytical_status": "unverified",
                "initialization": self.initialization.to_data()}


def resolve_temporal_problem(request: Any, *, program: Any) -> ResolvedTemporalProblem:
    """Qualify Region captures/results against one exact SolveRequest before lowering."""
    from pops.time.solve_request import SolveRequest
    from pops.time._program.value_validation import require_owned, validate_input_regions
    from pops.time._program.equation_identity import _equation_value
    if type(request) is not SolveRequest or type(request.problem) is not TemporalProblemRegion:
        raise SolveRequestError("invalid_temporal_problem", "resolution requires a temporal-region SolveRequest")
    problem = request.problem
    unknowns = {unknown.name: unknown for unknown in request.unknowns}
    if set(problem.equations) != set(unknowns):
        raise SolveRequestError("missing_equation", "equations must bind every request unknown exactly once")
    if set(problem.input_captures) != set(request.equation_inputs):
        raise SolveRequestError("equation_input_mismatch", "all frozen equation inputs require exact captures")
    bindings = {problem.unknown_captures[name].node_id: unknown.template for name, unknown in unknowns.items()}
    bindings.update({problem.input_captures[name].node_id: value
                     for name, value in request.equation_inputs.items()})
    scoped_values = (*bindings.values(), *(seed for seed in request.seeds.values() if seed is not None))
    try:
        for value in scoped_values:
            require_owned(program, value, "temporal equation binding")
        validate_input_regions(program, scoped_values, program._current_region(),
                               "temporal equation binding")
    except (TypeError, ValueError) as exc:
        raise SolveRequestError("invalid_binding_scope", str(exc)) from exc
    for name, equation in problem.equations.items():
        unknown = unknowns[name]
        signature = temporal_value_signature(unknown.template)
        if equation.result_signature is None or equation.result_signature != CanonicalData(signature):
            raise SolveRequestError("residual_type_mismatch", "equation result must retain unknown owner/quantity/space/components")
        source = equation.result_source()
        if getattr(source, "value_type", signature["value_type"]) != signature["value_type"]:
            raise SolveRequestError("residual_type_mismatch", "residual node has a different value type")
        if equation.result_source().point != unknown.template.point:
            raise SolveRequestError("residual_point_mismatch", "equation residual must use its unknown's exact temporal point")
        for capture in equation.captures:
            value = bindings.get(capture.value.node_id)
            if value is None:
                raise SolveRequestError("unbound_capture", "equation captured an undeclared unknown or input")
            if capture.clock != value.clock or capture.point != value.point or capture.signature is None \
                    or capture.signature != CanonicalData(temporal_value_signature(value)):
                raise SolveRequestError("capture_identity_mismatch", "capture owner/quantity/space/components/clock/point changed")
        window = problem.windows.get(name)
        if window is not None:
            point = unknown.template.point
            point = point.time if hasattr(point, "time") else point
            if not window.contains(point):
                raise SolveRequestError("invalid_window", "component unknown lies outside its declared window")
    outputs = request.outputs
    if outputs is None:
        raise SolveRequestError(
            "invalid_outputs", "resolved temporal solve requests require normalized output identities")
    contract = {"equations": [[name, region.to_data()] for name, region in problem.equations.items()],
                "unknowns": [unknown.to_data() for unknown in request.unknowns],
                "unknown_captures": [[name, ref.to_data()] for name, ref in problem.unknown_captures.items()],
                "input_captures": [[name, ref.to_data()] for name, ref in problem.input_captures.items()],
                "equation_inputs": [[name, _equation_value(program, value)]
                                    for name, value in request.equation_inputs.items()],
                "outputs": list(outputs),
                "windows": [[name, value.to_data()] for name, value in problem.windows.items()],
                "residual_interpretation": request.residual_interpretation,
                "error_interpretation": request.error_interpretation,
                "physical_problem": request.problem_metadata.to_data()}
    code = ("unsupported_interval_unknown" if any(item.interval is not None for item in request.unknowns)
            else "unsupported_unknown_product" if len(unknowns) > 1 else "unsupported_temporal_region")
    return ResolvedTemporalProblem(CanonicalData(contract), CanonicalData({
        "disposition": "unavailable", "code": code,
        "detail": "no authenticated native provider for this temporal equation region"}), CanonicalData({
            "seeds": [[name, None if seed is None else _equation_value(program, seed)]
                      for name, seed in request.seeds.items()]}))


__all__ = ["TemporalInterval", "TemporalProblemRegion", "ResolvedTemporalProblem",
           "resolve_temporal_problem", "temporal_value_signature"]
