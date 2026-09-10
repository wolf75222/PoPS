"""Typed solve requests: equation data, unknowns and initialization have distinct roles.

The request is interpreted by the existing Program solve provider.  It is not another
solver registry and does not make an unimplemented residual executable.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from fractions import Fraction
from types import MappingProxyType
from typing import Any

from pops.identity import make_identity
from pops.time.canonical_data import CanonicalData
from pops.time.points import TimePoint
from pops.time.residual_common import residual_name


class SolveRequestError(ValueError):
    """A structured refusal before a solve can publish a value."""

    def __init__(self, code: str, detail: str, *, where: str = "SolveRequest") -> None:
        self.code, self.detail, self.where = code, detail, where
        super().__init__("%s [%s]: %s" % (where, code, detail))

    def to_data(self) -> dict[str, str]:
        return {"code": self.code, "detail": self.detail, "where": self.where}


def _temporal_coordinate(point: TimePoint) -> Any:
    return point.step + Fraction(point.offset.to_python())


@dataclass(frozen=True, slots=True)
class TemporalInterval:
    """An exact closed logical interval; endpoints must share one clock."""

    start: TimePoint
    end: TimePoint
    __pops_ir_immutable__ = True

    def __post_init__(self) -> None:
        if type(self.start) is not TimePoint or type(self.end) is not TimePoint:
            raise SolveRequestError(
                "invalid_interval", "interval endpoints must be exact TimePoint values"
            )
        if self.start.clock != self.end.clock or _temporal_coordinate(
            self.start
        ) >= _temporal_coordinate(self.end):
            raise SolveRequestError(
                "invalid_interval", "interval endpoints need one clock and positive duration"
            )

    def contains(self, point: TimePoint) -> bool:
        return (
            type(point) is TimePoint
            and point.clock == self.start.clock
            and _temporal_coordinate(self.start)
            <= _temporal_coordinate(point)
            <= _temporal_coordinate(self.end)
        )

    def to_data(self) -> dict[str, Any]:
        return {"start": self.start.to_data(), "end": self.end.to_data()}


@dataclass(frozen=True, slots=True)
class DerivativeStrategy:
    """Selected derivative fidelity, never inferred from an opaque callback.

    A native call authenticates exact/approximate providers through its own
    ``derivative_contract``. Finite differences are an explicit request choice.
    The selected solver still has to implement that choice.
    """

    route: str = "exact"

    def __post_init__(self) -> None:
        if self.route not in ("exact", "approximate", "finite_difference", "unavailable"):
            raise SolveRequestError("invalid_derivative", "unknown derivative route")

    def to_data(self) -> dict[str, str]:
        return {"route": self.route}

    def for_native_call(self, call: Any) -> Any:
        contract = getattr(call, "derivative_contract", None)
        if self.route == "unavailable" or not callable(contract):
            raise SolveRequestError(
                "unsupported_derivative", "native call has no selected derivative authority")
        try:
            result = contract(self.route)
        except (TypeError, ValueError, NotImplementedError) as exc:
            raise SolveRequestError("unsupported_derivative", str(exc)) from exc
        if not isinstance(result, Mapping) or result.get("route") != self.route:
            raise SolveRequestError(
                "unsupported_derivative", "native derivative provider changed the selected route")
        return CanonicalData(result, where="native derivative").to_data()


@dataclass(frozen=True, slots=True)
class SolveUnknown:
    """An unknown's identity and value space, independently of its initial guess.

    ``template`` supplies owner, representation, components and temporal placement;
    its stored values do not seed the solve or enter the equation implicitly.
    """

    name: str
    template: Any = field(repr=False, compare=False)
    interval: Any = None

    def __post_init__(self) -> None:
        from pops.time.values import ProgramValue

        residual_name(self.name, "SolveUnknown name")
        if not isinstance(self.template, ProgramValue) or self.template.vtype not in (
            "state", "scalar_field"
        ):
            raise SolveRequestError(
                "invalid_unknown", "unknown template must be a typed state or field ProgramValue")

        if self.interval is not None:
            if type(self.interval) is not TemporalInterval:
                raise SolveRequestError("invalid_interval", "unknown interval must be a TemporalInterval")
            point = self.template.point
            point = point.time if hasattr(point, "time") else point
            if not self.interval.contains(point):
                raise SolveRequestError("invalid_interval", "unknown template point must lie in its interval")

    def to_data(self) -> dict[str, Any]:
        from pops.time.canonical_data import _json_ready

        value = self.template
        return {
            "name": self.name,
            "value_type": value.vtype,
            "block": _json_ready(value.block),
            "quantity": _json_ready(value.state_ref),
            "space": _json_ready(value.space),
            "components": value.logical_shape.get("n_comp") or value.attrs.get("ncomp", 1),
            "point": value.point.to_data(),
            **({"interval": self.interval.to_data()} if self.interval is not None else {}),
        }

    @property
    def identity(self) -> str:
        return make_identity("solve-unknown", self.to_data()).token


def _bindings(value: Any, *, name: str) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        pairs = tuple(value.items())
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        pairs = tuple(value)
    else:
        raise SolveRequestError("invalid_bindings", "%s must be named bindings" % name)
    result = {}
    for item in pairs:
        if not isinstance(item, (tuple, list)) or len(item) != 2:
            raise SolveRequestError("invalid_bindings", "%s requires name/value pairs" % name)
        key, binding = item
        residual_name(key, name)
        if key in result:
            raise SolveRequestError("duplicate_binding", "%s repeats %r" % (name, key))
        result[key] = binding
    return MappingProxyType(result)


@dataclass(frozen=True, slots=True)
class SolveRequest:
    """One problem instance and an ordered product of independently seeded unknowns.

    The first native adapter accepts ``LinearProblem`` with equation inputs exactly
    ``{'operator': A, 'rhs': b}``. Its seed map names every unknown, using ``None``
    for explicit zero initialization. Multiple unknowns are representable; native
    adapters must explicitly support their product before emission.

    ``problem_metadata`` retains the physical equation/source mapping. A discrete
    accumulation relation belongs there separately from the algebraic operator;
    it never changes the native equation without a problem-specific lowering.
    """

    problem: Any
    unknowns: tuple[SolveUnknown, ...]
    equation_inputs: Any
    seeds: Any
    outputs: tuple[str, ...] | None = None
    derivative: DerivativeStrategy = DerivativeStrategy()
    problem_metadata: Any = field(default_factory=dict)
    residual_interpretation: str = "operator(x)-rhs"
    error_interpretation: str = "solver_residual_l2"

    def __post_init__(self) -> None:
        unknowns = tuple(self.unknowns)
        if not unknowns or any(type(item) is not SolveUnknown for item in unknowns):
            raise SolveRequestError("invalid_unknown", "unknowns must be a non-empty typed tuple")
        names = tuple(item.name for item in unknowns)
        if len(set(names)) != len(names):
            raise SolveRequestError("duplicate_unknown", "each unknown must be bound exactly once")
        inputs = _bindings(self.equation_inputs, name="equation_inputs")
        seeds = _bindings(self.seeds, name="seeds")
        if set(seeds) != set(names):
            raise SolveRequestError(
                "missing_unknown_binding", "seeds must bind every unknown exactly once")
        outputs = names if self.outputs is None else tuple(self.outputs)
        if not outputs or len(set(outputs)) != len(outputs) or not set(outputs) <= set(names):
            raise SolveRequestError(
                "invalid_outputs", "outputs must select distinct declared unknown identities")
        if type(self.derivative) is not DerivativeStrategy:
            raise SolveRequestError("invalid_derivative", "derivative must be a DerivativeStrategy")
        for label in ("residual_interpretation", "error_interpretation"):
            residual_name(getattr(self, label), label)
        object.__setattr__(self, "unknowns", unknowns)
        object.__setattr__(self, "equation_inputs", inputs)
        object.__setattr__(self, "seeds", seeds)
        object.__setattr__(self, "outputs", outputs)
        object.__setattr__(self, "problem_metadata", CanonicalData(
            self.problem_metadata, where="SolveRequest physical problem metadata"))

    def build_program_solve(self, *, program: Any, prepared_solver: Any,
                            name: Any = None) -> Any:
        return program._build_solve_request(self, prepared_solver=prepared_solver, name=name)


__all__ = ["DerivativeStrategy", "SolveRequest", "SolveRequestError", "SolveUnknown"]
