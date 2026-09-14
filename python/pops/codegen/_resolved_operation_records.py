"""Immutable numerical records; scientific declarations remain the input authority."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, fields, is_dataclass
from fractions import Fraction
import math
from types import MappingProxyType
from typing import Any


def identifier(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value:
        raise TypeError("%s must be a non-empty string" % where)
    return value


def identifiers(value: Any, where: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)):
        raise TypeError("%s must be a sequence, not a string" % where)
    result = tuple(identifier(item, where) for item in value)
    if len(result) != len(set(result)):
        raise ValueError("%s contains duplicate identities" % where)
    return result


def freeze(value: Any) -> Any:
    """Accept data, never executable objects or permissive repr-based identities."""
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float and math.isfinite(value):
        return value
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("resolved metadata keys must be strings")
        return MappingProxyType({key: freeze(value[key]) for key in sorted(value)})
    if isinstance(value, (list, tuple)):
        return tuple(freeze(item) for item in value)
    raise TypeError("resolved metadata requires finite scalar/container data, got %s"
                    % type(value).__name__)


def data(value: Any) -> Any:
    if isinstance(value, Fraction):
        return [value.numerator, value.denominator]
    if is_dataclass(value):
        return {item.name: data(getattr(value, item.name)) for item in fields(value)}
    if isinstance(value, Mapping):
        return {key: data(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [data(item) for item in value]
    return value


def record_data(cls: Any, value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {item.name for item in fields(cls)}:
        raise TypeError("%s data has an invalid schema" % cls.__name__)
    return dict(value)


@dataclass(frozen=True, slots=True)
class TermOccurrence:
    """One signed occurrence, including multiplicity independent of shared evaluation."""

    identity: str
    operator: str
    target: str
    kind: str
    coefficient: Fraction = Fraction(1)

    def __post_init__(self) -> None:
        for name in ("identity", "operator", "target", "kind"):
            identifier(getattr(self, name), "occurrence.%s" % name)
        if type(self.coefficient) not in (Fraction, int, float):
            raise TypeError("occurrence coefficient must be an exact number")
        object.__setattr__(self, "coefficient", Fraction(self.coefficient))

    @property
    def sign(self) -> int:
        return (self.coefficient > 0) - (self.coefficient < 0)

    @classmethod
    def from_data(cls, value: Any) -> TermOccurrence:
        values = record_data(cls, value)
        coefficient = values["coefficient"]
        if not isinstance(coefficient, list) or len(coefficient) != 2 \
                or any(type(item) is not int for item in coefficient):
            raise TypeError("coefficient data must be [integer numerator, integer denominator]")
        values["coefficient"] = Fraction(*coefficient)
        result = cls(**values)
        if data(result) != dict(value):
            raise ValueError("occurrence coefficient data is not canonical")
        return result


@dataclass(frozen=True, slots=True)
class ResolvedAccess:
    """A typed leaf projected into one specific numerical sampling context."""

    reference: str
    kind: str
    representation: Mapping[str, Any]
    sampling: str = "cell"
    components: tuple[str, ...] = ()
    complete: bool = False
    origins: tuple[str, ...] = ()
    physical_map: str | None = None

    def __post_init__(self) -> None:
        for name in ("reference", "kind", "sampling"):
            identifier(getattr(self, name), "access.%s" % name)
        if not isinstance(self.representation, Mapping):
            raise TypeError("access representation must be typed descriptor data")
        if not self.representation:
            raise ValueError("access representation cannot be empty")
        object.__setattr__(self, "representation", freeze(self.representation))
        object.__setattr__(self, "components", identifiers(self.components, "access.components"))
        object.__setattr__(self, "origins", identifiers(self.origins, "access.origins"))
        if type(self.complete) is not bool:
            raise TypeError("access.complete must be bool")
        if self.physical_map is not None:
            identifier(self.physical_map, "access.physical_map")

    @classmethod
    def from_data(cls, value: Any) -> ResolvedAccess:
        return cls(**record_data(cls, value))


@dataclass(frozen=True, slots=True)
class ExchangeRecord:
    """A rate evaluation's exchange descriptor, before temporal weighting."""

    occurrence: str
    target: str
    orientation: int = 1
    measure: str = "face_measure"
    quadrature: str = "program_accepted_weights"

    def __post_init__(self) -> None:
        for name in ("occurrence", "target", "measure", "quadrature"):
            identifier(getattr(self, name), "exchange.%s" % name)
        if type(self.orientation) is not int or self.orientation not in (-1, 1):
            raise ValueError("exchange orientation must be exactly -1 or 1")

    @classmethod
    def from_data(cls, value: Any) -> ExchangeRecord:
        return cls(**record_data(cls, value))


@dataclass(frozen=True, slots=True)
class EvaluationRequest:
    """One requested numerical evaluation, not a timestep-wide occurrence count."""

    identity: str
    occurrences: tuple[TermOccurrence, ...]
    context: str = "declaration"

    def __post_init__(self) -> None:
        identifier(self.identity, "evaluation.identity")
        identifier(self.context, "evaluation.context")
        occurrences = tuple(self.occurrences)
        if any(type(item) is not TermOccurrence for item in occurrences):
            raise TypeError("evaluation requires exact TermOccurrence values")
        identifiers((item.identity for item in occurrences), "evaluation.occurrences")
        object.__setattr__(self, "occurrences", occurrences)

    @classmethod
    def from_data(cls, value: Any) -> EvaluationRequest:
        values = record_data(cls, value)
        values["occurrences"] = tuple(TermOccurrence.from_data(item)
                                       for item in values["occurrences"])
        return cls(**values)


@dataclass(frozen=True, slots=True)
class NumericalConstruction:
    """Resolved numerical choice and the exact physical occurrences it consumes.

    A native route names a compiler realization, never an executed capability.
    Empty ``native_route`` retains a meaningful operation with an explicit refusal.
    """

    identity: str
    evaluation: str
    consumes: tuple[str, ...]
    inputs: tuple[ResolvedAccess, ...] = ()
    outputs: tuple[Mapping[str, Any], ...] = ()
    sampling: str = "cell"
    dependencies: tuple[str, ...] = ()
    stencil_radius: int = 0
    materialized_inputs: tuple[str, ...] = ()
    reductions: tuple[str, ...] = ()
    communication: tuple[str, ...] = ()
    effects: tuple[str, ...] = ()
    guards: tuple[str, ...] = ()
    exchanges: tuple[ExchangeRecord, ...] = ()
    native_route: str | None = None
    refusal: str | None = None
    decomposition: tuple[tuple[str, ...], ...] = ()
    guarantees: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("identity", "evaluation", "sampling"):
            identifier(getattr(self, name), "construction.%s" % name)
        for name in ("consumes", "dependencies", "materialized_inputs", "reductions",
                     "communication", "effects", "guards"):
            object.__setattr__(self, name, identifiers(getattr(self, name), name))
        for name, cls in (("inputs", ResolvedAccess), ("exchanges", ExchangeRecord)):
            values = tuple(getattr(self, name))
            if any(type(item) is not cls for item in values):
                raise TypeError("construction.%s requires exact %s values" % (name, cls.__name__))
            object.__setattr__(self, name, values)
        outputs = tuple(self.outputs)
        if any(not isinstance(item, Mapping) or not item for item in outputs):
            raise TypeError("construction.outputs requires nonempty representation descriptors")
        object.__setattr__(self, "outputs", tuple(freeze(item) for item in outputs))
        def support(descriptor):
            if "support" in descriptor:
                return descriptor["support"]
            base = descriptor.get("base")
            return support(base) if isinstance(base, Mapping) else None

        target_supports = tuple(support(item) for item in self.outputs
                                if support(item) is not None)
        for access in self.inputs:
            origin = support(access.representation)
            if origin is not None and target_supports and any(
                    origin != target for target in target_supports) and access.physical_map is None:
                raise ValueError("different physical supports require an explicit physical map")
        if type(self.stencil_radius) is not int or self.stencil_radius < 0:
            raise ValueError("stencil radius must be a non-negative integer")
        if not set(self.materialized_inputs) <= set(self.dependencies):
            raise ValueError("materialized input must name a dependency")
        for item in self.materialized_inputs:
            if "halo_exchange:%s" % item not in self.communication:
                raise ValueError("materialized stencil input requires its halo exchange")
        if self.native_route is not None:
            identifier(self.native_route, "native route")
            if self.refusal is not None:
                raise ValueError("an available route cannot also have a refusal")
        elif self.refusal is None:
            object.__setattr__(self, "refusal", "native_realization_unavailable")
        if self.refusal is not None:
            identifier(self.refusal, "native refusal")
        groups = tuple(identifiers(group, "decomposition group") for group in self.decomposition)
        flattened = [item for group in groups for item in group]
        if groups and (len(set(flattened)) != len(flattened)
                       or set(flattened) != set(self.consumes) or any(not g for g in groups)):
            raise ValueError("declared decomposition must exactly partition consumed occurrences")
        object.__setattr__(self, "decomposition", groups)
        object.__setattr__(self, "guarantees", freeze(self.guarantees))
        if self.reductions and "collective" not in self.effects:
            raise ValueError("reduction construction must declare its collective effect")

    @property
    def disposition(self) -> str:
        return "resolved" if self.native_route is not None else "unsupported"

    def require_partition(self, requested: Any) -> None:
        requested = identifiers(requested, "requested partition")
        if set(requested) == set(self.consumes):
            return
        if not any(set(requested) == set(group) for group in self.decomposition):
            raise ValueError("joint construction has no declared realization for this partition")

    @classmethod
    def from_data(cls, value: Any) -> NumericalConstruction:
        values = record_data(cls, value)
        values["inputs"] = tuple(ResolvedAccess.from_data(item) for item in values["inputs"])
        values["exchanges"] = tuple(ExchangeRecord.from_data(item) for item in values["exchanges"])
        return cls(**values)


# Public spelling used by compiler consumers; an operation is a resolved construction,
# not another independently authored physical operator.
ResolvedOperation = NumericalConstruction


def require_semantic_rewrite(kind: str, *, spatially_constant_coefficient: bool = False) -> None:
    """Gate the specifically unsafe rewrites before a consumer changes the computation."""
    if type(spatially_constant_coefficient) is not bool:
        raise TypeError("constant-coefficient witness must be bool")
    if kind == "move_coefficient_through_divergence" and spatially_constant_coefficient:
        return
    if kind in {"move_coefficient_through_divergence", "discrete_accumulation_chain_rule",
                "infer_inverse_physical_map"}:
        raise ValueError("scientific normalization refuses %s without its semantic contract" % kind)
    raise ValueError("unregistered scientific normalization %r" % kind)
