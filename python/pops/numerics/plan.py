"""Family-organized numerical authoring and its owner-qualified resolved value."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Callable

from pops.descriptors import Descriptor
from pops.identity import Identity, semantic_identity
from pops.model import Handle, OperatorHandle
from pops.numerics.spatial import FiniteVolume


def _callable_projection(value: Any, where: str) -> dict[str, Any]:
    for name in ("to_data", "canonical_identity"):
        method = getattr(value, name, None)
        if callable(method):
            result = method()
            if not isinstance(result, dict):
                raise TypeError("%s.%s() must return a mapping" % (where, name))
            return result
    inspect = getattr(value, "inspect", None)
    if callable(inspect):
        result = inspect()
        if isinstance(result, dict):
            return result
    raise TypeError("%s has no canonical data projection" % where)


class _RateFamily:
    """Unique rate-to-method bindings; expressions can never become mapping keys."""

    def __init__(self) -> None:
        self._rows: list[tuple[OperatorHandle, FiniteVolume]] | tuple[tuple[OperatorHandle, FiniteVolume], ...] = []
        self._frozen = False

    def add(self, rate: Any, method: Any) -> None:
        if self._frozen:
            raise RuntimeError("DiscretizationPlan.rates is frozen")
        if not isinstance(rate, OperatorHandle) or rate.kind not in {"local_rate", "coupled_rate"}:
            raise TypeError("rates.add requires a typed rate OperatorHandle")
        if type(method) is not FiniteVolume:
            raise TypeError("rates.add currently requires an exact FiniteVolume descriptor")
        if any(existing == rate for existing, _ in self._rows):
            raise ValueError("rate %s already has a numerical method" % rate.qualified_id)
        self._rows.append((rate, method))

    def items(self) -> tuple[tuple[OperatorHandle, FiniteVolume], ...]:
        return tuple(self._rows)

    def freeze(self) -> None:
        if self._frozen:
            return
        for _, method in self._rows:
            method.freeze()
        self._rows = tuple(self._rows)
        self._frozen = True


class _PairFamily:
    """Unique physical/numerical descriptor pairs for a named numerical family."""

    def __init__(self, family: str) -> None:
        self.family = family
        self._rows: list[tuple[Any, Any]] | tuple[tuple[Any, Any], ...] = []
        self._frozen = False

    def add(self, subject: Any, method: Any) -> None:
        if self._frozen:
            raise RuntimeError("DiscretizationPlan.%s is frozen" % self.family)
        if subject is None or method is None:
            raise TypeError("%s.add requires an explicit subject and method" % self.family)
        if any(existing == subject for existing, _ in self._rows):
            raise ValueError("%s subject already has a numerical method" % self.family)
        self._rows.append((subject, method))

    def items(self) -> tuple[tuple[Any, Any], ...]:
        return tuple(self._rows)

    def freeze(self) -> None:
        if self._frozen:
            return
        for subject, method in self._rows:
            for value in (subject, method):
                freeze = getattr(value, "freeze", None)
                if callable(freeze):
                    freeze()
        self._rows = tuple(self._rows)
        self._frozen = True


class _ValueFamily:
    """Register-once descriptors such as boundary/interface authorities."""

    def __init__(self, family: str) -> None:
        self.family = family
        self._rows: list[Any] | tuple[Any, ...] = []
        self._frozen = False

    def add(self, value: Any) -> None:
        if self._frozen:
            raise RuntimeError("DiscretizationPlan.%s is frozen" % self.family)
        if value is None or not any(callable(getattr(value, name, None)) for name in (
                "to_data", "canonical_identity", "inspect")):
            raise TypeError("%s.add requires a typed inspectable descriptor" % self.family)
        if any(existing is value or existing == value for existing in self._rows):
            raise ValueError("duplicate %s authority" % self.family)
        self._rows.append(value)

    def values(self) -> tuple[Any, ...]:
        return tuple(self._rows)

    def freeze(self) -> None:
        if self._frozen:
            return
        for value in self._rows:
            freeze = getattr(value, "freeze", None)
            if callable(freeze):
                freeze()
        self._rows = tuple(self._rows)
        self._frozen = True


@dataclass(frozen=True, slots=True)
class ResolvedRateMethod:
    rate: OperatorHandle
    method: FiniteVolume

    def __post_init__(self) -> None:
        if not isinstance(self.rate, OperatorHandle) or not self.rate.is_resolved:
            raise TypeError("ResolvedRateMethod.rate must be owner-qualified")
        if type(self.method) is not FiniteVolume or not self.method.flux.is_resolved:
            raise TypeError("ResolvedRateMethod.method must have resolved physical handles")
        self.method.freeze()

    def to_data(self) -> dict[str, Any]:
        return {"rate": self.rate.canonical_identity(), "method": self.method.to_data()}


@dataclass(frozen=True, slots=True)
class ResolvedDiscretizationPlan:
    block: Handle
    rates: tuple[ResolvedRateMethod, ...]
    fields: tuple[Any, ...]
    boundaries: tuple[Any, ...]
    sources: tuple[Any, ...]
    interfaces: tuple[Any, ...]
    identity: Identity = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.block, Handle) or self.block.kind != "block" or not self.block.is_resolved:
            raise TypeError("resolved numerical plan requires a canonical BlockHandle")
        if not self.rates:
            raise ValueError("resolved numerical plan has no rate method for its block")
        for name in ("rates", "fields", "boundaries", "sources", "interfaces"):
            if not isinstance(getattr(self, name), tuple):
                raise TypeError("ResolvedDiscretizationPlan.%s must be a tuple" % name)
        object.__setattr__(self, "identity", semantic_identity(self._payload()))

    def _payload(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "block": self.block.canonical_identity(),
            "rates": [row.to_data() for row in self.rates],
            "fields": [_callable_projection(row, "resolved field") for row in self.fields],
            "boundaries": [_callable_projection(row, "resolved boundary") for row in self.boundaries],
            "sources": [_callable_projection(row, "resolved source") for row in self.sources],
            "interfaces": [_callable_projection(row, "resolved interface") for row in self.interfaces],
        }

    def to_data(self) -> dict[str, Any]:
        return {**self._payload(), "identity": self.identity.token}

    def primary_spatial(self) -> FiniteVolume:
        """Compatibility projection for the current per-block native spatial ABI.

        The resolved plan retains every per-rate binding. The native engine currently accepts one
        spatial method per block, so distinct methods fail explicitly instead of picking the first.
        """
        methods = [row.method for row in self.rates]
        first = methods[0].to_data()
        if any(method.to_data() != first for method in methods[1:]):
            raise ValueError(
                "native runtime requires one finite-volume method per block; resolved rates select "
                "distinct methods and cannot be lowered without a per-operator native ABI"
            )
        return methods[0]


class DiscretizationPlan(Descriptor):
    """One family-organized numerical authority, reusable across Model instances."""

    category = "discretization_plan"

    def __init__(self) -> None:
        self.rates = _RateFamily()
        self.fields = _PairFamily("fields")
        self.boundaries = _ValueFamily("boundaries")
        self.sources = _PairFamily("sources")
        self.interfaces = _ValueFamily("interfaces")

    def options(self) -> dict[str, Any]:
        return {
            "rates": self.rates.items(),
            "fields": self.fields.items(),
            "boundaries": self.boundaries.values(),
            "sources": self.sources.items(),
            "interfaces": self.interfaces.values(),
        }

    def freeze(self) -> "DiscretizationPlan":
        if getattr(self, "_frozen", False):
            return self
        for family in (self.rates, self.fields, self.boundaries, self.sources, self.interfaces):
            family.freeze()
        object.__setattr__(self, "_frozen", True)
        return self

    def validate_for(self, model: Any) -> bool:
        contracts = getattr(model, "_rate_contracts", None)
        if not isinstance(contracts, Mapping):
            raise TypeError("DiscretizationPlan requires a Model exposing typed rate contracts")
        selected = {rate: method for rate, method in self.rates.items()
                    if rate.owner_path == model.owner_path}
        expected = set(contracts)
        missing, extra = expected - set(selected), set(selected) - expected
        if missing or extra:
            raise ValueError(
                "DiscretizationPlan rate coverage mismatch for Model %r: missing=%s extra=%s"
                % (model.name, sorted(row.local_id for row in missing),
                   sorted(row.local_id for row in extra)))
        if not selected:
            raise ValueError("DiscretizationPlan has no rate binding for Model %r" % model.name)
        for rate, method in selected.items():
            contract = model.rate_contract(rate)
            if contract["flux"] != method.flux:
                raise ValueError(
                    "FiniteVolume flux does not match the physical flux referenced by rate %r"
                    % rate.local_id)
            state = method.variables.options.get("state")
            if state is not None and state != contract["state"]:
                raise ValueError(
                    "FiniteVolume variables do not reference the state differentiated by rate %r"
                    % rate.local_id)
            method.validate()
        return True

    def resolve_for(self, case: Any, block: Any) -> ResolvedDiscretizationPlan:
        model = case._block_registry.spec(block.local_id)["model"]
        self.validate_for(model)

        def resolve_handle(value: Handle) -> Handle:
            try:
                return case.resolve(value, block=block)
            except TypeError:
                return case.resolve(value)

        rates = []
        for rate, method in self.rates.items():
            if rate.owner_path != model.owner_path:
                continue
            rates.append(ResolvedRateMethod(
                case.resolve(rate, block=block), method.resolve_references(resolve_handle)))
        resolved_block = case.resolve(block)
        return ResolvedDiscretizationPlan(
            resolved_block,
            tuple(sorted(rates, key=lambda row: row.rate.qualified_id)),
            tuple(self.fields.items()),
            tuple(self.boundaries.values()),
            tuple(self.sources.items()),
            tuple(self.interfaces.values()),
        )

    def inspect(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "category": self.category,
            "rates": [
                {"rate": rate.qualified_id, "method": method.inspect()}
                for rate, method in self.rates.items()
            ],
            "fields": len(self.fields.items()),
            "boundaries": len(self.boundaries.values()),
            "sources": len(self.sources.items()),
            "interfaces": len(self.interfaces.values()),
        }


__all__ = [
    "DiscretizationPlan", "ResolvedDiscretizationPlan", "ResolvedRateMethod",
]
