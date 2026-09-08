"""Retained physical balances, occurrence-preserving views and accumulation.

These records express scientific equations. A numerical construction may consume a
view, but cannot alter its coefficients or manufacture a second physical equation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pops.identity.scalar import exact_numeric_scalar, scalar_data
from pops.model.handles import Handle


def _handle_data(handle: Handle) -> dict[str, Any]:
    from .quantity import _hash_owner
    if handle.owner_path == _hash_owner.get():
        result = {"kind": handle.kind, "local_id": handle.local_id,
                  "schema_version": handle.schema_version}
        target = getattr(handle, "registered_operator_name", None)
        if target is not None:
            result["registered_operator_name"] = target
        return result
    return handle.canonical_identity() if handle.is_resolved else handle.inspect()


@dataclass(frozen=True, slots=True, eq=False)
class Accumulation:
    """Discrete map from solver coordinates to the stored balance quantity.

    Nonidentity maps require both representation and quadrature contracts. This
    record never applies the law to an average or replaces it with its derivative.
    """

    state: Handle
    coordinates: Any = None
    law: Any = None
    representation: Any = None
    quadrature: Any = None
    __pops_ir_immutable__ = True

    def __post_init__(self) -> None:
        from .expr import Expr
        from .symbolic import freeze_symbolic_metadata

        if not isinstance(self.state, Handle) or self.state.kind != "state":
            raise TypeError("accumulation requires a declared state Handle")
        if self.law is None:
            if any(value is not None for value in (
                    self.coordinates, self.representation, self.quadrature)):
                raise ValueError("identity accumulation is inferred without coordinate metadata")
            return
        if self.coordinates is None or self.representation is None or self.quadrature is None:
            raise ValueError(
                "nonlinear accumulation requires coordinates, representation and quadrature")
        laws = self.law if isinstance(self.law, (tuple, list)) else (self.law,)
        if not laws or any(not isinstance(law, Expr) for law in laws):
            raise TypeError("accumulation law must contain immutable symbolic expressions")
        for name in ("coordinates", "law", "representation", "quadrature"):
            object.__setattr__(self, name, freeze_symbolic_metadata(getattr(self, name)))

    @property
    def is_identity(self) -> bool:
        return self.law is None

    def temporal_residual(self, candidate: Any, previous: Any, *, interval: Any,
                          rate: Any) -> DiscreteAccumulationEquation:
        return DiscreteAccumulationEquation(self, candidate, previous, interval, rate)

    def declaration_references(self) -> tuple[Handle, ...]:
        from .expr_references import collect_reference_value
        result: list[Handle] = [self.state]
        for value in (self.coordinates, self.law, self.representation, self.quadrature):
            collect_reference_value(value, result, set())
        return tuple(result)

    def resolve_references(self, resolver: Any) -> Accumulation:
        from .expr_references import resolve_reference_value
        return Accumulation(*(resolve_reference_value(value, resolver, {}) for value in (
            self.state, self.coordinates, self.law, self.representation, self.quadrature)))

    def to_data(self) -> dict[str, Any]:
        from pops.model.hash_data import canonical_hash_data
        return {
            "kind": "identity" if self.is_identity else "discrete_constitutive_map",
            "state": _handle_data(self.state),
            "coordinates": canonical_hash_data(self.coordinates),
            "law": canonical_hash_data(self.law),
            "representation": canonical_hash_data(self.representation),
            "quadrature": canonical_hash_data(self.quadrature),
        }


@dataclass(frozen=True, slots=True, eq=False)
class DiscreteAccumulationEquation:
    """The exact temporal relation Q_kappa(q+) - U^n = tau R_kappa."""

    accumulation: Accumulation
    candidate: Any
    previous: Any
    interval: Any
    rate: Any
    __pops_ir_immutable__ = True

    def __post_init__(self) -> None:
        from .symbolic import freeze_symbolic_metadata
        for name in ("candidate", "previous", "interval", "rate"):
            object.__setattr__(self, name, freeze_symbolic_metadata(getattr(self, name)))


@dataclass(frozen=True, slots=True, eq=False)
class BalanceOccurrence:
    """One signed occurrence; repeated payloads have distinct ordinal identities."""

    balance: Handle
    ordinal: int
    kind: str
    payload: Any
    coefficient: Any
    target: Handle
    __pops_ir_immutable__ = True

    def __post_init__(self) -> None:
        if not isinstance(self.balance, Handle) or self.balance.kind != "local_rate":
            raise TypeError("balance occurrence requires a local_rate Handle")
        if isinstance(self.ordinal, bool) or not isinstance(self.ordinal, int) or self.ordinal < 0:
            raise ValueError("balance occurrence ordinal must be a nonnegative integer")
        if self.kind not in ("flux", "diffusion", "source", "projection"):
            raise ValueError("unknown physical balance term kind %r" % self.kind)
        if not isinstance(self.target, Handle) or self.target.kind != "state":
            raise TypeError("balance occurrence target must be a state Handle")
        if self.kind == "projection" and getattr(self.payload, "target", None) != self.target:
            raise ValueError("a joint rate projection must target the differentiated quantity")
        object.__setattr__(self, "coefficient", exact_numeric_scalar(
            self.coefficient, where="balance occurrence coefficient"))

    @property
    def identity(self) -> tuple[Handle, int]:
        return self.balance, self.ordinal

    @property
    def sign(self) -> int:
        return (self.coefficient > 0) - (self.coefficient < 0)

    @property
    def scale(self) -> Any:
        return abs(self.coefficient)

    def to_data(self) -> dict[str, Any]:
        payload = (_handle_data(self.payload) if isinstance(self.payload, Handle)
                   else self.payload.to_data())
        return {"balance": _handle_data(self.balance), "ordinal": self.ordinal,
                "kind": self.kind, "payload": payload,
                "coefficient": scalar_data(self.coefficient), "target": _handle_data(self.target)}


@dataclass(frozen=True, slots=True, eq=False)
class Balance:
    """The single retained scientific equation, before numerical partitioning."""

    handle: Handle
    target: Handle
    occurrences: tuple[BalanceOccurrence, ...]
    accumulation: Accumulation
    __pops_ir_immutable__ = True

    def __post_init__(self) -> None:
        occurrences = tuple(self.occurrences)
        if self.accumulation.state != self.target:
            raise ValueError("balance accumulation and differentiated target must agree")
        for index, item in enumerate(occurrences):
            if (not isinstance(item, BalanceOccurrence) or item.balance != self.handle
                    or item.ordinal != index or item.target != self.target):
                raise ValueError("balance occurrences must retain their exact owner, target and order")
        object.__setattr__(self, "occurrences", occurrences)

    @classmethod
    def capture(cls, handle: Handle, target: Handle, terms: Any,
                accumulation: Accumulation | None = None) -> Balance:
        return cls(handle, target, tuple(
            BalanceOccurrence(handle, index, kind, payload, coefficient, target)
            for index, (kind, payload, coefficient) in enumerate(terms)),
            Accumulation(target) if accumulation is None else accumulation)

    def select(self, *selectors: Any) -> BalanceView:
        return self.full_view().select(*selectors)

    def full_view(self) -> BalanceView:
        return BalanceView(self, tuple(range(len(self.occurrences))))

    def declaration_references(self) -> tuple[Handle, ...]:
        from .expr_references import collect_reference_value
        result = [self.handle, self.target]
        for item in self.occurrences:
            collect_reference_value(item.payload, result, set())
        collect_reference_value(self.accumulation, result, set())
        return tuple(result)

    def resolve_references(self, resolver: Any) -> Balance:
        from .expr_references import resolve_reference_value
        memo: dict[int, Any] = {}
        def resolve(value):
            return resolve_reference_value(value, resolver, memo)
        return Balance.capture(resolve(self.handle), resolve(self.target),
            ((item.kind, resolve(item.payload), item.coefficient) for item in self.occurrences),
            self.accumulation.resolve_references(resolver))

    def to_data(self) -> dict[str, Any]:
        return {"schema": "pops.physical_balance.v1", "handle": _handle_data(self.handle),
                "target": _handle_data(self.target),
                "occurrences": [item.to_data() for item in self.occurrences],
                "accumulation": self.accumulation.to_data()}


@dataclass(frozen=True, slots=True, eq=False)
class BalanceView:
    """A subset of occurrences in one balance, retaining source order and multiplicity."""

    balance: Balance
    ordinals: tuple[int, ...]
    __pops_ir_immutable__ = True

    def __post_init__(self) -> None:
        ordinals = tuple(self.ordinals)
        if (any(isinstance(index, bool) or not isinstance(index, int)
                or index not in range(len(self.balance.occurrences)) for index in ordinals)
                or tuple(sorted(set(ordinals))) != ordinals):
            raise ValueError("a balance view requires unique occurrence ordinals in source order")
        object.__setattr__(self, "ordinals", ordinals)

    @property
    def occurrences(self) -> tuple[BalanceOccurrence, ...]:
        return tuple(self.balance.occurrences[index] for index in self.ordinals)

    @property
    def target(self) -> Handle:
        return self.balance.target

    @property
    def accumulation(self) -> Accumulation:
        return self.balance.accumulation

    def select(self, *selectors: Any) -> BalanceView:
        if not selectors:
            raise ValueError("select requires a physical term or an exact occurrence")
        selected: set[int] = set()
        for selector in selectors:
            if isinstance(selector, BalanceOccurrence):
                matches = [item.ordinal for item in self.occurrences if item is selector]
            else:
                matches = [item.ordinal for item in self.occurrences
                           if item.payload is selector or (isinstance(item.payload, Handle)
                                                          and item.payload == selector)
                           or (callable(getattr(item.payload, "same_projection", None))
                               and item.payload.same_projection(selector))]
            if not matches:
                raise ValueError("select cannot add a term absent from this physical balance view")
            selected.update(matches)
        return BalanceView(self.balance, tuple(index for index in self.ordinals if index in selected))

    def legacy_incompatibility(self) -> str | None:
        if any(item.kind == "projection" for item in self.occurrences):
            return "joint rate application projections require a native interaction realization (M4)"
        if not self.accumulation.is_identity:
            return "nonidentity accumulation requires a discrete temporal solve construction"
        fluxes = [item for item in self.occurrences if item.kind == "flux"]
        if len(fluxes) > 1:
            return "multiple divergence occurrences require an explicit numerical coverage construction"
        if any(item.coefficient != -1 for item in fluxes):
            return "signed or scaled divergence requires a matching numerical construction"
        if any(item.kind != "flux" and (item.kind != "source" or item.coefficient != 1)
               for item in self.occurrences):
            return "signed, scaled or joint source occurrences require a matching numerical construction"
        return None

    def declaration_references(self) -> tuple[Handle, ...]:
        return self.balance.declaration_references()

    def resolve_references(self, resolver: Any) -> BalanceView:
        return BalanceView(self.balance.resolve_references(resolver), self.ordinals)

    def to_data(self) -> dict[str, Any]:
        return {"balance": self.balance.to_data(), "ordinals": list(self.ordinals)}


def accumulation(state: Handle, *, coordinates: Any, law: Any,
                 representation: Any, quadrature: Any) -> Accumulation:
    """Declare Q_kappa explicitly for ``ddt(accumulation(...)) == rhs``."""
    return Accumulation(state, coordinates, law, representation, quadrature)


__all__ = ["Accumulation", "Balance", "BalanceOccurrence", "BalanceView",
           "DiscreteAccumulationEquation", "accumulation"]
