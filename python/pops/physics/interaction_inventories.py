"""Explicit physical inventory maps and accepted interaction quadrature checks.

These maps certify the composition's declared weights, not a local law merely
because it has opposite arrays. Numerical conservation still needs the executed
native law and accepted-update evidence.
"""
from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping
from fractions import Fraction
from typing import Any

from pops.identity.scalar import exact_numeric_scalar, scalar_data


@dataclass(frozen=True, slots=True)
class InventoryProjection:
    state: Any
    weights: tuple[Any, ...]
    __pops_ir_immutable__ = True

    def __post_init__(self):
        from pops.model.handles import Handle
        from pops.model.spaces import StateSpace
        space = getattr(self.state, "space", None)
        if not isinstance(self.state, Handle) or self.state.kind != "state" or not isinstance(space, StateSpace):
            raise TypeError("inventory projection requires a qualified StateHandle")
        if type(self.weights) is not tuple or len(self.weights) != len(space.components):
            raise ValueError("inventory projection must match its own exact state component shape")
        object.__setattr__(self, "weights", tuple(exact_numeric_scalar(v, where="inventory map weight")
                                                 for v in self.weights))

    def declaration_references(self):
        return (self.state,)

    def resolve_references(self, resolver):
        return type(self)(resolver(self.state), self.weights)

    def to_data(self):
        from pops._ir.balance import _handle_data
        return {"state": _handle_data(self.state), "weights": [scalar_data(v) for v in self.weights]}


@dataclass(frozen=True, slots=True)
class PhysicalInventoryMap:
    name: str
    projections: tuple[InventoryProjection, ...]
    __pops_ir_immutable__ = True

    def __post_init__(self):
        if type(self.name) is not str or not self.name:
            raise TypeError("physical inventory map requires an explicit name")
        if type(self.projections) is not tuple or not self.projections or any(
                type(p) is not InventoryProjection for p in self.projections):
            raise TypeError("physical inventory maps require immutable typed projections")
        if len({p.state for p in self.projections}) != len(self.projections):
            raise ValueError("physical inventory map contains duplicate state projections")

    def declaration_references(self):
        return tuple(p.state for p in self.projections)

    def resolve_references(self, resolver):
        return type(self)(self.name, tuple(p.resolve_references(resolver) for p in self.projections))

    def to_data(self):
        return {"name": self.name, "projections": [p.to_data() for p in self.projections]}

    def application_expression(self, application):
        """The declared common inventory rate, retaining the original joint application."""
        from pops._ir.expr import Const
        expression = Const(0)
        for projection in self.projections:
            values = application[projection.state]
            for weight, value in zip(projection.weights, values, strict=True):
                expression = expression + weight * value
        return expression

    def require_accepted_quadrature(self, application, recipient_weights, *, multiplicities=None):
        """Require the same effective accepted weight for every nonzero recipient map.

        Distinct signed occurrences remain the ledger's authority. Multiplicity is
        explicit here so sharing a calculation cannot falsely certify a twice-used
        contribution against a once-used opposite recipient.
        """
        if not isinstance(recipient_weights, Mapping):
            raise TypeError("accepted quadrature requires exact state-handle weights")
        states = {p.state for p in self.projections}
        if set(recipient_weights) != states:
            raise ValueError("accepted quadrature recipients differ from physical inventory map")
        if multiplicities is None:
            multiplicities = {state: 1 for state in states}
        if set(multiplicities) != states or any(type(v) is not int or v < 1 for v in multiplicities.values()):
            raise ValueError("accepted interaction multiplicities require positive exact integers")
        effective = []
        for projection in self.projections:
            application[projection.state]  # authenticate output target and its RateSpace
            weight = exact_numeric_scalar(recipient_weights[projection.state], where="accepted quadrature weight")
            if any(value != 0 for value in projection.weights):
                effective.append(Fraction(weight) * multiplicities[projection.state])
        if len(set(effective)) > 1:
            raise ValueError("recipient quadrature/multiplicity is nonconservative for the declared inventory")
        return {"inventory": self.to_data(), "application": application.to_data(),
                "accepted_weight": scalar_data(effective[0] if effective else 0),
                "claim": "compatible_quadrature_only; native_law_conservation_requires_evidence"}


__all__ = ["InventoryProjection", "PhysicalInventoryMap"]
