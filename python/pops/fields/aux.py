"""Owner-qualified auxiliary component producers.

An auxiliary component is declared by :class:`pops.model.Module` and referenced by its registry-
issued ``Handle``.  This module describes how that exact component is produced; it never assigns a
global numeric slot and contains no physical field names.  Resolve/bind owns compact slot allocation
through the component ``ProviderPack``.

``InputAux`` declares externally supplied accepted data. ``DerivedAux`` carries a PoPS expression
that code generation lowers to a native provider.  Derived values are fresh with respect to their
dependency generations and are recomputed after regrid/restart; users do not choose an execution
phase that could make a dependency stale.
"""

from __future__ import annotations

from typing import Any

from pops._ir.expr import Expr
from pops.descriptors import Descriptor
from pops.descriptors_report import CapabilitySet, RequirementSet
from pops.model import Handle

from ._identity import strict_field_data
from ._references import reference_label, resolve_handle


class AuxiliaryBoundary:
    """Owner-qualified ghost/boundary policy for one auxiliary component.

    ``inherit`` delegates periodic faces to the mesh topology and non-periodic
    faces to the owning physical boundary provider.  ``foextrap`` and
    ``dirichlet`` are explicit component-local physical policies.  The width
    is a scalar radius applied on every native axis, so the declaration stays
    rank-generic; the native route expands it after the selected dimension is
    known.
    """

    __slots__ = ("width", "kind", "value")
    __pops_ir_immutable__ = True

    def __init__(self, *, width: Any = 0, kind: Any = "inherit", value: Any = None) -> None:
        if isinstance(width, bool) or not isinstance(width, int) or width < 0:
            raise ValueError("AuxiliaryBoundary width must be a non-negative int")
        if kind not in {"inherit", "foextrap", "dirichlet"}:
            raise ValueError("AuxiliaryBoundary kind must be inherit, foextrap, or dirichlet")
        if kind == "dirichlet" and value is None:
            raise ValueError("AuxiliaryBoundary dirichlet policy requires a value")
        if kind != "dirichlet" and value is not None:
            raise ValueError("AuxiliaryBoundary value is valid only for dirichlet")
        if kind == "dirichlet":
            from pops.identity.scalar import scalar_cpp
            try:
                scalar_cpp(value)
            except (TypeError, ValueError) as exc:
                raise TypeError("AuxiliaryBoundary dirichlet value must be a finite scalar") from exc
        object.__setattr__(self, "width", width)
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "value", value)

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("AuxiliaryBoundary is immutable")

    def to_data(self) -> dict[str, Any]:
        return {"width": self.width, "kind": self.kind, "value": self.value}


def _aux_boundary(value: Any) -> AuxiliaryBoundary:
    if value is None:
        return AuxiliaryBoundary()
    if type(value) is not AuxiliaryBoundary:
        raise TypeError("auxiliary boundary must be an AuxiliaryBoundary")
    return value


def _aux_target(value: Any) -> Handle:
    if not isinstance(value, Handle) or value.kind != "aux":
        raise TypeError(
            "auxiliary producer target must be a Module.aux_field(...) Handle; "
            "names/strings and FieldSpace handles are not auxiliary identities"
        )
    return value


class _AuxProducer(Descriptor):
    """Immutable producer metadata shared by input and derived components."""

    __pops_ir_immutable__ = True
    category = "aux_provider"
    producer_kind = ""
    restart_policy = ""
    regrid_policy = ""
    _target: Handle
    boundary: AuxiliaryBoundary

    def __init__(self, target: Any, *, boundary: Any = None) -> None:
        object.__setattr__(self, "_target", _aux_target(target))
        object.__setattr__(self, "boundary", _aux_boundary(boundary))

    @property
    def name(self) -> str:
        return self._target.local_id

    @property
    def target(self) -> Handle:
        return self._target

    def _base_options(self) -> dict[str, Any]:
        return {
            "target": reference_label(self._target, where="auxiliary producer target"),
            "producer": self.producer_kind,
            "freshness": "dependency_generation",
            "restart": self.restart_policy,
            "regrid": self.regrid_policy,
            "boundary": self.boundary.to_data(),
        }

    def capabilities(self) -> CapabilitySet:
        return CapabilitySet(
            {
                "owner_qualified": True,
                "compact_slot": True,
                "transactional_publication": True,
            }
        )

    def freeze(self) -> _AuxProducer:
        return self

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("%s is immutable" % type(self).__name__)

    def __delattr__(self, name: str) -> None:
        raise AttributeError("%s is immutable" % type(self).__name__)


class InputAux(_AuxProducer):
    """Declare one external input component.

    The descriptor identifies the target only.  Its ranked scalar/field value belongs to bind
    inputs, not to the compiled artifact, so large simulation data is never serialized into a
    descriptor identity.
    """

    producer_kind = "input"
    restart_policy = "persist"
    regrid_policy = "transfer"

    def __init__(self, target: Any, *, boundary: Any = None) -> None:
        super().__init__(target, boundary=boundary)

    def options(self) -> dict[str, Any]:
        return self._base_options()

    def requirements(self) -> RequirementSet:
        return RequirementSet(
            {"aux_input": reference_label(self._target, where="auxiliary input target")}
        )

    def declaration_references(self) -> tuple[Handle, ...]:
        return (self._target,)

    def resolve_references(self, resolver: Any) -> InputAux:
        return InputAux(
            resolve_handle(self._target, resolver, where="auxiliary input target"),
            boundary=self.boundary,
        )

    def to_data(self) -> dict[str, Any]:
        return {
            "type": type(self).__name__,
            **self._base_options(),
            "target": self._target.canonical_identity(),
        }


class DerivedAux(_AuxProducer):
    """Declare one native derived component from an immutable PoPS expression."""

    producer_kind = "derived"
    restart_policy = "recompute"
    regrid_policy = "recompute"
    expression: Expr

    def __init__(self, target: Any, expression: Any, *, boundary: Any = None) -> None:
        if not isinstance(expression, Expr):
            raise TypeError("DerivedAux expression must be a pops Expr")
        super().__init__(target, boundary=boundary)
        object.__setattr__(self, "expression", expression)

    def options(self) -> dict[str, Any]:
        return {**self._base_options(), "expression": strict_field_data(self.expression)}

    def requirements(self) -> RequirementSet:
        return RequirementSet(
            {
                "aux_dependencies": tuple(
                    reference_label(reference, where="derived auxiliary dependency")
                    for reference in self.expression.declaration_references()
                )
            }
        )

    def capabilities(self) -> CapabilitySet:
        values = super().capabilities().to_dict()
        values.update({"derived": True, "native_kernel": True})
        return CapabilitySet(values)

    def declaration_references(self) -> tuple[Handle, ...]:
        references = [self._target]
        for reference in self.expression.declaration_references():
            if reference not in references:
                references.append(reference)
        return tuple(references)

    def resolve_references(self, resolver: Any) -> DerivedAux:
        return DerivedAux(
            resolve_handle(self._target, resolver, where="derived auxiliary target"),
            self.expression.resolve_references(resolver),
            boundary=self.boundary,
        )

    def to_data(self) -> dict[str, Any]:
        return {
            "type": type(self).__name__,
            **self._base_options(),
            "target": self._target.canonical_identity(),
            "expression": strict_field_data(self.expression),
        }


__all__ = ["AuxiliaryBoundary", "DerivedAux", "InputAux"]
