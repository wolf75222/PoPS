"""Typed spaces of the operator-first type system (Spec 2).

Defines the abstract spaces a model-free ``pops.time.Program`` composes:
``StateSpace`` (the components of ``U``), ``FieldSpace`` (auxiliary / solved
fields), ``RateSpace`` / ``Rate(U)`` (the tangent of a ``StateSpace``), plus the
``AuxSpace`` declarations a Module owns. Canonical scalar parameter declarations
live in :mod:`pops.params` and are referenced by ``ParamHandle``. These carry no
numerics and no array data; they are a TYPED VIEW only.

Imports only the standard library so it can be exercised without the compiled
``_pops`` extension.
"""
from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Any


def _freeze_metadata(value: Any) -> Any:
    """Deep-freeze inert type metadata carried across Program snapshots."""
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze_metadata(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_metadata(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(_freeze_metadata(item) for item in value)
    return value


def _metadata_key(value: Any) -> Any:
    """Hashable structural key for deeply frozen descriptor metadata."""
    if isinstance(value, Mapping):
        return tuple(sorted((str(key), _metadata_key(item)) for key, item in value.items()))
    if isinstance(value, tuple):
        return tuple(_metadata_key(item) for item in value)
    if isinstance(value, frozenset):
        return tuple(sorted((_metadata_key(item) for item in value), key=repr))
    return value


def _semantic_tag(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("%s must be a non-empty string" % where)
    return value


def _component_units(components: tuple[str, ...], units: Any) -> tuple[str | None, ...]:
    """Normalize units into component order; mapping insertion order is never semantic."""
    if units is None:
        return (None,) * len(components)
    if isinstance(units, Mapping):
        unknown = sorted(set(units) - set(components))
        missing = sorted(set(components) - set(units))
        if unknown or missing:
            raise ValueError(
                "Space units must name exactly the components (unknown=%r, missing=%r)"
                % (unknown, missing)
            )
        values = tuple(units[name] for name in components)
    else:
        values = tuple(units)
        if len(values) != len(components):
            raise ValueError("Space units must have one entry per component")
    for value in values:
        if value is not None and (not isinstance(value, str) or not value):
            raise ValueError("Space units entries must be non-empty strings or None")
    return values


class _ImmutableTypeValue:
    """Small value-object base: authoring type descriptors are immutable once built."""

    __pops_ir_immutable__ = True

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("%s is immutable" % type(self).__name__)

    def __delattr__(self, name: str) -> None:
        raise AttributeError("%s is immutable" % type(self).__name__)


class Space(_ImmutableTypeValue):
    """Base of a typed space: a kind, a name and an ordered tuple of components.

    Equality and hashing are by ``(kind, name, components, layout)`` so two spaces
    built independently from the same model compare equal (used by Program type
    checks). Subclasses may extend that structural identity with their own lowering-
    relevant metadata (for example ``StateSpace.storage`` and ``StateSpace.roles``).
    """

    kind = "space"

    def __init__(
        self,
        name: Any,
        components: Any = (),
        layout: str = "cell",
        *,
        representation: Any = None,
        centering: Any = None,
        units: Any = None,
        frame: Any = "model",
        clock: Any = "simulation",
    ) -> None:
        if not isinstance(name, str) or not name:
            raise ValueError("Space name must be a non-empty string")
        normalized = tuple(components)
        if any(not isinstance(component, str) or not component for component in normalized):
            raise ValueError("Space components must be non-empty strings")
        if len(set(normalized)) != len(normalized):
            raise ValueError("Space components must be unique")
        if not isinstance(layout, str) or not layout:
            raise ValueError("Space layout must be a non-empty string")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "components", normalized)
        object.__setattr__(self, "layout", layout)
        object.__setattr__(self, "representation", _semantic_tag(
            representation or self.kind, "Space representation"))
        object.__setattr__(self, "centering", _semantic_tag(
            centering or layout, "Space centering"))
        object.__setattr__(self, "units", _component_units(normalized, units))
        object.__setattr__(self, "frame", _semantic_tag(frame, "Space frame"))
        object.__setattr__(self, "clock", _semantic_tag(clock, "Space clock"))

    def _key(self) -> Any:
        return (
            self.kind,
            self.name,
            self.components,
            self.layout,
            self.representation,
            self.centering,
            self.units,
            self.frame,
            self.clock,
        )

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, Space) and self._key() == other._key()

    def __hash__(self) -> int:
        return hash(self._key())

    def __repr__(self) -> str:
        return "%s(%r, components=%r, layout=%r, representation=%r, centering=%r)" % (
            type(self).__name__, self.name, list(self.components), self.layout,
            self.representation, self.centering)

    def to_data(self) -> dict[str, Any]:
        """Stable structural type identity used by Program IR serialization."""
        return {
            "kind": self.kind,
            "name": self.name,
            "components": list(self.components),
            "layout": self.layout,
            "representation": self.representation,
            "centering": self.centering,
            "units": list(self.units),
            "frame": self.frame,
            "clock": self.clock,
        }

    # Operator-first signature sugar: ``U >> Fields`` and ``(U, Fields) >> Rate(U)``.
    def __rshift__(self, output: Any) -> Any:
        """``space >> output`` -- a Signature with this space as the sole input."""
        from .signatures import Signature
        return Signature((self,), output)

    def __rrshift__(self, inputs: Any) -> Any:
        """``(a, b) >> space`` -- this space is the output, the left tuple the inputs."""
        from .signatures import Signature
        return Signature(_as_signature_inputs(inputs), self)


def _as_signature_inputs(inputs: Any) -> Any:
    """Normalize the left side of ``>>`` to a tuple of input types."""
    if isinstance(inputs, (tuple, list)):
        return tuple(inputs)
    return (inputs,)


class StateSpace(Space):
    """A conservative state space: the components of ``U`` plus optional physical
    roles, storage kind and conserved flags. Roles are metadata for diagnostics /
    CFL / projections; a generic Program must not depend on a specific role."""

    kind = "state"

    def __init__(self, name: Any = "U", components: Any = (), roles: Any = None, layout: str = "cell",
                 storage: str = "multifab", *, representation: Any = "conservative",
                 centering: Any = None, units: Any = None, frame: Any = "model",
                 clock: Any = "simulation") -> None:
        super().__init__(
            name,
            components,
            layout,
            representation=representation,
            centering=centering,
            units=units,
            frame=frame,
            clock=clock,
        )
        object.__setattr__(self, "roles", _freeze_metadata(roles or {}))
        if not isinstance(storage, str) or not storage:
            raise ValueError("StateSpace storage must be a non-empty string")
        object.__setattr__(self, "storage", storage)

    def _key(self) -> Any:
        return super()._key() + (self.storage, _metadata_key(self.roles))

    def __repr__(self) -> str:
        return ("StateSpace(%r, components=%r, roles=%r, layout=%r, storage=%r)"
                % (self.name, list(self.components), dict(self.roles), self.layout, self.storage))

    def to_data(self) -> dict[str, Any]:
        data = super().to_data()
        data.update({"roles": dict(self.roles), "storage": self.storage})
        return data


class FieldSpace(Space):
    """An auxiliary / solved-field space (elliptic field, gradient, divergence,
    magnetic field, derived quantities). Not necessarily produced by Poisson."""

    kind = "field"


class RateSpace(Space):
    """The tangent space of a :class:`StateSpace` -- values of ``dU/dt``.

    A rate always retains its complete immutable base StateSpace.  There is no
    name-only wildcard: two same-named states with different component/layout/storage
    structure have different tangent spaces.
    """

    kind = "rate"

    def __init__(self, base: Any) -> None:
        if not isinstance(base, StateSpace):
            raise TypeError("Rate expects a StateSpace, got %r" % (base,))
        base_name = base.name
        # A tangent inherits the complete physical layout of its base.  The
        # base remains the type authority; these mirrored fields make generic
        # Space consumers (not only Rate-aware ones) report the right shape.
        super().__init__(
            "Rate(%s)" % base_name,
            components=base.components,
            layout=base.layout,
            representation="rate",
            centering=base.centering,
            units=base.units,
            frame=base.frame,
            clock=base.clock,
        )
        object.__setattr__(self, "base_name", base_name)
        object.__setattr__(self, "base_space", base)

    def _key(self) -> Any:
        return super()._key() + (self.base_space._key(),)

    def __repr__(self) -> str:
        return "RateSpace(%r, base=%r)" % (self.name, self.base_space)

    def to_data(self) -> dict[str, Any]:
        data = super().to_data()
        data["base_name"] = self.base_name
        data["base_space"] = self.base_space.to_data()
        return data


def Rate(base: Any) -> Any:  # noqa: N802 (type-constructor sugar, intentionally capitalized)
    """Return the :class:`RateSpace` tangent of an immutable ``StateSpace``."""
    return RateSpace(base)


class AuxSpace(_ImmutableTypeValue):
    """A named auxiliary field provided or updated by the Simulation (e.g. an externally
    imposed magnetic field, a mask). Distinct from a FieldSpace, which an operator
    produces; an AuxSpace is imposed runtime data the operators may read."""

    def __init__(self, name: Any, kind: str = "cell_scalar", *, representation: Any = "auxiliary",
                 centering: Any = "cell", unit: Any = None, frame: Any = "model",
                 clock: Any = "simulation") -> None:
        if not isinstance(name, str) or not name:
            raise ValueError("AuxSpace name must be a non-empty string")
        if not isinstance(kind, str) or not kind:
            raise ValueError("AuxSpace kind must be a non-empty string")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "representation", _semantic_tag(
            representation, "AuxSpace representation"))
        object.__setattr__(self, "centering", _semantic_tag(centering, "AuxSpace centering"))
        if unit is not None and (not isinstance(unit, str) or not unit):
            raise ValueError("AuxSpace unit must be a non-empty string or None")
        object.__setattr__(self, "unit", unit)
        object.__setattr__(self, "frame", _semantic_tag(frame, "AuxSpace frame"))
        object.__setattr__(self, "clock", _semantic_tag(clock, "AuxSpace clock"))

    def __repr__(self) -> str:
        return "AuxSpace(%r, kind=%r)" % (self.name, self.kind)

    def to_data(self) -> dict[str, Any]:
        return {
            "kind": "aux",
            "name": self.name,
            "aux_kind": self.kind,
            "representation": self.representation,
            "centering": self.centering,
            "unit": self.unit,
            "frame": self.frame,
            "clock": self.clock,
        }
