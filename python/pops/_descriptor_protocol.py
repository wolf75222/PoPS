"""Generic typed-descriptor protocol (ADC-619 split).

The Spec 5 sec.6 typed-descriptor family that is NOT the native-brick catalog:
:class:`Availability` (the explainable yes/no/partial status), the inert
:class:`Descriptor` base, the structural :class:`DescriptorProtocol`, and the
:func:`reject_string_selector` guard. Split out of ``pops.descriptors`` for the
500-line cap; ``pops.descriptors`` re-exports every name here so the historical
``from pops.descriptors import Availability, Descriptor, ...`` paths keep working.
A descriptor is INERT: it declares metadata and computes nothing.
"""
from __future__ import annotations

import typing
from collections.abc import Mapping
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

from pops._report import ReportTree

if TYPE_CHECKING:
    from pops.descriptors_report import (
        CapabilitySet, LoweredDescriptor, RequirementSet)


def _freeze_descriptor_value(value: Any) -> Any:
    """Recursively freeze storage reachable from a descriptor attribute."""
    if isinstance(value, Mapping):
        return MappingProxyType({
            _freeze_descriptor_value(key): _freeze_descriptor_value(item)
            for key, item in value.items()
        })
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_descriptor_value(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(_freeze_descriptor_value(item) for item in value)
    freeze = getattr(value, "freeze", None)
    if callable(freeze):
        result = freeze()
        if result is not None and result is not value:
            raise TypeError(
                "%s.freeze() must seal and return self" % type(value).__name__)
    return value


class Availability:
    """An explainable availability status (Spec 5 sec.6: not just True/False).

    ``status`` is ``"yes"`` / ``"no"`` / ``"partial"``; truthiness is ``status == "yes"`` so
    it reads naturally in a boolean test while still carrying the reason + alternatives, so a
    rejection can be reported before the runtime is ever touched.
    """

    _STATUSES = ("yes", "no", "partial")

    def __init__(self, status: str, reason: str = "", *, missing: Any = None,
                 alternatives: Any = None) -> None:
        if status not in self._STATUSES:
            raise ValueError("Availability status must be one of %s (got %r)"
                             % (", ".join(self._STATUSES), status))
        self.status = status
        self.reason = str(reason)
        self.missing = list(missing or [])
        self.alternatives = list(alternatives or [])

    @classmethod
    def yes(cls, reason: str = "") -> Availability:
        """An available status (truthy), with an optional reason."""
        return cls("yes", reason)

    @classmethod
    def no(cls, reason: str, *, missing: Any = None, alternatives: Any = None) -> Availability:
        """An unavailable status (falsy) carrying the reason, what is missing and alternatives."""
        return cls("no", reason, missing=missing, alternatives=alternatives)

    @classmethod
    def partial(cls, reason: str, *, missing: Any = None,
                alternatives: Any = None) -> Availability:
        """A partially-available status (falsy): usable with limitations named in the reason."""
        return cls("partial", reason, missing=missing, alternatives=alternatives)

    @property
    def ok(self) -> bool:
        return self.status == "yes"

    def __bool__(self) -> bool:
        return self.status == "yes"

    def __repr__(self) -> str:
        return "Availability(%r, reason=%r)" % (self.status, self.reason)

    def __str__(self) -> str:
        lines = ["available: %s" % self.status]
        if self.reason:
            lines.append("  reason: %s" % self.reason)
        if self.missing:
            lines.append("  missing: %s" % ", ".join(map(str, self.missing)))
        if self.alternatives:
            lines.append("  alternatives: %s" % ", ".join(map(str, self.alternatives)))
        return "\n".join(lines)


class Descriptor:
    """Base of the inert typed descriptors (Spec 5 sec.6).

    Subclasses set :attr:`category` and override :meth:`options` (and :meth:`available` /
    :meth:`validate` where a route can be refused). The default contract reports an empty
    requirements/capabilities set and an unconditionally-available status. :meth:`inspect`
    returns a plain dict and :meth:`__str__` a short, deterministic summary (Spec 5 sec.12.1)
    -- never a dump of runtime data. A descriptor computes nothing.
    """

    category = "descriptor"
    #: The native C++ symbol this descriptor selects, or ``None`` when it names a
    #: pure-Python / planned route with no compiled symbol yet. Subclasses that wrap a
    #: native brick set this (as a class or instance attribute).
    native_id = None

    def __copy__(self) -> Descriptor:
        """Return a detached, mutable authoring copy even when ``self`` is frozen.

        Reference-resolution protocols intentionally copy descriptors before replacing Handle
        leaves.  Python's default shallow-copy implementation also copied ``_frozen=True`` and
        therefore made the first replacement fail.  A copy is a new authoring transaction: retain
        the already-immutable member values, but never inherit the source object's lifecycle flag.
        """
        clone = type(self).__new__(type(self))
        names = list(getattr(self, "__dict__", {}))
        for owner in type(self).__mro__:
            slots = owner.__dict__.get("__slots__", ())
            if isinstance(slots, str):
                slots = (slots,)
            for name in slots:
                if name.startswith("__") and not name.endswith("__"):
                    name = "_%s%s" % (owner.__name__.lstrip("_"), name)
                if name not in names:
                    names.append(name)
        for name in names:
            if name != "_frozen":
                if name not in ("__dict__", "__weakref__") and hasattr(self, name):
                    object.__setattr__(clone, name, getattr(self, name))
        return clone

    def freeze(self) -> Descriptor:
        """Freeze this descriptor: a later attribute mutation RAISES (ADC-563). Returns ``self``.

        A descriptor is mutable while it is authored / composed; once the assembly that holds it is
        frozen (``pops.compile`` freezes the ``Problem``, which cascades ``freeze`` to its member
        descriptors), the descriptor is sealed so a post-freeze edit to a route the artifact already
        committed cannot silently diverge from what was compiled. Idempotent."""
        # A guard on attribute assignment is not enough: stale references to a list/dict stored in
        # the descriptor would bypass __setattr__. Replace every container recursively first.
        for name, value in tuple(getattr(self, "__dict__", {}).items()):
            if name != "_frozen":
                object.__setattr__(self, name, _freeze_descriptor_value(value))
        object.__setattr__(self, "_frozen", True)
        return self

    def __setattr__(self, key: str, value: Any) -> None:
        """Refuse an attribute mutation after :meth:`freeze` (ADC-563), naming the frozen descriptor.

        Before freeze (the ``_frozen`` flag is unset / False) every assignment passes -- construction
        and fluent-builder edits are unaffected. After freeze, any assignment RAISES a
        ``RuntimeError`` naming the descriptor and the reason; there is no warning and no
        shallow-copy escape (a copy is a fresh, unfrozen object, which is the point)."""
        if getattr(self, "_frozen", False):
            raise RuntimeError(
                "%s [%s] is frozen (ADC-563): cannot set %r after the assembly was frozen by "
                "pops.compile. A compiled artifact is frozen to exactly the routes it was compiled "
                "from; author a fresh descriptor / Problem and recompile instead of mutating this one."
                % (getattr(self, "name", type(self).__name__), self.category, key))
        object.__setattr__(self, key, value)

    @property
    def name(self) -> str:
        return type(self).__name__

    def requirements(self) -> RequirementSet:
        """What the route NEEDS from context, as a :class:`~pops.descriptors_report.RequirementSet`.

        The default is empty. The typed set is the ONE interface (ADC-625): a consumer reads it
        through :meth:`~pops.descriptors_report.RequirementSet.check` or :meth:`to_dict`.
        """
        from pops.descriptors_report import RequirementSet
        return RequirementSet()

    def capabilities(self) -> CapabilitySet:
        """What the route PROVIDES, as a :class:`~pops.descriptors_report.CapabilitySet` (ADC-527).

        Read it through :meth:`~pops.descriptors_report.CapabilitySet.supports` (for a ``supports_``
        tag) or :meth:`to_dict`; it is the ONE interface (ADC-625).
        """
        from pops.descriptors_report import CapabilitySet
        return CapabilitySet()

    def options(self) -> dict:
        return {}

    def available(self, context: Any = None) -> Availability:
        """The default contract: unconditionally available (a refusing route overrides this)."""
        return Availability.yes()

    def validate(self, context: Any = None) -> bool:
        """Validate strictly, raising a structured diagnostic on error.

        The historical ``True`` return remains for boolean-only callers.  The inspectable form is
        :meth:`validate_report`; both paths share the same immutable :class:`ReportTree`.
        """
        self.validate_report(context).raise_if_error()
        return True

    def validate_report(self, context: Any = None) -> ReportTree:
        """Return the immutable descriptor validation tree without raising."""
        report = ReportTree(
            phase="validation", severity="info", code="validation.descriptor.report",
            source=self.category, owner=self,
            evidence={"descriptor": self.name, "category": self.category},
        )
        status = self.available(context)
        # ``partial`` is a real route whose remaining, context-dependent constraints are
        # proved by its resolver/provider once the layout is known.  It must stay inspectably
        # non-``ok`` without being confused with the terminal ``no`` state during authoring
        # validation.  Concrete provider resolution remains fail-closed.
        if status.status == "no":
            report = report.error(self.category, "unavailable", str(status),
                                  alternatives=status.alternatives)
        return report

    def lower(self, context: Any = None) -> LoweredDescriptor:
        """Return the inert :class:`~pops.descriptors_report.LoweredDescriptor` for this route.

        The lowering is metadata ONLY -- the name, the category, the native id and the chosen
        options the C++ runtime will materialise. It NEVER runs a numeric loop, opens an extension or
        touches a cell; a descriptor computes nothing. The typed record exposes its fields as
        attributes and via :meth:`to_dict` (ADC-625).
        """
        from pops.descriptors_report import LoweredDescriptor
        return LoweredDescriptor(name=self.name, category=self.category,
                                 native_id=self.native_id, options=self.options())

    def inspect(self) -> dict:
        """A plain-dict view of the descriptor: identity, options, requirements, capabilities."""
        return {"name": self.name, "category": self.category, "native_id": self.native_id,
                "options": self.options(), "requirements": self.requirements().to_dict(),
                "capabilities": self.capabilities().to_dict()}

    def capability_matrix(self, context: Any = None) -> Any:
        """One-row ADC-549 capability matrix for this typed descriptor (metadata only)."""
        from pops._capabilities import CapabilityRouteMatrix, CapabilityRouteRow
        status_obj = self.available(context)
        status = {"yes": "available", "no": "unavailable",
                  "partial": "partial"}.get(status_obj.status, "unknown")
        # capabilities() returns a typed CapabilitySet (ADC-625): read the mpi/gpu route support
        # through the typed .supports() accessor and the layout kind through the narrow .get().
        caps = self.capabilities()
        row = CapabilityRouteRow(
            "%s:%s" % (self.category, self.name),
            layout=caps.get("layout", "context"),
            backend="native" if self.native_id else "context",
            platform="context", mpi=caps.supports("mpi"), gpu=caps.supports("gpu"),
            status=status, limitation=status_obj.reason,
            error_message="" if status_obj.ok else str(status_obj), source="descriptor")
        return CapabilityRouteMatrix(self.name, row.layout, [row])

    def _summary(self) -> str:
        return ", ".join("%s=%r" % (k, v) for k, v in self.options().items())

    def __repr__(self) -> str:
        return "%s(%s)" % (self.name, self._summary())

    def __str__(self) -> str:
        body = self._summary()
        head = "%s [%s]" % (self.name, self.category)
        return "%s(%s)" % (head, body) if body else head


# --- the formal descriptor protocol (Spec 5 sec.6 / sec.7 / sec.13.12.1) ----------------
@typing.runtime_checkable
class DescriptorProtocol(typing.Protocol):
    """The semantic contract every route-choosing object honours (Spec 5 sec.6).

    Spec 5 stabilises "every object that chooses a route is a typed descriptor that declares
    its requirements / capabilities / options and answers ``available(context)`` with an
    EXPLAINABLE status". This :class:`typing.Protocol` documents that contract so it can be
    type-checked structurally; the concrete families (:class:`Descriptor` /
    :class:`BrickDescriptor` / the mesh descriptors) satisfy it by duck typing, they need not
    inherit from it. A descriptor is INERT -- :meth:`lower` returns metadata, it never runs a
    numeric loop or touches the runtime.

    Attributes:
        name: A short, stable identifier for the route (typically the class name).
        category: The descriptor family ("riemann", "layout", "output", ...).
        native_id: The native C++ symbol selected, or ``None`` for a pure-Python / planned
            route with no compiled symbol.

    Methods:
        requirements(): What the route NEEDS from the context (a ``RequirementSet``).
        capabilities(): What the route PROVIDES / supports (a ``CapabilitySet``).
        options(): The configured knobs and their chosen values (a plain dict).
        available(context): An :class:`Availability` (yes / no / partial), never a bare bool.
        validate(context): Strict validation (raises a structured ``DiagnosticError`` on failure).
        lower(context): The inert ``LoweredDescriptor`` record (metadata only, no computation).
        inspect(): A plain-dict view of the descriptor for tooling and printing.

    ADC-527 / ADC-625: the result objects (``RequirementSet`` / ``CapabilitySet`` /
    ``LoweredDescriptor`` / ``ReportTree``) are TYPED, not ``dict`` subclasses. Each family
    returns the typed object directly; a consumer reads it through the typed accessors
    (``supports`` / ``check`` / the ``LoweredDescriptor`` attributes) or ``to_dict``.
    """

    name: str
    category: str
    native_id: str | None

    def requirements(self) -> RequirementSet: ...

    def capabilities(self) -> CapabilitySet: ...

    def options(self) -> dict: ...

    def available(self, context: Any = None) -> Availability: ...

    def validate(self, context: Any = None) -> bool: ...

    def lower(self, context: Any = None) -> LoweredDescriptor: ...

    def inspect(self) -> dict: ...


def reject_string_selector(value: Any, param: str, suggestion: str) -> None:
    """Raise a clear :class:`TypeError` for a free-string algorithm selector (Spec 5 sec.7).

    Spec 5 forbids naming a brick / scheme / layout with a bare string; every route is a typed
    descriptor. A public API that still receives a string for @p param raises through this
    helper with a uniform, actionable message that points at the typed @p suggestion. It is a
    pure guard -- it always raises, so a caller wires it on the string branch of a parameter.

    Args:
        value: The rejected string the caller passed.
        param: The parameter / keyword name the string was passed for.
        suggestion: The typed alternative to use instead (e.g. ``pops.numerics.riemann.HLL()``).
    """
    raise TypeError("String algorithm selector rejected: %s=%r. Use %s."
                    % (param, value, suggestion))
