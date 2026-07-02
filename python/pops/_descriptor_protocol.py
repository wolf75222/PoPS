"""Generic typed-descriptor protocol (ADC-619 split).

The Spec 5 sec.6 typed-descriptor family that is NOT the native-brick catalog:
:class:`Availability` (the explainable yes/no/partial status), the inert
:class:`Descriptor` base, the structural :class:`DescriptorProtocol`, and the
:func:`reject_string_selector` guard. Split out of ``pops.descriptors`` for the
500-line cap; ``pops.descriptors`` re-exports every name here so the historical
``from pops.descriptors import Availability, Descriptor, ...`` paths keep working.
A descriptor is INERT: it declares metadata and computes nothing.
"""

import typing


class Availability:
    """An explainable availability status (Spec 5 sec.6: not just True/False).

    ``status`` is ``"yes"`` / ``"no"`` / ``"partial"``; truthiness is ``status == "yes"`` so
    it reads naturally in a boolean test while still carrying the reason + alternatives, so a
    rejection can be reported before the runtime is ever touched.
    """

    _STATUSES = ("yes", "no", "partial")

    def __init__(self, status, reason="", *, missing=None, alternatives=None):
        if status not in self._STATUSES:
            raise ValueError("Availability status must be one of %s (got %r)"
                             % (", ".join(self._STATUSES), status))
        self.status = status
        self.reason = str(reason)
        self.missing = list(missing or [])
        self.alternatives = list(alternatives or [])

    @classmethod
    def yes(cls, reason=""):
        return cls("yes", reason)

    @classmethod
    def no(cls, reason, *, missing=None, alternatives=None):
        return cls("no", reason, missing=missing, alternatives=alternatives)

    @classmethod
    def partial(cls, reason, *, missing=None, alternatives=None):
        return cls("partial", reason, missing=missing, alternatives=alternatives)

    @property
    def ok(self):
        return self.status == "yes"

    def __bool__(self):
        return self.status == "yes"

    def __repr__(self):
        return "Availability(%r, reason=%r)" % (self.status, self.reason)

    def __str__(self):
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

    @property
    def name(self):
        return type(self).__name__

    def requirements(self):
        return {}

    def capabilities(self):
        return {}

    def options(self):
        return {}

    def available(self, context=None):
        return Availability.yes()

    def validate(self, context=None):
        status = self.available(context)
        if not status.ok:
            raise ValueError("%s is not available for this route:\n%s" % (self.name, status))
        return True

    def lower(self, context=None):
        """Return the inert lowering record for this route (Spec 5 sec.6 / sec.7).

        The lowering is metadata ONLY -- the name, the category, the native id and the
        chosen options the C++ runtime will materialise. It NEVER runs a numeric loop, opens
        an extension or touches a cell; a descriptor computes nothing. Subclasses that carry a
        richer payload may extend the dict, but the contract stays inert.
        """
        return {"name": self.name, "category": self.category,
                "native_id": self.native_id, "options": self.options()}

    def inspect(self):
        return {"name": self.name, "category": self.category, "native_id": self.native_id,
                "options": self.options(), "requirements": self.requirements(),
                "capabilities": self.capabilities()}

    def capability_matrix(self, context=None):
        """One-row ADC-549 capability matrix for this typed descriptor (metadata only)."""
        from pops._capabilities import CapabilityRouteMatrix, CapabilityRouteRow
        status_obj = self.available(context)
        status = {"yes": "available", "no": "unavailable",
                  "partial": "partial"}.get(status_obj.status, "unknown")
        caps = self.capabilities()
        row = CapabilityRouteRow(
            "%s:%s" % (self.category, self.name),
            layout=caps.get("layout", "context") if isinstance(caps, dict) else "context",
            backend="native" if self.native_id else "context",
            platform="context",
            mpi=caps.get("supports_mpi") if isinstance(caps, dict) else None,
            gpu=caps.get("supports_gpu") if isinstance(caps, dict) else None,
            status=status, limitation=status_obj.reason,
            error_message="" if status_obj.ok else str(status_obj), source="descriptor")
        return CapabilityRouteMatrix(self.name, row.layout, [row])

    def _summary(self):
        return ", ".join("%s=%r" % (k, v) for k, v in self.options().items())

    def __repr__(self):
        return "%s(%s)" % (self.name, self._summary())

    def __str__(self):
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
        requirements(): What the route NEEDS from the context (a plain dict).
        capabilities(): What the route PROVIDES / supports (a plain dict).
        options(): The configured knobs and their chosen values (a plain dict).
        available(context): An :class:`Availability` (yes / no / partial), never a bare bool.
        validate(context): Raise a clear error when the route cannot be used in @p context.
        lower(context): The inert lowering record (metadata only, no computation).
        inspect(): A plain-dict view of the descriptor for tooling and printing.
    """

    name: str
    category: str
    native_id: str | None

    def requirements(self) -> dict: ...

    def capabilities(self) -> dict: ...

    def options(self) -> dict: ...

    def available(self, context=None) -> "Availability": ...

    def validate(self, context=None): ...

    def lower(self, context=None) -> dict: ...

    def inspect(self) -> dict: ...


def reject_string_selector(value, param, suggestion):
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
