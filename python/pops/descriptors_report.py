"""pops.descriptors_report -- the typed result objects of the DescriptorProtocol (ADC-527).

Spec 5 stabilises the ONE descriptor protocol: every route-choosing object declares its
requirements / capabilities / options and answers ``available`` / ``validate`` / ``lower`` with an
EXPLAINABLE status. ADC-527 gives those answers TYPED result objects instead of bare dicts:

    RequirementSet     what a route NEEDS from the context (ordered, typed)
    CapabilitySet      what a route PROVIDES / supports (the ``supports_<tag>`` vocabulary)
    LoweredDescriptor  the inert lowering record (IR / native_id / manifest entry; no computation)
    ValidationReport   the accumulated, per-family structured errors (not a bare exception)

ADC-625 makes them the ONE final form: they are TYPED objects, NOT ``dict`` subclasses. The only
mapping bridge is :meth:`to_dict` -- a caller that needs a plain dict asks for one explicitly. The
typed helpers (:meth:`RequirementSet.check`, :meth:`CapabilitySet.supports`, and the
:class:`LoweredDescriptor` attributes) are the vocabulary every producer and consumer speaks. These
objects run NO numeric loop and touch no runtime.
"""


class Requirement:
    """One typed requirement: a key, its required value, why, and what may satisfy it."""

    def __init__(self, key, *, value=True, reason="", satisfied_by=None):
        self.key = str(key)
        self.value = value
        self.reason = str(reason)
        self.satisfied_by = satisfied_by

    def to_dict(self):
        return {"key": self.key, "value": self.value, "reason": self.reason,
                "satisfied_by": self.satisfied_by}

    def __repr__(self):
        return "Requirement(%r, value=%r)" % (self.key, self.value)


class RequirementSet:
    """An ordered, typed set of what a route NEEDS from context (ADC-527 / ADC-625).

    A TYPED object (not a ``dict`` subclass): :meth:`to_dict` is the only mapping bridge, and the
    typed :meth:`add` / :meth:`check` helpers are the interface. Constructed from a plain dict OR an
    iterable of :class:`Requirement`.
    """

    def __init__(self, requirements=()):
        self._data = {}
        if isinstance(requirements, dict):
            self._data.update(requirements)
        else:
            for req in requirements:
                self._data[req.key] = req.value

    @classmethod
    def from_dict(cls, mapping):
        return cls(dict(mapping or {}))

    def add(self, key, *, value=True, reason=""):
        """Add a requirement (chains)."""
        self._data[str(key)] = value
        return self

    def check(self, context):
        """Metadata-only membership check: report each requirement the context does not satisfy.

        NO numerics -- a requirement is satisfied when @p context carries a truthy value for its
        key (a dict) or an attribute of that name. Returns a :class:`ValidationReport`.
        """
        report = ValidationReport()
        ctx = context or {}
        for key, value in self._data.items():
            if isinstance(ctx, dict):
                present = bool(ctx.get(key))
            else:
                present = bool(getattr(ctx, key, None))
            if value and not present:
                report.error("requirement", "unsatisfied",
                             "route requires %r, not present in the context" % key,
                             context={"requirement": key})
        return report

    def to_dict(self):
        return dict(self._data)

    def __eq__(self, other):
        return isinstance(other, RequirementSet) and self._data == other._data

    __hash__ = None

    def __bool__(self):
        return bool(self._data)

    def __repr__(self):
        return "RequirementSet(%r)" % (self._data,)


class CapabilitySet:
    """An ordered, typed set of what a route PROVIDES / supports (ADC-527 / ADC-625).

    Wraps the ``supports_<tag>`` vocabulary (``pops.solvers.requirements.CAPABILITY_TAGS``) plus any
    free provider capabilities. A TYPED object (not a ``dict`` subclass): :meth:`supports` reads a tag
    and is False (never raises) when the tag is absent, and :meth:`to_dict` is the mapping bridge.
    Subsumes both existing shapes -- the ``capability_map(...)`` dict and a BrickDescriptor's
    ``capabilities`` attribute dict -- via :meth:`from_dict`.
    """

    def __init__(self, capabilities=()):
        self._data = {}
        if isinstance(capabilities, dict):
            self._data.update(capabilities)
        else:
            for item in capabilities:
                key, value = item
                self._data[key] = value

    @classmethod
    def from_dict(cls, mapping):
        return cls(dict(mapping or {}))

    def supports(self, tag):
        """True when the route supports @p tag (reads ``supports_<tag>``; False if absent)."""
        key = tag if str(tag).startswith("supports_") else "supports_%s" % tag
        return bool(self._data.get(key, self._data.get(str(tag), False)))

    def get(self, key, default=None):
        """Read one capability value (metadata only; ``default`` when absent).

        A narrow, explicit accessor for the handful of non-``supports_`` capabilities a route reads
        by name (``layout`` kind, backend ``tier``); it is NOT the dict emulation -- it never exposes
        ``__getitem__`` / iteration. Prefer :meth:`supports` for ``supports_<tag>`` flags.
        """
        return self._data.get(key, default)

    def to_dict(self):
        return dict(self._data)

    def __eq__(self, other):
        return isinstance(other, CapabilitySet) and self._data == other._data

    __hash__ = None

    def __bool__(self):
        return bool(self._data)

    def __repr__(self):
        return "CapabilitySet(%r)" % (self._data,)


class LoweredDescriptor:
    """The inert lowering record: IR / native_id / manifest entry (ADC-527 / ADC-625).

    A TYPED object (not a ``dict`` subclass): the name / category / native_id / options / scheme /
    ir / manifest_entry / extra payload are ATTRIBUTES, and :meth:`to_dict` is the mapping bridge.
    Constructing one runs NO numeric loop, opens no extension and touches no cell (ADC-527). If
    ``native_id is None`` for a route that requires a compiled symbol, ``lower`` / ``validate`` must
    fail loud upstream (no silent fallback).
    """

    def __init__(self, *, name, category, native_id, options=None, ir=None,
                 manifest_entry=None, extra=None, scheme=None):
        self.name = str(name)
        self.category = str(category)
        self.native_id = native_id
        self.options = dict(options or {})
        self.ir = ir
        self.manifest_entry = manifest_entry
        self.extra = dict(extra or {})
        self.scheme = scheme

    @classmethod
    def from_dict(cls, mapping):
        data = dict(mapping or {})
        return cls(name=data.get("name", ""), category=data.get("category", "descriptor"),
                   native_id=data.get("native_id"), options=data.get("options"),
                   ir=data.get("ir"), manifest_entry=data.get("manifest_entry"),
                   extra=data.get("extra"), scheme=data.get("scheme"))

    def to_dict(self):
        out = {"name": self.name, "category": self.category, "native_id": self.native_id,
               "options": dict(self.options)}
        if self.scheme is not None:
            out["scheme"] = self.scheme
        if self.ir is not None:
            out["ir"] = self.ir
        if self.manifest_entry is not None:
            out["manifest_entry"] = self.manifest_entry
        # ``extra`` carries the route-specific top-level payload a specialised lowering adds
        # (a LinearProblem's method / preconditioner / tol / max_iter / restart, ...); flatten it
        # into the record so ``to_dict()[<field>]`` reads it directly.
        out.update(self.extra)
        return out

    def __eq__(self, other):
        return isinstance(other, LoweredDescriptor) and self.to_dict() == other.to_dict()

    __hash__ = None

    def __repr__(self):
        return "LoweredDescriptor(name=%r, category=%r, native_id=%r)" % (
            self.name, self.category, self.native_id)


class ValidationIssue:
    """One structured validation error: its family, a stable code, a message and user context."""

    def __init__(self, *, family, code, message, context=None, severity="error",
                 alternatives=()):
        self.family = str(family)
        self.code = str(code)
        self.message = str(message)
        self.context = dict(context or {})
        self.severity = str(severity)
        self.alternatives = list(alternatives or [])

    def to_dict(self):
        return {"family": self.family, "code": self.code, "message": self.message,
                "context": dict(self.context), "severity": self.severity,
                "alternatives": list(self.alternatives)}

    def __str__(self):
        head = "[%s/%s] %s" % (self.family, self.code, self.message)
        if self.alternatives:
            head += " (alternatives: %s)" % ", ".join(self.alternatives)
        return head

    def __repr__(self):
        return "ValidationIssue(family=%r, code=%r)" % (self.family, self.code)


class ValidationReport:
    """Accumulated, per-family structured validation errors with user context (ADC-527).

    Not a bare exception: chaining accumulators (:meth:`add` / :meth:`error` / :meth:`extend`)
    collect issues so one pass reports EVERY problem. :meth:`by_family` groups them, :attr:`ok` /
    ``__bool__`` give the verdict, and :meth:`raise_if_error` keeps the fail-loud behaviour strict
    callers rely on. The subject-bound :class:`pops.problem.report.ProblemValidationReport` is the
    Problem-facing view over this same shape.
    """

    def __init__(self, subject=None):
        self.subject = subject
        self._issues = []

    def add(self, issue):
        self._issues.append(issue)
        return self

    def error(self, family, code, message, *, context=None, alternatives=()):
        return self.add(ValidationIssue(family=family, code=code, message=message,
                                        context=context, severity="error",
                                        alternatives=alternatives))

    def extend(self, other):
        if other is not None:
            self._issues.extend(other.issues)
        return self

    @property
    def issues(self):
        return list(self._issues)

    def by_family(self):
        grouped = {}
        for issue in self._issues:
            grouped.setdefault(issue.family, []).append(issue)
        return grouped

    @property
    def ok(self):
        return not any(issue.severity == "error" for issue in self._issues)

    def __bool__(self):
        return self.ok

    def __iter__(self):
        return iter(self._issues)

    def __len__(self):
        return len(self._issues)

    def raise_if_error(self):
        if not self.ok:
            raise ValueError(str(self))

    def to_dict(self):
        return {"subject": getattr(self.subject, "name", None),
                "ok": self.ok, "issues": [issue.to_dict() for issue in self._issues]}

    def __str__(self):
        if self.ok:
            return "validation ok"
        lines = ["validation failed:"]
        for family, issues in self.by_family().items():
            lines.append("  %s:" % family)
            for issue in issues:
                lines.append("    - %s" % issue)
        return "\n".join(lines)

    def __repr__(self):
        return "ValidationReport(%d issue(s), ok=%s)" % (len(self._issues), self.ok)


__all__ = ["Requirement", "RequirementSet", "CapabilitySet", "LoweredDescriptor",
           "ValidationIssue", "ValidationReport"]
