"""pops.descriptors_report -- the typed result objects of the DescriptorProtocol (ADC-527).

Spec 5 stabilises the ONE descriptor protocol: every route-choosing object declares its
requirements / capabilities / options and answers ``available`` / ``validate`` / ``lower`` with an
EXPLAINABLE status. ADC-527 gives those answers TYPED result objects instead of bare dicts:

    RequirementSet     what a route NEEDS from the context (ordered, typed, Mapping-compatible)
    CapabilitySet      what a route PROVIDES / supports (the ``supports_<tag>`` vocabulary)
    LoweredDescriptor  the inert lowering record (IR / native_id / manifest entry; no computation)
    ValidationReport   the accumulated, per-family structured errors (not a bare exception)

They are Mapping-compatible -- they subclass ``dict`` so every existing caller that treated
``requirements()`` / ``capabilities()`` / ``lower()`` as a plain dict keeps working unchanged
(``isinstance(x, dict)`` stays true, ``x[key]`` / ``x.get(...)`` / ``dict(x)`` all work) -- while
adding the typed helpers. A family may still return a bare dict during migration; the base
:class:`pops.descriptors.Descriptor` wraps it into the typed object, so a family is conform the moment
it inherits ``Descriptor`` (no big-bang). These objects run NO numeric loop and touch no runtime.
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


class RequirementSet(dict):
    """An ordered, typed set of what a route NEEDS from context (Mapping-compatible; ADC-527).

    Subclasses ``dict`` so it IS a mapping (``x[key]`` / ``x.get`` / ``dict(x)`` / ``isinstance(x,
    dict)`` all work) -- existing dict-consuming callers are unchanged -- while adding the typed
    :meth:`add` / :meth:`check` helpers. Constructed from a plain dict OR an iterable of
    :class:`Requirement`.
    """

    def __init__(self, requirements=()):
        super().__init__()
        if isinstance(requirements, dict):
            self.update(requirements)
        else:
            for req in requirements:
                self[req.key] = req.value

    @classmethod
    def from_dict(cls, mapping):
        return cls(dict(mapping or {}))

    def add(self, key, *, value=True, reason=""):
        """Add a requirement (chains)."""
        self[str(key)] = value
        return self

    def check(self, context):
        """Metadata-only membership check: report each requirement the context does not satisfy.

        NO numerics -- a requirement is satisfied when @p context carries a truthy value for its
        key (a dict) or an attribute of that name. Returns a :class:`ValidationReport`.
        """
        report = ValidationReport()
        ctx = context or {}
        for key, value in self.items():
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
        return dict(self)


class CapabilitySet(dict):
    """An ordered, typed set of what a route PROVIDES / supports (Mapping-compatible; ADC-527).

    Wraps the ``supports_<tag>`` vocabulary (``pops.solvers.requirements.CAPABILITY_TAGS``) plus any
    free provider capabilities. Subclasses ``dict`` for back-compat; :meth:`supports` reads a tag and
    is False (never raises) when the tag is absent. Subsumes both existing shapes -- the
    ``capability_map(...)`` dict and a BrickDescriptor's ``capabilities`` attribute dict -- via
    :meth:`from_dict`.
    """

    def __init__(self, capabilities=()):
        super().__init__()
        if isinstance(capabilities, dict):
            self.update(capabilities)
        else:
            for item in capabilities:
                key, value = item
                self[key] = value

    @classmethod
    def from_dict(cls, mapping):
        return cls(dict(mapping or {}))

    def supports(self, tag):
        """True when the route supports @p tag (reads ``supports_<tag>``; False if absent)."""
        key = tag if str(tag).startswith("supports_") else "supports_%s" % tag
        return bool(self.get(key, self.get(str(tag), False)))

    def to_dict(self):
        return dict(self)


class LoweredDescriptor(dict):
    """The inert lowering record: IR / native_id / manifest entry (Mapping-compatible; ADC-527).

    Subclasses ``dict`` so it is a superset of today's ``lower()`` dict (``name`` / ``category`` /
    ``native_id`` / ``options``) plus the optional ``ir`` / ``manifest_entry`` / ``extra`` payload.
    Constructing one runs NO numeric loop, opens no extension and touches no cell (ADC-527). If
    ``native_id is None`` for a route that requires a compiled symbol, ``lower`` / ``validate`` must
    fail loud upstream (no silent fallback).
    """

    def __init__(self, *, name, category, native_id, options=None, ir=None,
                 manifest_entry=None, extra=None):
        super().__init__()
        self.name = str(name)
        self.category = str(category)
        self.native_id = native_id
        self.options = dict(options or {})
        self.ir = ir
        self.manifest_entry = manifest_entry
        self.extra = dict(extra or {})
        self.update({"name": self.name, "category": self.category,
                     "native_id": self.native_id, "options": self.options})
        if ir is not None:
            self["ir"] = ir
        if manifest_entry is not None:
            self["manifest_entry"] = manifest_entry
        if self.extra:
            self["extra"] = self.extra

    @classmethod
    def from_dict(cls, mapping):
        data = dict(mapping or {})
        return cls(name=data.get("name", ""), category=data.get("category", "descriptor"),
                   native_id=data.get("native_id"), options=data.get("options"),
                   ir=data.get("ir"), manifest_entry=data.get("manifest_entry"),
                   extra=data.get("extra"))

    def to_dict(self):
        return dict(self)


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
