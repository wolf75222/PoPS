"""pops.output.runtime_policies -- the typed runtime-policy bundle (ADC-562).

A simulation's RUNTIME concerns -- when to write output, when to checkpoint, which diagnostics to
record, on what schedule -- do not belong in the scientific model script. :class:`RuntimePolicies`
groups them as ONE typed object so ``problem.runtime(policies)`` attaches them in a single call and
the physics ``block`` / ``field`` / ``time`` declarations stay clean.

TYPED members only: ``output=`` is an :class:`pops.output.OutputPolicy`, ``checkpoint=`` a
:class:`pops.output.CheckpointPolicy`, ``diagnostics=`` typed diagnostic-measure descriptors,
``schedules=`` typed :class:`pops.time.schedule.Schedule` objects. There is NO options bag, NO
``**kwargs`` and NO string keys -- a non-descriptor argument RAISES at construction, so a typo is a
loud error, not a silent runtime surprise. The bundle inspects and validates ON ITS OWN (without the
physics facade) and refuses an AMR / MPI / backend-incompatible member before the runtime is touched.
It is inert: it records typed declarations and computes nothing.
"""


from pops._report import Report


class RuntimePolicies:
    """A typed bundle of the runtime output / checkpoint / diagnostics / schedule policies (ADC-562).

    ``RuntimePolicies(output=OutputPolicy(...), checkpoint=CheckpointPolicy(...),
    diagnostics=[Norm(...), ...], schedules=[every(20), ...])``. Every member is a typed descriptor;
    a non-descriptor argument raises. Attach it with ``problem.runtime(policies)``; it is
    independently inspectable (:meth:`inspect`) and validatable (:meth:`validate`).
    """

    category = "runtime_policies"
    native_id = None

    def __init__(self, *, output=None, checkpoint=None, diagnostics=(), schedules=()):
        self.output = _require_category(output, "output", ("output_policy",))
        self.checkpoint = _require_category(checkpoint, "checkpoint", ("checkpoint_policy",))
        self.diagnostics = tuple(_require_diagnostic(d) for d in _as_tuple(diagnostics))
        self.schedules = tuple(_require_schedule(s) for s in _as_tuple(schedules))

    @property
    def name(self):
        return "RuntimePolicies"

    def members(self):
        """The typed members in a flat list (output / checkpoint / each diagnostic / each schedule)."""
        out = []
        if self.output is not None:
            out.append(self.output)
        if self.checkpoint is not None:
            out.append(self.checkpoint)
        out.extend(self.diagnostics)
        out.extend(self.schedules)
        return out

    def outputs(self):
        """The output / checkpoint policy list ``problem.output(...)`` records (order: output first)."""
        return [p for p in (self.output, self.checkpoint) if p is not None]

    def options(self):
        return {"has_output": self.output is not None,
                "has_checkpoint": self.checkpoint is not None,
                "n_diagnostics": len(self.diagnostics), "n_schedules": len(self.schedules)}

    def requirements(self):
        """The union of every member's requirements (e.g. an HDF5 parallel output -> parallel_io)."""
        from pops.descriptors_report import RequirementSet
        merged = {}
        for member in self.members():
            req = getattr(member, "requirements", None)
            if callable(req):
                merged.update(req().to_dict())
        return RequirementSet(merged)

    def validate(self, context=None):
        """Refuse an AMR / MPI / backend-incompatible member GIVEN @p context, before the runtime.

        Returns a :class:`~pops.problem.report.ProblemValidationReport`. A parallel-only output on a
        serial backend, or a member whose declared requirement the resolved layout / backend context
        does not satisfy, is refused HERE -- the bundle validates without the physics facade. A
        member's own ``validate`` failure is folded in too. Never a bare raise: the report
        accumulates, and a strict caller uses ``raise_if_error()``.
        """
        from pops.problem.report import ProblemValidationReport
        report = ProblemValidationReport(subject=self)
        ctx = context or {}
        # Each member's own validation (a solver-less checkpoint, an incomplete measure, ...).
        for member in self.members():
            member_validate = getattr(member, "validate", None)
            if callable(member_validate):
                try:
                    member_validate(ctx)
                except Exception as exc:  # noqa: BLE001 -- surface the member's message as an issue
                    report.error(self.category, "member_invalid", str(exc),
                                 context={"member": type(member).__name__})
        # The bundle's own compatibility check: a declared requirement the context does not satisfy.
        self._check_context_requirements(report, ctx)
        return report

    # The requirements that name a real backend INCOMPATIBILITY (a parallel-only policy on a serial
    # backend). A requirement satisfiable serially (e.g. a diagnostic mpi_reduction, which reduces
    # fine on one rank) is NOT gated -- gating it would be a false positive. Refuse ONLY when the
    # resolved context EXPLICITLY declares the incompatible backend state (no false positive: a
    # context that does not know its parallel state is never rejected), matching the Spec-6 discipline.
    _INCOMPATIBILITY_REQUIREMENTS = {"parallel_io": ("parallel", "mpi", "supports_mpi")}

    def _check_context_requirements(self, report, ctx):
        """Refuse a parallel-only member on an EXPLICITLY serial / non-MPI context (no false positive).

        Only a requirement in :attr:`_INCOMPATIBILITY_REQUIREMENTS` is gated, and only when the
        context explicitly declares the backend cannot serve it (a context key set to a falsey value).
        A context that carries no such key (unknown parallel state) is NEVER rejected."""
        req = self.requirements().to_dict()
        for key, context_keys in self._INCOMPATIBILITY_REQUIREMENTS.items():
            if not req.get(key):
                continue
            declared = [k for k in context_keys if isinstance(ctx, dict) and k in ctx]
            if declared and not any(bool(ctx.get(k)) for k in declared):
                report.error(
                    self.category, "incompatible_policy",
                    "a runtime policy requires %r but the resolved runtime context declares a serial "
                    "/ non-MPI backend (%s); choose a serial-compatible policy or compile for a "
                    "parallel backend" % (key, ", ".join("%s=%r" % (k, ctx.get(k)) for k in declared)),
                    context={"requirement": key})
        return report

    def inspect(self):
        """A typed :class:`RuntimePoliciesReport` of the bundle (no physics facade, no runtime)."""
        return RuntimePoliciesReport(
            output=_member_options(self.output),
            checkpoint=_member_options(self.checkpoint),
            diagnostics=[_member_options(d) for d in self.diagnostics],
            schedules=[_schedule_options(s) for s in self.schedules],
            requirements=self.requirements().to_dict())

    def to_dict(self):
        """The plain-dict bridge (the typed report's ``to_dict``)."""
        return self.inspect().to_dict()

    def __repr__(self):
        return ("RuntimePolicies(output=%s, checkpoint=%s, diagnostics=%d, schedules=%d)"
                % (self.output is not None, self.checkpoint is not None,
                   len(self.diagnostics), len(self.schedules)))

    def __str__(self):
        return str(self.inspect())


class RuntimePoliciesReport(Report):
    """The typed inspection report of a :class:`RuntimePolicies` bundle (ADC-562 / ADC-564).

    Attributes + :meth:`to_dict`, NEVER a dict subclass (adopts the shared :class:`pops.Report`
    base). Carries the output / checkpoint / diagnostic / schedule member options and the unioned
    requirements, so a caller sees the runtime concerns grouped, independent of the physics assembly.
    """

    report_type = "runtime_policies"
    schema_version = 1

    def __init__(self, *, output, checkpoint, diagnostics, schedules, requirements):
        self.output = output
        self.checkpoint = checkpoint
        self.diagnostics = list(diagnostics)
        self.schedules = list(schedules)
        self.requirements = dict(requirements)

    def to_dict(self):
        return self._stamp({"output": self.output, "checkpoint": self.checkpoint,
                            "diagnostics": [dict(d) for d in self.diagnostics],
                            "schedules": [dict(s) for s in self.schedules],
                            "requirements": dict(self.requirements)})

    def __str__(self):
        lines = ["runtime policies:"]
        lines.append("  output      : %s" % (self.output or "(none)"))
        lines.append("  checkpoint  : %s" % (self.checkpoint or "(none)"))
        lines.append("  diagnostics : %d" % len(self.diagnostics))
        lines.append("  schedules   : %d" % len(self.schedules))
        if self.requirements:
            lines.append("  requires    : %s" % ", ".join(sorted(self.requirements)))
        return "\n".join(lines)

    def __repr__(self):
        return ("RuntimePoliciesReport(output=%s, checkpoint=%s, diagnostics=%d)"
                % (self.output is not None, self.checkpoint is not None, len(self.diagnostics)))


def _as_tuple(value):
    """Normalise a single descriptor / an iterable of them to a tuple (a bare policy is wrapped)."""
    if value is None:
        return ()
    if isinstance(value, (list, tuple)):
        return tuple(value)
    return (value,)


def _require_category(value, role, categories):
    """Refuse a @p role member whose ``category`` is not one of @p categories (None passes)."""
    if value is None:
        return None
    cat = getattr(value, "category", None)
    if cat not in categories:
        raise TypeError(
            "RuntimePolicies(%s=...) expects a typed pops.output policy of category %s; got %r "
            "(category %r). Pass a typed descriptor, not an options bag or a string."
            % (role, " / ".join(categories), type(value).__name__, cat))
    return value


def _require_diagnostic(value):
    """Refuse a diagnostics member that is not a typed diagnostic-measure / conservation descriptor."""
    cat = getattr(value, "category", None)
    if not (isinstance(cat, str) and (cat.startswith("diagnostic") or cat == "conservation_check")):
        raise TypeError(
            "RuntimePolicies(diagnostics=...) expects typed pops.diagnostics measures (Norm / "
            "Integral / MinMax / ConservationCheck); got %r (category %r). No options bag, no string."
            % (type(value).__name__, cat))
    return value


def _require_schedule(value):
    """Refuse a schedules member that is not a typed :class:`pops.time.schedule.Schedule`."""
    from pops.time.schedule import Schedule
    if not isinstance(value, Schedule):
        raise TypeError(
            "RuntimePolicies(schedules=...) expects typed pops.time.schedule.Schedule objects "
            "(every(N) / when(cond) / ...); got %r. No options bag, no string." % (type(value).__name__,))
    return value


def _member_options(member):
    """The ``{name, category, options}`` view of a policy member, or ``None`` (JSON-ready options)."""
    if member is None:
        return None
    opts = getattr(member, "options", None)
    raw = opts() if callable(opts) else {}
    return {"name": getattr(member, "name", type(member).__name__),
            "category": getattr(member, "category", None),
            "options": {k: _jsonable(v) for k, v in raw.items()}}


def _jsonable(value):
    """Coerce an option value to a JSON-ready form (a non-scalar, e.g. a Schedule, becomes a token)."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    return getattr(value, "name", None) or repr(value)


def _schedule_options(schedule):
    """The ``{kind, policy}`` view of a Schedule (it is not a Descriptor, so read its fields)."""
    return {"kind": getattr(schedule, "kind", None), "policy": getattr(schedule, "policy", None)}


__all__ = ["RuntimePolicies", "RuntimePoliciesReport"]
