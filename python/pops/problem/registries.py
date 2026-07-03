"""pops.problem.registries -- the typed internal registries of a Problem (ADC-553).

The old ``pops.case.Case`` kept its assembly in six FLAT dicts on the instance
(``_blocks _fields _params _aux _outputs _time``) with all subsystem logic inlined in one
``validate`` / ``available`` / ``inspect``. ADC-553 splits that monolith into TYPED registries,
one per family, each independently inspectable and validatable:

    BlockRegistry          physics blocks (name -> model + spatial + time + diagnostics)
    FieldRegistry          elliptic field problems (name -> FieldProblem)
    TimeRegistry           the whole-system time Program (a single slot)
    ParamRegistry          runtime / const parameter declarations
    RuntimePolicyRegistry  aux inputs + output / checkpoint policies (runtime-facing)
    ConstraintRegistry     structural constraints + AMR refinement criteria (layout-free)

Each registry: ``add`` / ``get`` / ``names`` / ``__iter__`` / ``inspect()`` /
``validate(context) -> ProblemValidationReport``. The :class:`~pops.problem.problem.Problem`
facade owns one of each and DELEGATES to them; it holds no flat dict and no inline subsystem
logic. A registry owns NO runtime data, imports no ``_pops`` / runtime / codegen, and computes
nothing -- it records typed declarations and reports structured errors.
"""
from pops.problem.handles import BlockHandle, FieldHandle
from pops.problem.report import ProblemValidationReport

# Sentinel distinguishing "no kind= passed" from "kind=None": ParamRegistry rejects any kind=
# keyword (Spec 5 sec.7) with a clear error naming the typed alternative.
_NO_KIND = object()


class BlockRegistry:
    """The physics blocks declared on a Problem (name -> model + spatial + time + diagnostics).

    A block records its physics ``model`` (required), its ``spatial`` discretisation brick, and the
    optional per-block ``time`` scheme and ``diagnostics`` (ADC-526's ``add_block`` superset). A
    duplicate name is refused loudly at declaration -- the earliest, per-family error.
    """

    family = "block"

    def __init__(self):
        self._blocks = {}

    def add(self, name, model, *, spatial=None, time=None, diagnostics=None):
        """Record a block ``name`` with its ``model`` (required). Returns a stable :class:`BlockHandle`."""
        key = str(name)
        if model is None:
            raise ValueError("add_block(%r): a physics model is required" % key)
        if key in self._blocks:
            raise ValueError("add_block(%r): a block of that name already exists" % key)
        self._blocks[key] = {"model": model, "spatial": spatial, "time": time,
                             "diagnostics": diagnostics}
        return BlockHandle(key)

    def get(self, name):
        return self._blocks.get(str(name))

    def names(self):
        return list(self._blocks)

    def spec(self, name):
        """The full ``{model, spatial, time, diagnostics}`` record for @p name (or ``None``)."""
        return self._blocks.get(str(name))

    def items(self):
        return self._blocks.items()

    def __iter__(self):
        return iter(self._blocks)

    def __len__(self):
        return len(self._blocks)

    def __contains__(self, name):
        return str(name) in self._blocks

    def validate(self, context=None):
        """Report a structured error when there is no block, or a block has no model."""
        report = ProblemValidationReport()
        if not self._blocks:
            report.error(self.family, "no_block",
                         "no block declared; add one with add_block(name, model, spatial)",
                         alternatives=["add_block(name, model, spatial)"])
            return report
        for name, spec in self._blocks.items():
            if spec.get("model") is None:
                report.error(self.family, "no_model", "block %r has no physics model" % name,
                             context={"block": name})
        return report

    def inspect(self):
        return {name: {"model": getattr(spec["model"], "name", repr(spec["model"])),
                       "spatial": getattr(spec["spatial"], "name", spec["spatial"]),
                       "time": getattr(spec["time"], "name", None),
                       "diagnostics": getattr(spec["diagnostics"], "name", None)}
                for name, spec in self._blocks.items()}


class FieldRegistry:
    """The elliptic field problems declared on a Problem (keyed on the field's name)."""

    family = "field"

    def __init__(self):
        self._fields = {}

    def add(self, field_problem):
        """Register a :class:`pops.fields.FieldProblem` (keyed on its name). Returns a :class:`FieldHandle`."""
        from pops.fields import FieldProblem  # lazy: keep pops.problem free of a fields module edge
        if not isinstance(field_problem, FieldProblem):
            raise TypeError("field: expected a pops.fields.FieldProblem; got %r"
                            % type(field_problem).__name__)
        key = field_problem.name
        if key in self._fields:
            raise ValueError("field: a field named %r already exists" % key)
        self._fields[key] = field_problem
        return FieldHandle(key)

    def get(self, name):
        return self._fields.get(str(name))

    def names(self):
        return list(self._fields)

    def items(self):
        return self._fields.items()

    def solvers(self):
        """The ``{field_name: solver}`` mapping (skips a field with no solver)."""
        return {name: fp.solver for name, fp in self._fields.items() if fp.solver is not None}

    def __iter__(self):
        return iter(self._fields)

    def __len__(self):
        return len(self._fields)

    def __contains__(self, name):
        return str(name) in self._fields

    def validate(self, context=None):
        """Report each field problem's own validation failure (structured, never a bare raise)."""
        report = ProblemValidationReport()
        for name, field in self._fields.items():
            try:
                field.validate(context)
            except Exception as exc:  # noqa: BLE001 -- surface the field's own message as an issue
                report.error(self.family, "field_invalid", str(exc), context={"field": name})
        return report

    def inspect(self):
        return {name: fp.inspect() for name, fp in self._fields.items()}


class TimeRegistry:
    """The whole-system time scheme slot (a single ``pops.time.Program``, attached at compile)."""

    family = "time"

    def __init__(self):
        self._program = None

    def set(self, program):
        """Record the time scheme (the whole-system Program). Overwrites a prior one."""
        self._program = program

    @property
    def program(self):
        return self._program

    def names(self):
        return [getattr(self._program, "name", "program")] if self._program is not None else []

    def __iter__(self):
        return iter([self._program] if self._program is not None else [])

    def validate(self, context=None):
        """The time scheme is optional at assembly (supplied at compile); nothing to reject here."""
        return ProblemValidationReport()

    def inspect(self):
        return {"program": getattr(self._program, "name", None)
                if self._program is not None else None}


class ParamRegistry:
    """The runtime / const parameter declarations (name -> {default, kind})."""

    family = "params"

    def __init__(self):
        self._params = {}
        # The TYPED declaration object per name (a pops.params RuntimeParam / ConstParam carrying its
        # domain), retained so the bind-time domain check (ADC-541) can call decl.check_bind(value).
        # A bare (name, default) declaration has no typed object -> None.
        self._declarations = {}

    def add(self, name, default=None, *, kind=_NO_KIND):
        """Declare a parameter. A bare ``kind=`` string is rejected (Spec 5 sec.7)."""
        if kind is not _NO_KIND:
            raise TypeError(
                "param: the kind= string is removed (Spec 5 sec.7); pass a typed param object "
                "(pops.physics.RuntimeParam(name, value) or pops.physics.ConstParam(name, value)) "
                "instead of kind=%r" % (kind,))
        if hasattr(name, "kind") and hasattr(name, "name") and hasattr(name, "value"):
            # A pops.physics RuntimeParam/ConstParam (Param): kind + value carried directly.
            if default is not None:
                raise TypeError(
                    "param: a typed param was given; do not also pass a default (%r)" % (default,))
            self._params[str(name.name)] = {"default": name.value, "kind": str(name.kind)}
            self._declarations[str(name.name)] = name
        elif getattr(name, "category", None) in ("runtime_param", "const_param") \
                and hasattr(name, "name"):
            # A pops.params typed param carrying a DOMAIN (RuntimeParam(domain=...) / ConstParam):
            # retain it as the declaration so the bind-time domain check (ADC-541) can call
            # check_bind(value); its kind is derived from the category.
            if default is not None:
                raise TypeError(
                    "param: a typed param was given; do not also pass a default (%r)" % (default,))
            kind_of = {"runtime_param": "runtime", "const_param": "const"}[name.category]
            declared = getattr(name, "default", getattr(name, "value", None))
            self._params[str(name.name)] = {"default": declared, "kind": kind_of}
            self._declarations[str(name.name)] = name
        else:
            self._params[str(name)] = {"default": default, "kind": "const"}
            self._declarations[str(name)] = None

    def get(self, name):
        return self._params.get(str(name))

    def names(self):
        return list(self._params)

    def items(self):
        return self._params.items()

    def declarations(self):
        """The ``{name: typed declaration}`` map (a ``RuntimeParam``/``ConstParam`` or ``None``).

        The bind-time domain check (ADC-541) reads it to call ``decl.check_bind(value)`` on each
        supplied runtime param. A name declared without a typed object maps to ``None``.
        """
        return dict(self._declarations)

    def __iter__(self):
        return iter(self._params)

    def __len__(self):
        return len(self._params)

    def validate(self, context=None):
        return ProblemValidationReport()

    def inspect(self):
        return dict(self._params)


class RuntimePolicyRegistry:
    """Runtime-facing declarations: static aux inputs and output / checkpoint policies.

    These describe what the runtime does with the assembly (background aux fields, when to write /
    checkpoint); they carry no runtime data themselves. Output entries are validated to be real
    policy descriptors (a non-policy object is a typo caught here, not at run time).
    """

    family = "runtime"
    _POLICY_CATEGORIES = ("output_policy", "checkpoint_policy")

    def __init__(self):
        self._aux = {}
        self._outputs = []
        # The typed RuntimePolicies bundle (ADC-562), retained so its self-contained validate runs
        # with the compile context; its output / checkpoint members are ALSO unpacked into _outputs.
        self._policies = None

    def add_aux(self, name, value=None):
        """Declare a static aux input ``name`` (e.g. a background field)."""
        self._aux[str(name)] = value

    def add_output(self, policy):
        """Attach an output / checkpoint policy descriptor."""
        self._outputs.append(policy)

    def set_policies(self, policies):
        """Record a typed :class:`pops.output.RuntimePolicies` bundle (ADC-562).

        Unpacks the bundle's output / checkpoint members into ``_outputs`` (so ``run(output_dir=...)``
        fires them exactly like :meth:`add_output`) and retains the bundle for its self-contained
        :meth:`validate`. A non-bundle argument is refused loudly (no options bag)."""
        from pops.output.runtime_policies import RuntimePolicies
        if not isinstance(policies, RuntimePolicies):
            raise TypeError(
                "problem.runtime(...) expects a typed pops.RuntimePolicies bundle; got %r. Group the "
                "runtime concerns with pops.RuntimePolicies(output=..., checkpoint=..., "
                "diagnostics=..., schedules=...)." % (type(policies).__name__,))
        self._policies = policies
        for policy in policies.outputs():
            self._outputs.append(policy)

    @property
    def policies(self):
        return self._policies

    @property
    def aux(self):
        return dict(self._aux)

    @property
    def outputs(self):
        return list(self._outputs)

    def names(self):
        return sorted(self._aux) + [getattr(p, "name", repr(p)) for p in self._outputs]

    def __iter__(self):
        return iter(self._outputs)

    def validate(self, context=None):
        """Refuse a bad output entry and run the RuntimePolicies bundle's self-contained validate.

        Each ``_outputs`` entry must be a real output / checkpoint policy descriptor. When a typed
        :class:`pops.output.RuntimePolicies` bundle was attached (``problem.runtime(...)``), its OWN
        ``validate(context)`` runs too, so an AMR / MPI / backend-incompatible policy is refused
        before the runtime -- with the resolved layout / backend @p context (ADC-562)."""
        report = ProblemValidationReport()
        for policy in self._outputs:
            cat = getattr(policy, "category", None)
            if cat not in self._POLICY_CATEGORIES:
                report.error(
                    self.family, "bad_output_policy",
                    "output() expects a pops.output.OutputPolicy / CheckpointPolicy; got %r "
                    "(category %r)" % (type(policy).__name__, cat),
                    context={"policy": type(policy).__name__, "category": cat})
        if self._policies is not None:
            report.extend(self._policies.validate(context))
        return report

    def inspect(self):
        info = {"aux": sorted(self._aux),
                "outputs": [getattr(p, "name", repr(p)) for p in self._outputs]}
        if self._policies is not None:
            info["policies"] = self._policies.inspect().to_dict()
        return info


class ConstraintRegistry:
    """Structural constraints + layout-free AMR refinement criteria (ADC-526).

    A Problem carries no layout, so the AMR refinement criteria (refine / regrid / nesting / patches)
    are recorded HERE as inert descriptors and applied to the ``Uniform`` / ``AMR`` layout at
    ``pops.compile(problem, layout=...)``. It also holds cross-family structural checks (a block and
    a field must not share a name). It owns no layout and no runtime.
    """

    family = "amr"

    def __init__(self):
        self._criteria = {}  # refine / regrid / nesting / patches -> descriptor

    def set_refinement(self, *, refine=None, regrid=None, nesting=None, patches=None):
        """Record the AMR refinement criteria (layout-free; applied at compile)."""
        if refine is not None:
            self._criteria["refine"] = refine
        if regrid is not None:
            self._criteria["regrid"] = regrid
        if nesting is not None:
            self._criteria["nesting"] = nesting
        if patches is not None:
            self._criteria["patches"] = patches

    @property
    def refinement(self):
        return dict(self._criteria)

    def names(self):
        return list(self._criteria)

    def __iter__(self):
        return iter(self._criteria.items())

    def __len__(self):
        return len(self._criteria)

    def validate(self, context=None):
        """No layout at assembly, so refinement criteria cannot be checked here (deferred to compile)."""
        return ProblemValidationReport()

    def inspect(self):
        return {kind: getattr(desc, "name", repr(desc))
                for kind, desc in self._criteria.items()}


__all__ = ["BlockRegistry", "FieldRegistry", "TimeRegistry", "ParamRegistry",
           "RuntimePolicyRegistry", "ConstraintRegistry"]
