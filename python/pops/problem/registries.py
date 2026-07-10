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
from __future__ import annotations

from typing import Any

from pops.problem._registry_freeze import (
    FreezableRegistry as _FreezableRegistry, flatten_freeze_members)
from pops.problem.handles import BlockHandle, FieldHandle
from pops.problem.report import ProblemValidationReport

# Sentinel distinguishing "no kind= passed" from "kind=None": ParamRegistry rejects any kind=
# keyword (Spec 5 sec.7) with a clear error naming the typed alternative.
_NO_KIND = object()


def _strict_name(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value:
        raise TypeError("%s must be a non-empty string" % where)
    return value


class BlockRegistry(_FreezableRegistry):
    """The physics blocks declared on a Problem (name -> model + spatial + time + diagnostics).

    A block records its physics ``model`` (required), its ``spatial`` discretisation brick, and the
    optional per-block ``time`` scheme and ``diagnostics`` (ADC-526's ``add_block`` superset). A
    duplicate name is refused loudly at declaration -- the earliest, per-family error.
    """

    family = "block"

    def __init__(self, owner: Any) -> None:
        from pops.model import OwnerPath
        self._owner_path = OwnerPath.coerce(owner)
        self._blocks = {}

    @property
    def owner_path(self) -> Any:
        return self._owner_path

    def _freezable_members(self) -> Any:
        return flatten_freeze_members(*(
            value for spec in self._blocks.values()
            for value in (spec["model"], spec["spatial"], spec["time"], spec["diagnostics"])))

    def add(self, name: Any, model: Any, *, spatial: Any = None, time: Any = None,
            diagnostics: Any = None) -> Any:
        """Record a block ``name`` with its ``model`` (required). Returns a stable :class:`BlockHandle`."""
        self._guard_frozen("add a block")
        key = _strict_name(name, "block name")
        if model is None:
            raise ValueError("add_block(%r): a physics model is required" % key)
        if key in self._blocks:
            raise ValueError("add_block(%r): a block of that name already exists" % key)
        self._blocks[key] = {"model": model, "spatial": spatial, "time": time,
                             "diagnostics": diagnostics}
        return BlockHandle(key, owner=self.owner_path)

    def get(self, name: Any) -> Any:
        return self._blocks.get(_strict_name(name, "block name"))

    def names(self) -> Any:
        return list(self._blocks)

    def spec(self, name: Any) -> Any:
        """The full ``{model, spatial, time, diagnostics}`` record for @p name (or ``None``)."""
        return self._blocks.get(_strict_name(name, "block name"))

    def items(self) -> Any:
        return self._blocks.items()

    def __iter__(self) -> Any:
        return iter(self._blocks)

    def __len__(self) -> int:
        return len(self._blocks)

    def __contains__(self, name: Any) -> bool:
        return isinstance(name, str) and name in self._blocks

    def validate(self, context: Any = None) -> Any:
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

    def inspect(self) -> Any:
        return {name: {"model": getattr(spec["model"], "name", repr(spec["model"])),
                       "spatial": getattr(spec["spatial"], "name", spec["spatial"]),
                       "time": getattr(spec["time"], "name", None),
                       "diagnostics": getattr(spec["diagnostics"], "name", None)}
                for name, spec in self._blocks.items()}


class FieldRegistry(_FreezableRegistry):
    """The elliptic field problems declared on a Problem (keyed on the field's name)."""

    family = "field"

    def __init__(self, owner: Any) -> None:
        from pops.model import OwnerPath
        self._owner_path = OwnerPath.coerce(owner)
        self._fields = {}

    @property
    def owner_path(self) -> Any:
        return self._owner_path

    def _freezable_members(self) -> Any:
        return list(self._fields.values())

    def add(self, field_problem: Any) -> Any:
        """Register a :class:`pops.fields.FieldProblem` (keyed on its name). Returns a :class:`FieldHandle`."""
        self._guard_frozen("add a field")
        from pops.fields import FieldProblem  # lazy: keep pops.problem free of a fields module edge
        if not isinstance(field_problem, FieldProblem):
            raise TypeError("field: expected a pops.fields.FieldProblem; got %r"
                            % type(field_problem).__name__)
        key = _strict_name(field_problem.name, "field name")
        if key in self._fields:
            raise ValueError("field: a field named %r already exists" % key)
        self._fields[key] = field_problem
        return FieldHandle(key, owner=self.owner_path)

    def get(self, name: Any) -> Any:
        return self._fields.get(_strict_name(name, "field name"))

    def names(self) -> Any:
        return list(self._fields)

    def items(self) -> Any:
        return self._fields.items()

    def solvers(self) -> Any:
        """The ``{field_name: solver}`` mapping (skips a field with no solver)."""
        return {name: fp.solver for name, fp in self._fields.items() if fp.solver is not None}

    def __iter__(self) -> Any:
        return iter(self._fields)

    def __len__(self) -> int:
        return len(self._fields)

    def __contains__(self, name: Any) -> bool:
        return isinstance(name, str) and name in self._fields

    def validate(self, context: Any = None) -> Any:
        """Report each field problem's own validation failure (structured, never a bare raise)."""
        report = ProblemValidationReport()
        for name, field in self._fields.items():
            try:
                field.validate(context)
            except Exception as exc:  # noqa: BLE001 -- surface the field's own message as an issue
                report.error(self.family, "field_invalid", str(exc), context={"field": name})
        return report

    def inspect(self) -> Any:
        return {name: fp.inspect() for name, fp in self._fields.items()}


class TimeRegistry(_FreezableRegistry):
    """The whole-system time scheme slot (a single ``pops.time.Program``, attached at compile)."""

    family = "time"

    def __init__(self) -> None:
        self._program = None

    def _freezable_members(self) -> Any:
        return flatten_freeze_members(self._program)

    def set(self, program: Any) -> None:
        """Record the time scheme (the whole-system Program). Overwrites a prior one."""
        self._guard_frozen("set the time scheme")
        self._program = program

    @property
    def program(self) -> Any:
        return self._program

    def names(self) -> Any:
        return [getattr(self._program, "name", "program")] if self._program is not None else []

    def __iter__(self) -> Any:
        return iter([self._program] if self._program is not None else [])

    def validate(self, context: Any = None) -> Any:
        """The time scheme is optional at assembly (supplied at compile); nothing to reject here."""
        return ProblemValidationReport()

    def inspect(self) -> Any:
        return {"program": getattr(self._program, "name", None)
                if self._program is not None else None}


class ParamRegistry(_FreezableRegistry):
    """The runtime / const parameter declarations (name -> {default, kind})."""

    family = "params"

    def __init__(self) -> None:
        self._params = {}
        # The TYPED declaration object per name (a pops.params RuntimeParam / ConstParam carrying its
        # domain), retained so the bind-time domain check (ADC-541) can call decl.check_bind(value).
        # A bare (name, default) declaration has no typed object -> None.
        self._declarations = {}

    def _freezable_members(self) -> Any:
        return [d for d in self._declarations.values() if d is not None]

    def add(self, name: Any, default: Any = None, *, kind: Any = _NO_KIND) -> None:
        """Declare a parameter. A bare ``kind=`` string is rejected (Spec 5 sec.7)."""
        self._guard_frozen("declare a param")
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
            key = _strict_name(name.name, "parameter name")
            kind_name = _strict_name(name.kind, "parameter kind")
            self._params[key] = {"default": name.value, "kind": kind_name}
            self._declarations[key] = name
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
            key = _strict_name(name.name, "parameter name")
            self._params[key] = {"default": declared, "kind": kind_of}
            self._declarations[key] = name
        else:
            key = _strict_name(name, "parameter name")
            self._params[key] = {"default": default, "kind": "const"}
            self._declarations[key] = None

    def get(self, name: Any) -> Any:
        return self._params.get(_strict_name(name, "parameter name"))

    def names(self) -> Any:
        return list(self._params)

    def items(self) -> Any:
        return self._params.items()

    def declarations(self) -> Any:
        """The ``{name: typed declaration}`` map (a ``RuntimeParam``/``ConstParam`` or ``None``).

        The bind-time domain check (ADC-541) reads it to call ``decl.check_bind(value)`` on each
        supplied runtime param. A name declared without a typed object maps to ``None``.
        """
        return dict(self._declarations)

    def __iter__(self) -> Any:
        return iter(self._params)

    def __len__(self) -> int:
        return len(self._params)

    def validate(self, context: Any = None) -> Any:
        return ProblemValidationReport()

    def inspect(self) -> Any:
        return {name: dict(spec) for name, spec in self._params.items()}


class RuntimePolicyRegistry(_FreezableRegistry):
    """Runtime-facing declarations: static aux inputs and output / checkpoint policies.

    These describe what the runtime does with the assembly (background aux fields, when to write /
    checkpoint); they carry no runtime data themselves. Output entries are validated to be real
    policy descriptors (a non-policy object is a typo caught here, not at run time).
    """

    family = "runtime"
    _POLICY_CATEGORIES = ("output_policy", "checkpoint_policy")
    _DIAGNOSTIC_CATEGORIES = ("diagnostic_norm", "diagnostic_integral", "diagnostic_minmax",
                              "conservation_check")

    def __init__(self) -> None:
        self._aux = {}
        self._outputs = []
        # Declared typed diagnostic measures (ADC-542). The bundle drops them nowhere now: they are
        # unpacked here so the run loop FIRES them via the native reductions (registries used to
        # unpack only output / checkpoint, silently dropping diagnostics).
        self._diagnostics = []
        # The typed RuntimePolicies bundle (ADC-562), retained so its self-contained validate runs
        # with the compile context; its output / checkpoint members are ALSO unpacked into _outputs.
        self._policies = None

    def _freezable_members(self) -> Any:
        return flatten_freeze_members(self._outputs, self._diagnostics, self._policies)

    def add_aux(self, name: Any, value: Any = None) -> None:
        """Declare a static aux input ``name`` (e.g. a background field)."""
        self._guard_frozen("declare an aux input")
        self._aux[_strict_name(name, "aux name")] = value

    def add_output(self, policy: Any) -> None:
        """Attach an output / checkpoint policy descriptor."""
        self._guard_frozen("attach an output policy")
        self._outputs.append(policy)

    def set_policies(self, policies: Any) -> None:
        """Record a typed :class:`pops.output.RuntimePolicies` bundle (ADC-562).

        Unpacks the bundle's output / checkpoint members into ``_outputs`` (so ``run(output_dir=...)``
        fires them exactly like :meth:`add_output`) and retains the bundle for its self-contained
        :meth:`validate`. A non-bundle argument is refused loudly (no options bag)."""
        self._guard_frozen("attach runtime policies")
        from pops.output.runtime_policies import RuntimePolicies
        if not isinstance(policies, RuntimePolicies):
            raise TypeError(
                "problem.runtime(...) expects a typed pops.RuntimePolicies bundle; got %r. Group the "
                "runtime concerns with pops.RuntimePolicies(output=..., checkpoint=..., "
                "diagnostics=..., schedules=...)." % (type(policies).__name__,))
        self._policies = policies
        for policy in policies.outputs():
            self._outputs.append(policy)
        for measure in policies.diagnostics:
            self._diagnostics.append(measure)

    @property
    def policies(self) -> Any:
        return self._policies

    @property
    def aux(self) -> Any:
        return dict(self._aux)

    @property
    def outputs(self) -> Any:
        return list(self._outputs)

    @property
    def diagnostics(self) -> Any:
        return list(self._diagnostics)

    def names(self) -> Any:
        return sorted(self._aux) + [getattr(p, "name", repr(p)) for p in self._outputs]

    def __iter__(self) -> Any:
        return iter(self._outputs)

    def validate(self, context: Any = None) -> Any:
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
        for measure in self._diagnostics:
            cat = getattr(measure, "category", None)
            if cat not in self._DIAGNOSTIC_CATEGORIES:
                report.error(
                    self.family, "bad_diagnostic_measure",
                    "diagnostics=[...] expects a pops.diagnostics measure (Norm / Integral / "
                    "MinMax / ConservationCheck); got %r (category %r)"
                    % (type(measure).__name__, cat),
                    context={"measure": type(measure).__name__, "category": cat})
        if self._policies is not None:
            report.extend(self._policies.validate(context))
        return report

    def inspect(self) -> Any:
        info = {"aux": sorted(self._aux),
                "outputs": [getattr(p, "name", repr(p)) for p in self._outputs],
                "diagnostics": [getattr(m, "name", repr(m)) for m in self._diagnostics]}
        if self._policies is not None:
            info["policies"] = self._policies.inspect().to_dict()
        return info


class ConstraintRegistry(_FreezableRegistry):
    """Structural constraints + layout-free AMR refinement criteria (ADC-526).

    A Problem carries no layout, so the AMR refinement criteria (refine / regrid / nesting / patches)
    are recorded HERE as inert descriptors and applied to the ``Uniform`` / ``AMR`` layout at
    ``pops.compile(problem, layout=...)``. It also holds cross-family structural checks (a block and
    a field must not share a name). It owns no layout and no runtime.
    """

    family = "amr"

    def __init__(self) -> None:
        self._criteria = {}  # refine / regrid / nesting / patches -> descriptor

    def _freezable_members(self) -> Any:
        return flatten_freeze_members(self._criteria)

    def set_refinement(self, *, refine: Any = None, regrid: Any = None, nesting: Any = None,
                       patches: Any = None) -> None:
        """Record the AMR refinement criteria (layout-free; applied at compile)."""
        self._guard_frozen("record AMR refinement criteria")
        if refine is not None:
            self._criteria["refine"] = refine
        if regrid is not None:
            self._criteria["regrid"] = regrid
        if nesting is not None:
            self._criteria["nesting"] = nesting
        if patches is not None:
            self._criteria["patches"] = patches

    @property
    def refinement(self) -> Any:
        return dict(self._criteria)

    def names(self) -> Any:
        return list(self._criteria)

    def __iter__(self) -> Any:
        return iter(self._criteria.items())

    def __len__(self) -> int:
        return len(self._criteria)

    def validate(self, context: Any = None) -> Any:
        """No layout at assembly, so refinement criteria cannot be checked here (deferred to compile)."""
        return ProblemValidationReport()

    def inspect(self) -> Any:
        return {kind: getattr(desc, "name", repr(desc))
                for kind, desc in self._criteria.items()}


__all__ = ["BlockRegistry", "FieldRegistry", "TimeRegistry", "ParamRegistry",
           "RuntimePolicyRegistry", "ConstraintRegistry"]
