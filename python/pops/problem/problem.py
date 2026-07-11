"""The inert, typed, top-level Problem assembly.

Problem owns exactly one registry per declaration family and aggregates their inspectable
validation reports.  It computes nothing and imports neither the native extension nor runtime or
codegen.  ``pops.compile(problem, layout=..., time=...)`` is the lowering boundary; one assembly can
therefore be compiled against different compatible layout descriptors.
"""
from __future__ import annotations

from typing import Any

from pops.descriptors import Availability
from pops.mesh.layouts import AMR, Uniform
from pops.problem.amr_handle import ProblemAmrHandle
from pops.problem.registries import (
    BlockRegistry, ConstraintRegistry, FieldRegistry, ParamRegistry,
    RuntimePolicyRegistry, TimeRegistry)
from pops.problem._inspection import inspect_payload, serialization_payload
from pops.problem.report import ProblemValidationReport
from pops.model import OwnerKind, OwnerPath


class Problem:
    """A typed, inert top-level assembly: blocks + fields + params + aux + outputs + time.

    ``Problem(name="plasma")`` then chained::

        problem = (pops.Problem(name="plasma")
                   .block("ne", physics=model, spatial=pops.FiniteVolume())
                   .field(pops.fields.PoissonProblem(unknown="phi", equation=eq, solver=mg))
                   .time(pops.time.Program(...)))
        compiled = pops.compile(problem, layout=Uniform(CartesianMesh()))

    Each assembly setter RETURNS the Problem so calls chain. A Problem CONTAINS descriptors (the
    blocks' physics, the field problems) but is NOT itself a :class:`pops.descriptors.Descriptor`
    (Spec 5 sec.6 / sec.15). It exposes the same inspectable surface --
    ``requirements`` / ``capabilities`` / ``options`` / ``available`` / ``validate`` / ``inspect`` /
    ``lower`` -- implemented DIRECTLY here (by delegating to the registries), so it duck-types as a
    route-describing object without inheriting a descriptor identity.
    """

    category = "problem"
    #: A Problem names a pure-Python assembly, not a single native C++ symbol.
    native_id = None

    def __setattr__(self, name: str, value: Any) -> None:
        if name in ("_name", "_owner_path") and hasattr(self, name):
            raise AttributeError("pops.Problem identity is immutable; construct a new Problem")
        if getattr(self, "_frozen", False):
            if name != "_frozen" or value is not True:
                raise RuntimeError("pops.Problem is frozen: cannot change %s" % name)
        object.__setattr__(self, name, value)

    def __delattr__(self, name: str) -> None:
        if name in ("_name", "_owner_path"):
            raise AttributeError("pops.Problem identity is immutable; construct a new Problem")
        if getattr(self, "_frozen", False):
            raise RuntimeError("pops.Problem is frozen: cannot delete %s" % name)
        object.__delattr__(self, name)

    def __init__(self, layout: Any = None, name: Any = None) -> None:
        if name is None:
            name = "Problem"
        if not isinstance(name, str) or not name:
            raise TypeError("Problem: name must be a non-empty string")
        self._name = name
        self._owner_path = OwnerPath.fresh(OwnerKind.CASE, self._name)
        # ADC-563 freeze: a Problem is MUTABLE during assembly and FROZEN by pops.compile. After
        # freeze every mutating setter RAISES (naming the Problem) and freeze() returns a stable
        # AuthoringSnapshot the compile cache keys on.
        self._frozen = False
        self._snapshot = None
        # ADC-526: the Problem does NOT own a layout. Layout is supplied at compile
        # (pops.compile(problem, layout=Uniform(...)|AMR(...))), so ONE Problem compiles under
        # either. A constructor layout= is still accepted (back-compat) and, when given, is carried
        # here so pops.compile can cross-check it against the layout passed at compile.
        self._layout = layout
        self._block_registry = BlockRegistry(self.owner_path)
        self._field_registry = FieldRegistry(self.owner_path)
        self._time_registry = TimeRegistry()
        self._param_registry = ParamRegistry(self.owner_path)
        self._runtime_registry = RuntimePolicyRegistry()
        self._constraint_registry = ConstraintRegistry()

    @property
    def name(self) -> Any:
        return self._name

    @property
    def owner_path(self) -> OwnerPath:
        """Immutable qualified identity anchor for every handle owned by this Problem."""
        return self._owner_path

    # --- freeze lifecycle (ADC-563) -----------------------------------------------------
    def _guard_mutable(self, what: Any) -> None:
        """Raise when a mutating setter runs after :meth:`freeze` (ADC-563), naming the Problem."""
        if self._frozen:
            raise RuntimeError(
                "pops.Problem %r is frozen (ADC-563): cannot %s after pops.compile froze it. A "
                "compiled artifact is frozen to exactly the assembly it was compiled from; author a "
                "fresh Problem (or edit BEFORE compile) and recompile -- a post-compile mutation "
                "cannot change a bound artifact." % (self._name, what))

    def freeze(self) -> Any:
        """Freeze the assembly and return its stable :class:`~pops.problem._snapshot.AuthoringSnapshot`.

        ``pops.compile`` calls this on the Problem it compiles: after freeze every mutating setter
        RAISES (naming the Problem), the member registries and their descriptors are sealed, and the
        returned snapshot's ``.hash`` is the frozen identity the compile cache key folds in. Idempotent
        -- a second call returns the SAME snapshot, so ``compile`` can freeze a Problem the caller
        already froze."""
        if self._frozen:
            return self._snapshot
        from pops.problem._snapshot import build_problem_snapshot
        # Two-phase commit: canonicalisation can call descriptor projections and therefore fail.
        # Build and validate the complete inert snapshot while every authoring object is still
        # mutable; only a successful candidate is followed by the irreversible registry/descriptor
        # seal.  A serialization error leaves the Problem exactly as editable as it was before the
        # call, so the user can repair the declaration and retry freeze().
        candidate = build_problem_snapshot(self)
        self._freeze_registries()
        self._snapshot = candidate
        self._frozen = True
        return self._snapshot

    def _freeze_registries(self) -> None:
        """Cascade freeze to each registry and the member descriptors (fields' solvers, layout)."""
        from pops.problem._freeze_transaction import freeze_problem_graph
        freeze_problem_graph(self)

    @property
    def frozen(self) -> Any:
        return self._frozen

    @property
    def snapshot(self) -> Any:
        """The :class:`AuthoringSnapshot` from the last :meth:`freeze` (``None`` before freeze)."""
        return self._snapshot

    # --- registry access (the typed internals, each independently inspectable) ----------
    @property
    def _blocks(self) -> Any:
        return self._block_registry

    @property
    def _fields(self) -> Any:
        return self._field_registry

    @property
    def _constraints(self) -> Any:
        return self._constraint_registry

    # --- assembly (chaining setters delegate to the registries) -------------------------
    def block(self, name: Any, physics: Any, spatial: Any = None) -> Any:
        """Declare a physics block ``name`` (its ``physics`` model is REQUIRED). Chains."""
        self._guard_mutable("add a block")
        self._block_registry.add(name, physics, spatial=spatial)
        return self

    def add_block(self, name: Any, model: Any, spatial: Any = None, time: Any = None,
                  diagnostics: Any = None) -> Any:
        """Declare a physics block and return its stable :class:`~pops.problem.handles.BlockHandle`.

        The ADC-526 declarative form (a superset of :meth:`block`): a block carries its ``model``
        (required), its ``spatial`` discretisation, and the optional per-block ``time`` scheme and
        ``diagnostics``. Unlike the chaining :meth:`block`, this returns a stable HANDLE the user can
        hold to reference the block later. A duplicate name / missing model is refused loudly here.
        """
        self._guard_mutable("add a block")
        return self._block_registry.add(name, model, spatial=spatial, time=time,
                                        diagnostics=diagnostics)

    def field(self, field_problem: Any) -> Any:
        """Register an elliptic :class:`~pops.fields.FieldProblem` (keyed on its name). Chains."""
        self._guard_mutable("add a field")
        self._field_registry.add(field_problem)
        return self

    def add_field(self, field_problem: Any) -> Any:
        """Register a field problem and return its stable :class:`~pops.problem.handles.FieldHandle`.

        The handle-returning counterpart of the chaining :meth:`field` (ADC-526 stable handles).
        """
        self._guard_mutable("add a field")
        return self._field_registry.add(field_problem)

    def param(self, declaration: Any) -> Any:
        """Register a case-owned typed parameter and return its ParamHandle.

        Case parameters serve runtime consumers outside one physics model, for
        example AMR indicators or solver tolerances.  Only explicit canonical
        declarations are accepted; the removed ``(name, value)`` shorthand
        cannot silently turn a runtime value into a compile-time constant.
        """
        self._guard_mutable("declare a param")
        return self._param_registry.add(declaration)

    def value(self, parameter: Any) -> Any:
        """Return an owner-qualified symbolic read of a case parameter."""
        from pops.ir import ValueExpr

        return ValueExpr(self._param_registry.handle(parameter))

    def aux(self, name: Any, value: Any = None) -> Any:
        """Declare a static aux input ``name`` (e.g. a background field). Chains."""
        self._guard_mutable("declare an aux input")
        self._runtime_registry.add_aux(name, value)
        return self

    def output(self, policy: Any) -> Any:
        """Attach an output / checkpoint policy. Chains."""
        self._guard_mutable("attach an output policy")
        self._runtime_registry.add_output(policy)
        return self

    def runtime(self, policies: Any) -> Any:
        """Attach a typed :class:`pops.output.RuntimePolicies` bundle (ADC-562). Chains.

        Groups the runtime concerns (output / checkpoint / diagnostics / schedules) out of the
        physics script: ``problem.runtime(pops.RuntimePolicies(output=..., checkpoint=...))``. The
        bundle's typed members are unpacked exactly once into the runtime registry (its output /
        checkpoint policies feed ``run(output_dir=...)`` exactly like :meth:`output`). The input
        bundle is not retained as a second declaration authority: validation, inspection and the
        compile snapshot consume only those flattened members. ``output()`` / ``aux()`` stay the
        granular primitives the bundle composes.
        """
        self._guard_mutable("attach runtime policies")
        self._runtime_registry.set_policies(policies)
        return self

    def time(self, program: Any) -> Any:
        """Attach the time scheme (a ``pops.time.Program``) used at compile. Chains."""
        self._guard_mutable("set the time scheme")
        self._time_registry.set(program)
        return self

    def program(self, program: Any) -> Any:
        """Attach the whole-system time ``Program`` (the ADC-526 spelling of :meth:`time`). Chains."""
        self._guard_mutable("set the time program")
        self._time_registry.set(program)
        return self

    # --- compile-time compatibility accessors (read by pops.codegen.orchestration) ------
    @property
    def _time(self) -> Any:
        return self._time_registry.program

    @property
    def _params(self) -> Any:
        return self._param_registry.declarations()

    @property
    def _param_declarations(self) -> Any:
        """Canonical declarations retained for strict bind-schema construction."""
        return self._param_registry.declarations()

    @property
    def _aux(self) -> Any:
        return self._runtime_registry.aux

    @property
    def _outputs(self) -> Any:
        return self._runtime_registry.outputs

    @property
    def _diagnostics(self):
        return self._runtime_registry.diagnostics

    # --- layout / amr access -------------------------------------------------
    @property
    def layout(self) -> Any:
        return self._layout

    @property
    def amr(self) -> Any:
        """The AMR refinement-policy handle (records criteria on the constraint registry; ADC-526).

        A layout-free Problem always exposes ``.amr`` -- the criteria are applied to the layout at
        ``pops.compile(problem, layout=AMR(...))``. When a layout WAS given to the constructor it
        must be AMR (a Uniform layout has no level to refine onto), so a back-compat
        ``Problem(layout=Uniform(...)).amr`` is still refused loudly.
        """
        if self._layout is not None and not isinstance(self._layout, AMR):
            raise ValueError(
                "problem.amr: only available with no constructor layout (supply layout=AMR(...) at "
                "compile) or layout=AMR(...); this problem has layout %r"
                % type(self._layout).__name__)
        return ProblemAmrHandle(self)

    def blocks(self) -> Any:
        """The declared blocks as ``{name: BlockHandle}`` (ADC-526 stable-handle accessor)."""
        return self._block_registry.handles()

    def fields(self) -> Any:
        """The declared field problems as ``{name: FieldHandle}`` (ADC-526 stable-handle accessor)."""
        return self._field_registry.handles()

    def qualify(self, declaration: Any, *, block: Any = None) -> Any:
        """Resolve a model-local handle to one block-qualified instance handle.

        ``block`` is a :class:`BlockHandle`, never a string.  Omitting it is accepted only when one
        and only one registered block instantiates the declaration owner; ambiguity reports every
        candidate owner before lowering.
        """
        return self._block_registry.qualify(declaration, block=block)

    def resolve(self, declaration: Any, *, block: Any = None) -> Any:
        """Return the canonical identity of one authenticated declaration reference.

        Model-local references are first qualified to exactly one block.  A local reference used by
        multiple blocks is therefore rejected with all candidate owners instead of being guessed.
        """
        from pops.model import Handle, ParamHandle
        from pops.problem.handles import BlockHandle, FieldHandle

        case_root_owned = (
            isinstance(declaration, Handle)
            and len(declaration.owner_path.nodes) == 1
            and declaration.owner_path.nodes[0].kind is OwnerKind.CASE
        )
        if isinstance(declaration, BlockHandle) or (
            case_root_owned and declaration.kind == "block"
        ):
            if block is not None:
                raise TypeError("block declarations do not accept block=")
            return self._block_registry.canonical_block(declaration)
        if isinstance(declaration, FieldHandle) or (
            case_root_owned and declaration.kind == "field"
        ):
            if block is not None:
                raise TypeError("case-owned fields do not accept block=")
            return self._field_registry.canonicalize(declaration)
        if isinstance(declaration, ParamHandle) and case_root_owned:
            if block is not None:
                raise TypeError("case-owned parameters do not accept block=")
            return self._param_registry.canonicalize(declaration)
        return self._block_registry.canonicalize(declaration, block=block)

    # --- DescriptorProtocol surface (pure Python; no runtime, no codegen) ----
    def _layout_name(self) -> Any:
        return self._layout.name if self._layout is not None else None

    def options(self) -> Any:
        """The authoring summary: name, layout and the per-registry counts (a plain dict)."""
        return {"name": self._name, "layout": self._layout_name(),
                "n_blocks": len(self._block_registry), "n_fields": len(self._field_registry),
                "n_params": len(self._param_registry), "n_aux": len(self._runtime_registry.aux),
                "n_outputs": len(self._runtime_registry.outputs),
                "has_time": self._time_registry.program is not None}

    def requirements(self) -> Any:
        """The route's requirements as a typed :class:`~pops.descriptors_report.RequirementSet`."""
        from pops.descriptors_report import RequirementSet
        base = self._layout.requirements().to_dict() if self._layout is not None else {}
        req = RequirementSet(base)
        if len(self._field_registry):
            req.add("elliptic_solve")
        req.add("time_scheme")
        return req

    def capabilities(self) -> Any:
        """The route's capabilities as a typed :class:`~pops.descriptors_report.CapabilitySet`."""
        from pops.descriptors_report import CapabilitySet
        caps = self._layout.capabilities().to_dict() if self._layout is not None else {}
        caps["blocks"] = sorted(self._block_registry.names())
        caps["fields"] = sorted(self._field_registry.names())
        return CapabilitySet(caps)

    def available(self, context: Any = None) -> Any:
        """An EXPLAINABLE availability status, computed from the parts (no runtime)."""
        if self._layout is not None:
            layout_status = self._layout.available(context)
            if not layout_status.ok:
                return layout_status
        if not len(self._block_registry):
            return Availability.no("problem has no block; add one with .block(name, physics)",
                                   missing=["block"])
        for name in self._block_registry.names():
            if self._block_registry.spec(name).get("model") is None:
                return Availability.no("block %r has no physics model" % name, missing=["physics"])
        for field in self._field_registry.items():
            status = field[1].available(context)
            if not status.ok:
                return status
        return Availability.yes()

    def validate(self, context: Any = None) -> Any:
        """Structural validation; aggregate the registries' per-family reports and fail loud.

        Runs the layout check plus each registry's own ``validate``, folds them into ONE
        :class:`~pops.problem.report.ProblemValidationReport`, and raises (via ``raise_if_error``)
        when any error accumulated -- so the legacy callers that expect a loud exception keep
        working, while the report is available for per-family inspection (ADC-553).
        """
        report = self.validate_report(context)
        report.raise_if_error()
        return True

    def validate_report(self, context: Any = None) -> Any:
        """Aggregate the per-family validation reports into ONE report (no raise; ADC-553).

        When the Problem carries NO layout (the ADC-526 default), the layout-specific checks
        (the layout's own ``validate`` and the Uniform-with-AMR-criteria refusal) DEFER to
        ``pops.compile(problem, layout=...)``, where the layout is finally known; the structural
        registry checks always run.
        """
        report = ProblemValidationReport(subject=self)
        if self._layout is not None:
            self._refuse_uniform_with_amr_criteria(report)
            try:
                self._layout.validate(context)
            except Exception as exc:  # noqa: BLE001 -- surface the layout's own message as an issue
                report.error("layout", "layout_invalid", str(exc))
        report.extend(self._block_registry.validate(context))
        # Carry the mesh layout into each field problem's validation so its solver can refuse a
        # layout it cannot serve (Spec 6 sec.8/9), precisely, before any compile.
        report.extend(self._field_registry.validate(self._field_validation_context(context)))
        report.extend(self._param_registry.validate(context))
        runtime_context: dict[str, Any] = dict(context) if isinstance(context, dict) else {}
        runtime_context["declaration_resolver"] = self.resolve
        report.extend(self._runtime_registry.validate(runtime_context))
        report.extend(self._constraint_registry.validate(context))
        return report

    def _refuse_uniform_with_amr_criteria(self, report: Any) -> None:
        """Refuse a ``Uniform`` layout with an AMR criterion and no explicit escape (sec.8.6/5.14)."""
        layout = self._layout
        criterion = getattr(layout, "refine", None) if isinstance(layout, Uniform) else None
        if criterion is None or getattr(layout, "ignore_amr", None) is not None:
            return
        sub_criteria = getattr(criterion, "criteria", None)
        names = [c.name for c in sub_criteria] if sub_criteria is not None else [criterion.name]
        report.error(
            "amr", "uniform_with_amr_criteria",
            "layout=Uniform(...) carries active AMR criteria (%s) but a single-level layout has no "
            "level to refine onto; a criterion is never silently ignored. Use layout=AMR(...) to "
            "actually refine, or pass Uniform(mesh, refine=..., "
            "ignore_amr=pops.mesh.amr.IgnoreAMRCriteria()) to keep the criterion attached but "
            "explicitly unused." % ", ".join(names),
            context={"criteria": names})

    def _field_validation_context(self, context: Any) -> Any:
        """The validation context handed to each field problem, carrying the mesh layout."""
        merged: dict[str, Any] = dict(context) if isinstance(context, dict) else {}
        merged["layout"] = self._layout
        merged["declaration_resolver"] = self.resolve
        return merged

    def explain_routes(self) -> Any:
        """Return a printable route matrix sourced from the C++ authoritative facts (sec.13.12.1)."""
        from pops._capabilities import native_capability_matrix
        return native_capability_matrix(owner=self.name, layout=self._layout_name() or "context",
                                        target="module")

    def lower(self, context: Any = None) -> Any:
        """The inert lowering record for the assembly (metadata only; no computation)."""
        from pops.descriptors_report import LoweredDescriptor
        return LoweredDescriptor(name=self._name, category=self.category,
                                 native_id=self.native_id, options=self.options())

    def inspect(self) -> Any:
        """A typed :class:`~pops.problem.report_view.ProblemReport` of the assembly (ADC-564).

        Attributes + ``to_dict()`` (never a dict subclass), carrying the name / blocks / fields /
        params / aux / outputs / constraints / requirements / capabilities. Inert: no build, no
        compile, no validation. ``pops.inspect(problem)`` is the explicit dict bridge over its
        ``to_dict()``.
        """
        from pops.problem.report_view import ProblemReport
        return ProblemReport(self._inspect_payload())

    def _inspect_payload(self) -> Any:
        """The ordered inspection dict (the historical inspect() shape) the ProblemReport wraps."""
        return inspect_payload(self)

    def to_dict(self) -> Any:
        """A JSON-ready, array-free inspection serialisation for codegen / debug.

        A superset of :meth:`_inspect_payload` that also names each block's stable handle id, so the
        whole declaration round-trips through a plain dict with no runtime object and no numpy array
        (ADC-526). It stays a plain dict for codegen, distinct from the typed :meth:`inspect` report.
        The compile-cache snapshot reads the raw typed registries instead, because display summaries
        here intentionally abbreviate descriptors and are not a complete structural identity.
        """
        return serialization_payload(self)

    def __str__(self) -> str:
        return ("%s [%s] layout=%s | blocks=%d | fields=%d | params=%d | aux=%d | time=%s"
                % (self._name, self.category, self._layout_name() or "none",
                   len(self._block_registry), len(self._field_registry),
                   len(self._param_registry), len(self._runtime_registry.aux),
                   "set" if self._time_registry.program is not None else "none"))

    def __repr__(self) -> str:
        return ("Problem(name=%r, layout=%s, blocks=%s, fields=%s)"
                % (self._name, self._layout_name() or "none",
                   sorted(self._block_registry.names()), sorted(self._field_registry.names())))


__all__ = ["Problem"]
