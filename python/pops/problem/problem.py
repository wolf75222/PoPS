"""pops.problem.problem -- the top-level compilable assembly (Spec 5 sec.5.16 / sec.11).

:class:`Problem` is the inert, typed top-level assembly a user authors before lowering: physics
``block`` declarations, elliptic ``field`` problems, runtime ``param`` declarations, static ``aux``
inputs, ``output`` policies and the ``time`` scheme. It is an ASSEMBLY that CONTAINS descriptors; it
is NOT itself a :class:`pops.descriptors.Descriptor` (Spec 5 sec.6 / sec.15: an assemblage of
descriptors, not a descriptor). It answers the same inspectable surface -- it declares its
requirements / capabilities / options and answers ``available(context)`` / :meth:`validate` with an
EXPLAINABLE status before the runtime is ever touched. It computes nothing.

ADC-553 splits the assembly internals into TYPED registries (:mod:`pops.problem.registries`): the
facade owns one registry per family (blocks / fields / time / params / runtime policies /
constraints) and DELEGATES to them; it holds no flat dict and no inline subsystem logic. The
:meth:`validate` aggregates the child registries' per-family reports into ONE
:class:`~pops.problem.report.ProblemValidationReport`.

``pops.compile(problem, layout=..., time=...)`` -- the public front door -- lowers the assembly
through the existing codegen; ``pops.bind(compiled, ...)`` wires it onto the runtime. The Problem
here owns no codegen and no runtime of its own; it never imports ``_pops``. The Problem does NOT
choose the layout (ADC-526): the layout (``Uniform`` / ``AMR``) is supplied at
``pops.compile(problem, layout=...)``, so ONE Problem compiles under either. A layout given to the
constructor is still accepted (back-compat) and must agree with the one passed at compile.
"""
from pops.descriptors import Availability
from pops.mesh.layouts import AMR, Uniform
from pops.problem.amr_handle import ProblemAmrHandle
from pops.problem.registries import (
    _NO_KIND, BlockRegistry, ConstraintRegistry, FieldRegistry, ParamRegistry,
    RuntimePolicyRegistry, TimeRegistry)
from pops.problem.report import ProblemValidationReport


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

    def __init__(self, layout=None, name=None):
        self._name = str(name) if name else "Problem"
        # ADC-526: the Problem does NOT own a layout. Layout is supplied at compile
        # (pops.compile(problem, layout=Uniform(...)|AMR(...))), so ONE Problem compiles under
        # either. A constructor layout= is still accepted (back-compat) and, when given, is carried
        # here so pops.compile can cross-check it against the layout passed at compile.
        self._layout = layout
        self._block_registry = BlockRegistry()
        self._field_registry = FieldRegistry()
        self._time_registry = TimeRegistry()
        self._param_registry = ParamRegistry()
        self._runtime_registry = RuntimePolicyRegistry()
        self._constraint_registry = ConstraintRegistry()

    @property
    def name(self):
        return self._name

    # --- registry access (the typed internals, each independently inspectable) ----------
    @property
    def _blocks(self):
        return self._block_registry

    @property
    def _fields(self):
        return self._field_registry

    @property
    def _constraints(self):
        return self._constraint_registry

    # --- assembly (chaining setters delegate to the registries) -------------------------
    def block(self, name, physics, spatial=None):
        """Declare a physics block ``name`` (its ``physics`` model is REQUIRED). Chains."""
        self._block_registry.add(name, physics, spatial=spatial)
        return self

    def add_block(self, name, model, spatial=None, time=None, diagnostics=None):
        """Declare a physics block and return its stable :class:`~pops.problem.handles.BlockHandle`.

        The ADC-526 declarative form (a superset of :meth:`block`): a block carries its ``model``
        (required), its ``spatial`` discretisation, and the optional per-block ``time`` scheme and
        ``diagnostics``. Unlike the chaining :meth:`block`, this returns a stable HANDLE the user can
        hold to reference the block later. A duplicate name / missing model is refused loudly here.
        """
        return self._block_registry.add(name, model, spatial=spatial, time=time,
                                        diagnostics=diagnostics)

    def field(self, field_problem):
        """Register an elliptic :class:`~pops.fields.FieldProblem` (keyed on its name). Chains."""
        self._field_registry.add(field_problem)
        return self

    def add_field(self, field_problem):
        """Register a field problem and return its stable :class:`~pops.problem.handles.FieldHandle`.

        The handle-returning counterpart of the chaining :meth:`field` (ADC-526 stable handles).
        """
        return self._field_registry.add(field_problem)

    def param(self, name, default=None, *, kind=_NO_KIND):
        """Declare a runtime/const parameter and its default value. Chains.

        The KIND is a TYPED param object (Spec 5 sec.7), not a ``kind=`` string:
        ``problem.param(pops.physics.RuntimeParam("alpha", 1.0))`` /
        ``problem.param(pops.physics.ConstParam("gamma", 1.4))`` / ``problem.param("alpha", 1.0)``.
        A bare ``kind="const"/"runtime"`` keyword is REJECTED.
        """
        self._param_registry.add(name, default, kind=kind)
        return self

    def aux(self, name, value=None):
        """Declare a static aux input ``name`` (e.g. a background field). Chains."""
        self._runtime_registry.add_aux(name, value)
        return self

    def output(self, policy):
        """Attach an output / checkpoint policy. Chains."""
        self._runtime_registry.add_output(policy)
        return self

    def runtime(self, policies):
        """Attach a typed :class:`pops.output.RuntimePolicies` bundle (ADC-562). Chains.

        Groups the runtime concerns (output / checkpoint / diagnostics / schedules) out of the
        physics script: ``problem.runtime(pops.RuntimePolicies(output=..., checkpoint=...))``. The
        bundle's typed members are unpacked into the runtime registry (its output / checkpoint
        policies feed ``run(output_dir=...)`` exactly like :meth:`output`), and the bundle is retained
        so its self-contained ``validate`` runs with the compile context. ``output()`` / ``aux()``
        stay the granular primitives the bundle composes.
        """
        self._runtime_registry.set_policies(policies)
        return self

    def time(self, program):
        """Attach the time scheme (a ``pops.time.Program``) used at compile. Chains."""
        self._time_registry.set(program)
        return self

    def program(self, program):
        """Attach the whole-system time ``Program`` (the ADC-526 spelling of :meth:`time`). Chains."""
        self._time_registry.set(program)
        return self

    # --- compile-time compatibility accessors (read by pops.codegen.orchestration) ------
    @property
    def _time(self):
        return self._time_registry.program

    @property
    def _params(self):
        return dict(self._param_registry.items())

    @property
    def _param_declarations(self):
        """The ``{name: typed declaration}`` map for the bind-time domain check (ADC-541)."""
        return self._param_registry.declarations()

    @property
    def _aux(self):
        return self._runtime_registry.aux

    @property
    def _outputs(self):
        return self._runtime_registry.outputs

    # --- layout / amr access -------------------------------------------------
    @property
    def layout(self):
        return self._layout

    @property
    def amr(self):
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

    def blocks(self):
        """The declared blocks as ``{name: BlockHandle}`` (ADC-526 stable-handle accessor)."""
        from pops.problem.handles import BlockHandle
        return {name: BlockHandle(name, owner=self) for name in self._block_registry.names()}

    def fields(self):
        """The declared field problems as ``{name: FieldHandle}`` (ADC-526 stable-handle accessor)."""
        from pops.problem.handles import FieldHandle
        return {name: FieldHandle(name, owner=self) for name in self._field_registry.names()}

    # --- DescriptorProtocol surface (pure Python; no runtime, no codegen) ----
    def _layout_name(self):
        return self._layout.name if self._layout is not None else None

    def options(self):
        return {"name": self._name, "layout": self._layout_name(),
                "n_blocks": len(self._block_registry), "n_fields": len(self._field_registry),
                "n_params": len(self._param_registry), "n_aux": len(self._runtime_registry.aux),
                "n_outputs": len(self._runtime_registry.outputs),
                "has_time": self._time_registry.program is not None}

    def requirements(self):
        """The route's requirements as a typed :class:`~pops.descriptors_report.RequirementSet`."""
        from pops.descriptors_report import RequirementSet
        base = self._layout.requirements().to_dict() if self._layout is not None else {}
        req = RequirementSet(base)
        if len(self._field_registry):
            req.add("elliptic_solve")
        req.add("time_scheme")
        return req

    def capabilities(self):
        """The route's capabilities as a typed :class:`~pops.descriptors_report.CapabilitySet`."""
        from pops.descriptors_report import CapabilitySet
        caps = self._layout.capabilities().to_dict() if self._layout is not None else {}
        caps["blocks"] = sorted(self._block_registry.names())
        caps["fields"] = sorted(self._field_registry.names())
        return CapabilitySet(caps)

    def available(self, context=None):
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
        collisions = set(self._block_registry.names()) & set(self._field_registry.names())
        if collisions:
            return Availability.no(
                "block and field share name(s): %s" % ", ".join(sorted(collisions)),
                missing=list(collisions))
        return Availability.yes()

    def validate(self, context=None):
        """Structural validation; aggregate the registries' per-family reports and fail loud.

        Runs the layout check plus each registry's own ``validate``, folds them into ONE
        :class:`~pops.problem.report.ProblemValidationReport`, and raises (via ``raise_if_error``)
        when any error accumulated -- so the legacy callers that expect a loud exception keep
        working, while the report is available for per-family inspection (ADC-553).
        """
        report = self.validate_report(context)
        report.raise_if_error()
        return True

    def validate_report(self, context=None):
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
        report.extend(self._runtime_registry.validate(context))
        report.extend(self._constraint_registry.validate(context))
        collisions = set(self._block_registry.names()) & set(self._field_registry.names())
        if collisions:
            report.error("field", "name_collision",
                         "block and field share name(s): %s" % ", ".join(sorted(collisions)),
                         context={"names": sorted(collisions)})
        return report

    def _refuse_uniform_with_amr_criteria(self, report):
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

    def _field_validation_context(self, context):
        """The validation context handed to each field problem, carrying the mesh layout."""
        merged = dict(context) if isinstance(context, dict) else {}
        merged["layout"] = self._layout
        return merged

    def explain_routes(self):
        """Return a printable route matrix sourced from the C++ authoritative facts (sec.13.12.1)."""
        from pops._capabilities import native_capability_matrix
        return native_capability_matrix(owner=self.name, layout=self._layout_name() or "context",
                                        target="module")

    def lower(self, context=None):
        """The inert lowering record for the assembly (metadata only; no computation)."""
        from pops.descriptors_report import LoweredDescriptor
        return LoweredDescriptor(name=self._name, category=self.category,
                                 native_id=self.native_id, options=self.options())

    def inspect(self):
        """A plain, JSON-serialisable structured view of the assembly (no build, no compile)."""
        info = {"name": self._name, "category": self.category, "native_id": self.native_id,
                "options": self.options(), "requirements": self.requirements().to_dict(),
                "capabilities": self.capabilities().to_dict()}
        info["layout"] = self._layout.inspect() if self._layout is not None else None
        info["blocks"] = self._block_registry.inspect()
        info["fields"] = self._field_registry.inspect()
        info["params"] = self._param_registry.inspect()
        info["aux"] = self._runtime_registry.inspect()["aux"]
        info["outputs"] = self._runtime_registry.inspect()["outputs"]
        info["constraints"] = self._constraint_registry.inspect()
        info["time"] = self._time_registry.inspect()["program"]
        return info

    def to_dict(self):
        """A JSON-ready, array-free serialisation of the assembly for cache / codegen / debug.

        A superset of :meth:`inspect` that also names each block's stable handle id and the
        refinement criteria recorded on the constraint registry, so the whole declaration
        round-trips through a plain dict with no runtime object and no numpy array (ADC-526).
        """
        info = self.inspect()
        info["handles"] = {"blocks": [h.handle_id for h in self.blocks().values()],
                           "fields": [h.handle_id for h in self.fields().values()]}
        return info

    def __str__(self):
        return ("%s [%s] layout=%s | blocks=%d | fields=%d | params=%d | aux=%d | time=%s"
                % (self._name, self.category, self._layout_name() or "none",
                   len(self._block_registry), len(self._field_registry),
                   len(self._param_registry), len(self._runtime_registry.aux),
                   "set" if self._time_registry.program is not None else "none"))

    def __repr__(self):
        return ("Problem(name=%r, layout=%s, blocks=%s, fields=%s)"
                % (self._name, self._layout_name() or "none",
                   sorted(self._block_registry.names()), sorted(self._field_registry.names())))


__all__ = ["Problem"]
