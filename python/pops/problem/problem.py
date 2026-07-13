"""The inert, typed, top-level Case assembly.

Case owns one registry per declaration family and aggregates their validation reports. It is
inert; ``pops.resolve(pops.validate(case), layout=...)`` is the semantic lowering boundary.
"""
from __future__ import annotations

from typing import Any

from pops.descriptors import Availability
from pops.problem.registries import (
    BlockRegistry, FieldRegistry, InitialConditionRegistry, ParamRegistry, TimeRegistry)
from pops.problem._inspection import inspect_payload, serialization_payload
from pops.problem._validation import account_block_plan_fields
from pops._report import ReportTree
from pops.model import OwnerKind, OwnerPath


class Case:
    """A typed, inert top-level assembly with one authority per declaration family.

    ``Case("plasma")`` then assembled explicitly::

        case = pops.Case("plasma")
        electron = case.block("ne", model)
        case.numerics(discretization, block=electron)
        potential = case.field(field_operator, field_discretization)
        case.program(time_program)
        case.consumers(consumer_graph)
        validated = pops.validate(case)
        resolved = pops.resolve(validated, layout=Uniform(CartesianMesh()))
        compiled = pops.compile(resolved)

    Declaration methods return stable owner-qualified handles; singleton assembly authorities such
    as :meth:`program` return the Case for optional chaining. A Case CONTAINS descriptors (the
    blocks' physics and field bindings) but is NOT itself a :class:`pops.descriptors.Descriptor`
    (Spec 5 sec.6 / sec.15). It exposes the same inspectable surface --
    ``requirements`` / ``capabilities`` / ``options`` / ``available`` / ``validate`` / ``inspect`` /
    ``lower`` -- implemented DIRECTLY here (by delegating to the registries), so it duck-types as a
    route-describing object without inheriting a descriptor identity.
    """

    category = "case"
    #: A Case names a pure-Python assembly, not a single native C++ symbol.
    native_id = None

    def __setattr__(self, name: str, value: Any) -> None:
        if name in ("_name", "_owner_path") and hasattr(self, name):
            raise AttributeError("pops.Case identity is immutable; construct a new Case")
        if getattr(self, "_frozen", False):
            if name != "_frozen" or value is not True:
                raise RuntimeError("pops.Case is frozen: cannot change %s" % name)
        object.__setattr__(self, name, value)

    def __delattr__(self, name: str) -> None:
        if name in ("_name", "_owner_path"):
            raise AttributeError("pops.Case identity is immutable; construct a new Case")
        if getattr(self, "_frozen", False):
            raise RuntimeError("pops.Case is frozen: cannot delete %s" % name)
        object.__delattr__(self, name)

    def __init__(self, name: Any = "Case") -> None:
        if not isinstance(name, str) or not name:
            raise TypeError("Case: name must be a non-empty string")
        self._name = name
        self._owner_path = OwnerPath.fresh(OwnerKind.CASE, self._name)
        # Validation freezes the assembly and commits the snapshot used by compile identity.
        self._frozen = False
        self._snapshot = None
        self._block_registry = BlockRegistry(self.owner_path)
        self._field_registry = FieldRegistry(self.owner_path)
        self._time_registry = TimeRegistry()
        self._param_registry = ParamRegistry(self.owner_path)
        self._initial_registry = InitialConditionRegistry(self.owner_path, self.resolve)
        self._numerics_assignments = {}
        self._consumer_graph = None

    @property
    def name(self) -> Any:
        return self._name

    @property
    def owner_path(self) -> OwnerPath:
        """Immutable qualified identity anchor for every handle owned by this Case."""
        return self._owner_path

    # --- freeze lifecycle (ADC-563) -----------------------------------------------------
    def _guard_mutable(self, what: Any) -> None:
        """Raise when a mutating setter runs after :meth:`freeze` (ADC-563), naming the Case."""
        if self._frozen:
            raise RuntimeError(
                "pops.Case %r is frozen: cannot %s after pops.validate accepted it. Author a "
                "fresh Case, edit it before validation, then repeat resolve/compile; a post-validation "
                "mutation cannot change a resolved plan or bound artifact." % (self._name, what))

    def freeze(self) -> Any:
        """Validate, freeze, and return the stable authoring snapshot.

        This method cannot bypass validation: it is the snapshot-returning form of the same phase
        transition as :meth:`validate`.  Resolution accepts only the successfully frozen result.
        """
        if self._frozen:
            return self._snapshot
        report = self.validate_report()
        report.raise_if_error()
        return self._commit_freeze()

    def _commit_freeze(self) -> Any:
        """Commit a graph already proven valid; internal half of the phase transition."""
        from pops.problem._snapshot import build_problem_snapshot
        # Two-phase commit: canonicalisation can call descriptor projections and therefore fail.
        # Build and validate the complete inert snapshot while every authoring object is still
        # mutable; only a successful candidate is followed by the irreversible registry/descriptor
        # seal.  A serialization error leaves the Case exactly as editable as it was before the
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
    def initials(self) -> Any:
        """The sole registry for typed initial conditions."""
        return self._initial_registry

    # --- assembly -----------------------------------------------------------------------
    def block(
        self,
        name: Any,
        model: Any,
        *,
        states: Any = None,
    ) -> Any:
        """Declare one model instance and return its stable owner-qualified block handle.

        ``block`` is the sole block-registration path. A model-local state or operator becomes
        unambiguous for consumers only after qualification through this returned handle. Numerical,
        temporal and consumer authorities are attached through their dedicated Case families.
        """
        self._guard_mutable("add a block")
        return self._block_registry.add(name, model, states=states)

    def field(self, operator: Any, discretization: Any) -> Any:
        """Register one field operator/discretization pair and return its case-owned handle."""
        self._guard_mutable("add a field")
        return self._field_registry.add(operator, discretization)

    def numerics(self, plan: Any, *, block: Any = None) -> Any:
        """Assign one :class:`DiscretizationPlan` authority to a block.

        Omitting ``block`` is unambiguous only for a single-block Case. A multi-block Case must name
        the returned BlockHandle explicitly; model names and strings never select an instance.
        """
        from pops.numerics import DiscretizationPlan

        self._guard_mutable("assign a numerical plan")
        if type(plan) is not DiscretizationPlan:
            raise TypeError("Case.numerics requires an exact DiscretizationPlan")
        handles = self._block_registry.handles()
        if block is None:
            if len(handles) != 1:
                raise ValueError(
                    "Case.numerics without block= requires exactly one block; got %d" % len(handles))
            live = next(iter(handles.values()))
        else:
            canonical = self._block_registry.canonical_block(block)
            live = self._block_registry.handle(canonical.local_id)
        if live.local_id in self._numerics_assignments:
            raise ValueError("block %r already has a DiscretizationPlan" % live.local_id)
        if self._block_registry.spec(live.local_id)["spatial"] is not None:
            raise ValueError(
                "block %r already carries a spatial method; select numerics only through "
                "DiscretizationPlan" % live.local_id)
        spec = self._block_registry.spec(live.local_id)
        plan.validate_for(spec["model"], states=spec["states"])
        self._numerics_assignments[live.local_id] = plan
        return self

    def _resolved_numerics_for(self, block_name: str) -> Any:
        plan = self._numerics_assignments.get(block_name)
        if plan is None:
            return None
        return plan.resolve_for(self, self._block_registry.handle(block_name))

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

    def consumers(self, graph: Any) -> Any:
        """Attach the sole transactional :class:`ConsumerGraph` authority."""
        from pops.runtime.consumer import ConsumerGraph

        self._guard_mutable("attach the consumer graph")
        if type(graph) is not ConsumerGraph:
            raise TypeError("Case.consumers requires an exact ConsumerGraph")
        if self._consumer_graph is not None:
            raise ValueError("the Case already has a ConsumerGraph authority")
        self._consumer_graph = graph
        return self

    def program(self, program: Any) -> Any:
        """Attach the sole whole-system :class:`pops.Program` authority."""
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
    def _consumers(self) -> Any:
        return self._consumer_graph

    @property
    def _initials(self) -> Any:
        return self._initial_registry

    @property
    def _numerics(self) -> Any:
        return dict(self._numerics_assignments)

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
    def options(self) -> Any:
        """The authoring summary and per-family counts; layout is resolved later."""
        return {"name": self._name,
                "n_blocks": len(self._block_registry), "n_fields": len(self._field_registry),
                "n_params": len(self._param_registry),
                "n_initials": len(self._initial_registry),
                "n_numerical_plans": len(self._numerics_assignments),
                "has_consumers": self._consumer_graph is not None,
                "has_time": self._time_registry.program is not None}

    def requirements(self) -> Any:
        """The route's requirements as a typed :class:`~pops.descriptors_report.RequirementSet`."""
        from pops.descriptors_report import RequirementSet
        req = RequirementSet()
        if len(self._field_registry):
            req.add("elliptic_solve")
        req.add("time_scheme")
        return req

    def capabilities(self) -> Any:
        """The route's capabilities as a typed :class:`~pops.descriptors_report.CapabilitySet`."""
        from pops.descriptors_report import CapabilitySet
        caps = {"blocks": sorted(self._block_registry.names())}
        caps["fields"] = sorted(self._field_registry.names())
        return CapabilitySet(caps)

    def available(self, context: Any = None) -> Any:
        """An EXPLAINABLE availability status, computed from the parts (no runtime)."""
        if not len(self._block_registry):
            return Availability.no("case has no block; add one with .block(name, model)",
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
        """Validate and atomically freeze the complete authoring graph.

        Runs each authoring family's own ``validate``, folds them into ONE
        :class:`~pops.ReportTree`, and raises (via ``raise_if_error``)
        when any error accumulated.  Success seals the Case before returning, making this the
        sole transition from mutable authoring into the validated phase; resolution refuses an
        unfrozen Case and never repairs or mutates it.
        """
        report = self.validate_report(context)
        report.raise_if_error()
        if not self._frozen:
            self._commit_freeze()
        return True

    def validate_report(self, context: Any = None) -> Any:
        """Aggregate the per-family validation reports into ONE report (no raise; ADC-553).

        Layout-independent registry checks run here. Layout/provider compatibility is proved by
        ``pops.resolve(validated_case, layout=...)``, where that separate authority is finally known.
        """
        report = ReportTree(
            phase="validation", severity="info", code="validation.problem.root",
            source="problem", owner=self.owner_path)
        report = report.extend(self._block_registry.validate(context))
        report = account_block_plan_fields(report, self._block_registry)
        # Carry the mesh layout into each field problem's validation so its solver can refuse a
        # layout it cannot serve (Spec 6 sec.8/9), precisely, before any compile.
        report = report.extend(
            self._field_registry.validate(self._field_validation_context(context)))
        report = report.extend(self._param_registry.validate(context))
        report = report.extend(self._initial_registry.validate(context))
        if self._consumer_graph is not None:
            try:
                self._consumer_graph.validate_references(self.resolve)
            except Exception as exc:  # noqa: BLE001 -- aggregate exact consumer refusal
                report = report.error(
                    "consumer", "invalid_consumer_graph", str(exc))
        for block_name in sorted(self._block_registry.names()):
            spec = self._block_registry.spec(block_name)
            model = spec["model"]
            rate_contracts = getattr(model, "_rate_contracts", {})
            selected_rates = tuple(
                rate for rate, contract in rate_contracts.items()
                if contract["state"] in spec["states"])
            if selected_rates and block_name not in self._numerics_assignments:
                report = report.error(
                    "numerics", "missing_discretization_plan",
                    "block %r has no DiscretizationPlan; evolved rates never receive a "
                    "scientific fallback" % block_name,
                    context={"block": block_name})
        for block_name, plan in sorted(self._numerics_assignments.items()):
            try:
                spec = self._block_registry.spec(block_name)
                plan.validate_for(spec["model"], states=spec["states"])
                self._resolved_numerics_for(block_name)
            except Exception as exc:  # noqa: BLE001 -- aggregate exact numerical refusal
                report = report.error(
                    "numerics", "invalid_discretization_plan", str(exc),
                    context={"block": block_name})
        return report

    def _field_validation_context(self, context: Any) -> Any:
        """The validation context handed to each field problem, carrying the mesh layout."""
        merged: dict[str, Any] = dict(context) if isinstance(context, dict) else {}
        merged["layout"] = None
        merged["declaration_resolver"] = self.resolve
        return merged

    def layout_subjects(self) -> Any:
        """Return the immutable exact block/state/field snapshot assigned by a LayoutPlan."""
        from pops.problem.layout_subjects import LayoutSubjects
        from pops.problem._layout_protocol import materialized_layout_subjects
        return LayoutSubjects(**materialized_layout_subjects(self))

    def explain_routes(self) -> Any:
        """Return a printable route matrix sourced from the C++ authoritative facts (sec.13.12.1)."""
        from pops._capabilities import native_capability_matrix
        return native_capability_matrix(owner=self.name, layout="resolve-time",
                                        target="module")

    def lower(self, context: Any = None) -> Any:
        """The inert lowering record for the assembly (metadata only; no computation)."""
        from pops.descriptors_report import LoweredDescriptor
        return LoweredDescriptor(name=self._name, category=self.category,
                                 native_id=self.native_id, options=self.options())

    def inspect(self) -> Any:
        """A typed :class:`~pops.problem.report_view.CaseReport` of the assembly (ADC-564).

        Attributes + ``to_dict()`` (never a dict subclass), carrying the name / blocks / fields /
        params / consumers / requirements / capabilities. Inert: no build, no
        compile, no validation. ``pops.inspect(problem)`` is the explicit dict bridge over its
        ``to_dict()``.
        """
        from pops.problem.report_view import CaseReport
        return CaseReport(self._inspect_payload())

    def _inspect_payload(self) -> Any:
        """The ordered inspection dict (the historical inspect() shape) the CaseReport wraps."""
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
        return ("%s [%s] | blocks=%d | fields=%d | params=%d | consumers=%s | time=%s"
                % (self._name, self.category,
                   len(self._block_registry), len(self._field_registry),
                   len(self._param_registry), "set" if self._consumer_graph is not None else "none",
                   "set" if self._time_registry.program is not None else "none"))

    def __repr__(self) -> str:
        return ("Case(name=%r, blocks=%s, fields=%s)"
                % (self._name,
                   sorted(self._block_registry.names()), sorted(self._field_registry.names())))


__all__ = ["Case"]
