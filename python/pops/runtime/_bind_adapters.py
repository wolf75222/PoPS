"""Internal runtime adapters for ``pops.bind`` (ADC-583).

This is the LOWERING-BINDING layer between the public ``pops.bind`` entry point and the internal
C++-backed runtime engines (:class:`pops.runtime.system.System` /
:class:`pops.runtime.amr_system.AmrSystem`). It hides the legacy engine setters
(``add_block`` / ``add_equation`` / ``set_poisson`` / ``set_refinement`` / ``install_program`` /
...) from the user: ``pops.bind`` builds one of these adapters, the adapter constructs the engine,
lowers the validated Problem objects onto it, installs through the INTERNAL ``_install_compiled`` seam
and returns a :class:`pops.runtime._bound_sim.BoundSimulation` view -- never the raw engine.

The Problem's ``layout`` descriptor is the selector carried by ``pops.compile``:
``layout=Uniform(...)`` selects :class:`_UniformRuntimeAdapter` (target ``"system"``),
``layout=AMR(...)`` selects :class:`_AmrRuntimeAdapter` (target ``"amr_system"``). Both share the
bind logic in :class:`_RuntimeAdapter`; only ``build_engine`` / ``install`` differ (the AMR adapter
derives the ``AmrSystemConfig`` from the layout, flows the typed refinement, installs the native
per-block loaders and -- when the handle carries a whole-system compiled time Program (ADC-634) --
that Program on the hierarchy via ``install_program``).

The engines stay first-class C++ runtimes reachable for low-level / internal tests (they are still
importable from :mod:`pops.runtime.system`); they are simply no longer the recommended route. This
module lives in the ``runtime`` layer, which may import ``mesh`` / ``codegen`` / ``_pops``; the
mesh / ``_bootstrap`` / engine imports are kept LAZY (function scope) to mirror the existing bind
path and keep the import-graph architecture gates green.
"""
from __future__ import annotations

from typing import Any

from pops.runtime._amr_bind_lowering import amr_config_from_layout as _amr_config_from_layout


class _RuntimeAdapter:
    """Shared bind logic of the Uniform / AMR runtime adapters.

    A concrete adapter supplies ``build_engine`` (construct the internal engine) and ``install``
    (lower the validated instances/params/aux/solvers/cadence/outputs onto that engine's
    ``_install_compiled`` seam). :meth:`build` runs the shared sequence -- build the engine, install,
    wrap in the :class:`BoundSimulation` view -- so Uniform and AMR share it byte-for-byte.

    The adapter calls the engine with VALIDATED / LOWERED objects (typed refinement, resolved field
    solvers, per-instance state mappings), never with raw user strings.
    """

    #: The compile target this adapter serves (``"system"`` / ``"amr_system"``). Overridden.
    target = None

    def build_engine(self, layout: Any) -> Any:
        """Construct and return the internal engine (System / AmrSystem). Overridden per adapter."""
        raise NotImplementedError

    def install(self, engine, compiled, *, instances, params, aux, solvers, cadence, outputs,
                diagnostics):
        """Lower the bind inputs onto @p engine's internal ``_install_compiled`` seam. Overridden."""
        raise NotImplementedError

    def build(self, compiled, *, layout, instances, params, aux, solvers, cadence, outputs,
              diagnostics=()):
        """Build the engine, install the compiled Problem onto it, wrap it in a bound-simulation view.

        This is the one place Uniform and AMR share: the adapter-specific ``build_engine`` /
        ``install`` are the only branch points. Returns a :class:`BoundSimulation` -- the delegating
        view users obtain from ``pops.bind`` -- NOT the raw engine.
        """
        from pops.runtime._bound_sim import BoundSimulation

        engine = self.build_engine(layout)
        self.install(engine, compiled, instances=instances, params=params, aux=aux,
                     solvers=solvers, cadence=cadence, outputs=outputs, diagnostics=diagnostics)
        return BoundSimulation(engine)
    def build_install_plan(self, install_plan: Any) -> Any:
        from pops.codegen._plans import require_install_plan
        from pops.runtime._bound_sim import BoundSimulation
        plan = require_install_plan(install_plan)
        if plan.target != self.target:
            raise ValueError(
                "pops.bind: InstallPlan target %r cannot be consumed by the %r adapter"
                % (plan.target, self.target)
            )
        unsupported = sorted(set(plan.resources) - {"execution_context"})
        if unsupported:
            raise NotImplementedError(
                "pops.bind: InstallPlan carries runtime resources %s, but this adapter has no "
                "typed native resource consumer" % unsupported
            )
        engine = self.build_engine_for_plan(plan)
        engine._execution_context = plan.execution_context
        self.install_plan(engine, plan)
        return BoundSimulation(engine)

    def build_engine_for_plan(self, install_plan: Any) -> Any:
        return self.build_engine(install_plan.layout)
    def install_plan(self, engine: Any, install_plan: Any) -> None:
        raise NotImplementedError


class _UniformRuntimeAdapter(_RuntimeAdapter):
    """Bind adapter for ``layout=Uniform(...)`` (the single-level ``System`` engine).

    Builds a :class:`pops.runtime.system.System` and installs the whole-system compiled time
    ``Program`` through its ``_install_compiled(compiled, instances=, ...)`` seam.
    """

    target = "system"

    def build_engine(self, layout: Any) -> Any:
        # Resolved from pops.runtime.system at call time so a monkeypatched System (low-level tests)
        # still takes effect. The Uniform layout carries the single-level mesh: derive the System's
        # SystemConfig (n / L / periodic) from it so the engine matches the Problem's grid, mirroring the
        # AMR adapter. A handle with no layout (not produced by pops.compile) binds on the System()
        # defaults.
        from pops.runtime.system import System

        if layout is None:
            return System()
        return System(_system_config_from_layout(layout))

    def install(self, engine, compiled, *, instances, params, aux, solvers, cadence, outputs,
                diagnostics):
        engine._install_compiled(compiled, instances=instances, params=params, aux=aux,
                                 solvers=solvers, cadence=cadence, outputs=outputs,
                                 diagnostics=diagnostics)

    def install_plan(self, engine: Any, install_plan: Any) -> None:
        artifact = install_plan.artifact
        if artifact.program is None:
            raise ValueError("pops.bind: system InstallPlan has no compiled program")
        resolved = artifact.plan
        engine._install_compiled(
            artifact, instances=install_plan.instances,
            params=install_plan.params,
            aux=install_plan.aux,
            field_plans=resolved.field_plans,
            cadence=None,
            outputs=resolved.outputs,
            diagnostics=resolved.diagnostics,
        )


class _AmrRuntimeAdapter(_RuntimeAdapter):
    """Bind adapter for ``layout=AMR(...)`` (the refined ``AmrSystem`` engine).

    Builds a :class:`pops.runtime.amr_system.AmrSystem` from an ``AmrSystemConfig`` DERIVED from the
    AMR layout (:func:`_amr_config_from_layout`), flows the typed refinement onto it
    (:func:`_flow_amr_layout`) BEFORE the blocks are installed, then installs through the
    ``_install_compiled`` seam. Each instance carries its own ``target='amr_system'``
    ``CompiledModel``. When the handle carries a whole-system compiled time ``Program``
    (``compiled.program is not None``, ADC-634) it is passed through so
    ``AmrSystem._install_compiled`` -> ``_finish_program_install`` installs it on the hierarchy
    (``install_program`` + ``set_program_params`` + ``set_program_cadence``); a native per-block
    handle passes ``compiled=None``.
    """

    target = "amr_system"

    def __init__(self, n_blocks: Any = 1) -> None:
        # The declared block count decides the single- vs multi-block refinement wiring.
        self._n_blocks = n_blocks

    def build_engine(self, layout: Any) -> Any:
        # Resolved from pops.runtime.system at call time (mirrors the Uniform adapter) so a
        # monkeypatched AmrSystem is honored. A missing layout on an AMR target is a bind bug.
        from pops.runtime.system import AmrSystem

        if layout is None:
            raise TypeError(
                "pops.bind: an AMR target carries no layout descriptor; the compiled handle must "
                "come from pops.compile(problem_with_AMR_layout, ...)")
        self._layout = layout
        return AmrSystem(_amr_config_from_layout(layout))

    def build_engine_for_plan(self, install_plan: Any) -> Any:
        from pops.runtime.system import AmrSystem

        self._layout = install_plan.layout
        return AmrSystem(
            _amr_config_from_layout(
                install_plan.layout,
                hierarchy=install_plan.resolved_hierarchy,
            )
        )

    def install(self, engine, compiled, *, instances, params, aux, solvers, cadence, outputs,
                diagnostics):
        # A whole-system compiled time Program (compiled.program is not None, ADC-634) installs on the
        # AMR hierarchy via AmrSystem._install_compiled -> _finish_program_install -> install_program
        # (the ADC-508 per-level driver); a native per-block handle (compiled.program is None) installs
        # with compiled=None, each instance carrying its OWN target='amr_system' CompiledModel wired
        # with add_equation -> add_native_block. Discriminate by the duck-typed getattr signal, exactly
        # like System._install_compiled -- never by the handle class. A native CompiledModel has no
        # .program attribute, so getattr(..., None) selects compiled=None for it.
        plan = getattr(compiled, "install_plan", None)
        program = compiled if getattr(plan, "has_program", False) else None
        schema = getattr(plan, "bind_schema", None)
        _flow_amr_layout(
            engine,
            self._layout,
            n_blocks=self._n_blocks,
            bind_schema=schema,
            params=params,
        )
        engine._install_compiled(compiled=program, instances=instances, params=params, aux=aux,
                                 solvers=solvers, cadence=cadence, outputs=outputs,
                                 diagnostics=diagnostics,
                                 bind_schema=schema)

    def install_plan(self, engine: Any, install_plan: Any) -> None:
        artifact = install_plan.artifact
        if artifact.program is None:
            raise NotImplementedError(
                "pops.bind: [amr:typed-native-install unavailable] an AMR artifact without a "
                "whole-system Program cannot yet preserve the full CompiledSimulationArtifact "
                "identity through native installation; compile with a time Program"
            )
        resolved = artifact.plan
        schema = artifact.bind_schema
        by_id = {
            handle.qualified_id: value
            for handle, value in install_plan.initial_values.items()
        }
        initial_rows = []
        if install_plan.initial_condition_plan is not None:
            from pops.mesh.amr import AnalyticReprojection
            selections = {
                row.subject.qualified_id: row.method
                for row in install_plan.bootstrap_plan.selections
            }
            physical = {
                requirement.subject.qualified_id: requirement
                for entry in install_plan.amr_transfer.entries
                for requirement in entry.requirements
                if requirement.materialization == "physical"
            }
            for binding in install_plan.initial_condition_plan.bindings:
                subject = binding.subject
                if subject.kind == "particle":
                    raise NotImplementedError(
                        "pops.bind: particle/hybrid particle-grid is outside this AMR target"
                    )
                if subject.kind != "state":
                    raise ValueError(
                        "pops.bind: AMR initial values must target state/particle Handles"
                    )
                requirement = physical[subject.qualified_id]
                key = requirement.key.to_data()
                space = key["space"]["name"]
                centering = key["centering"]["name"]
                block = subject.block_ref.local_id if subject.block_ref is not None else None
                analytic = type(selections[subject.qualified_id]) is AnalyticReprojection
                initial_rows.append(
                    (
                        subject.qualified_id,
                        block,
                        by_id.get(subject.qualified_id),
                        space,
                        centering,
                        "analytic" if analytic else "prolong",
                        binding.source.options.to_data(),
                    )
                )
        if install_plan.bootstrap_plan is None:
            _flow_amr_layout(
                engine,
                self._layout,
                n_blocks=len(install_plan.instances),
                bind_schema=schema,
                params=install_plan.params,
            )
        else:
            _flow_bootstrap_tagging(
                engine, install_plan.bootstrap_plan, install_plan.params
            )
        engine._install_compiled(
            compiled=artifact, instances=install_plan.instances,
            params=install_plan.params,
            aux=install_plan.aux,
            field_plans=resolved.field_plans,
            cadence=None,
            outputs=resolved.outputs,
            diagnostics=resolved.diagnostics,
            bind_schema=schema,
            initial_values=tuple(initial_rows),
            bootstrap_plan=install_plan.bootstrap_plan,
            amr_transfer=install_plan.amr_transfer,
        )


def adapter_for(target: Any, layout: Any, n_blocks: Any = 1) -> Any:
    """Select the runtime adapter for a compiled Problem.

    ``layout=Uniform(...)`` compiles to ``target='system'`` and selects the Uniform adapter;
    ``layout=AMR(...)`` compiles to ``target='amr_system'`` and selects the AMR adapter. The Problem's
    layout descriptor (carried on the handle by :func:`pops.compile`) is the selector: this function
    maps the target the layout produced onto the matching adapter.

    Args:
        target: The compile target carried on the handle (``"system"`` / ``"amr_system"``).
        layout: The AMR layout descriptor (required for ``"amr_system"``, ignored for ``"system"``).
        n_blocks: The declared block count (AMR only), deciding the single- vs multi-block
            refinement wiring.

    Returns:
        A :class:`_RuntimeAdapter` (Uniform or AMR) ready to :meth:`_RuntimeAdapter.build`.
    """
    if target == "amr_system":
        # An AMR target with no carried layout descriptor is a compile/bind mismatch; keep the
        # existing clear message (the adapter re-checks it at build_engine too).
        if layout is None:
            raise TypeError(
                "pops.bind: an AMR target carries no layout descriptor; the compiled handle must "
                "come from pops.compile(problem_with_AMR_layout, ...)")
        return _AmrRuntimeAdapter(n_blocks=n_blocks)
    if target == "system":
        return _UniformRuntimeAdapter()
    raise ValueError(
        "pops.bind: InstallPlan target must be exactly 'system' or 'amr_system'; "
        "got %r" % (target,)
    )


def install_plan(install_plan: Any) -> Any:
    from pops.codegen._plans import require_install_plan
    from pops.runtime._bind_validation import run_bind_gates
    plan = require_install_plan(install_plan)
    artifact = plan.artifact
    inputs = plan.bind_inputs
    if plan.target == "amr_system" and artifact.program is None:
        raise NotImplementedError(
            "pops.bind: [amr:typed-native-install unavailable] an AMR artifact without a "
            "whole-system Program cannot yet preserve the full CompiledSimulationArtifact "
            "identity through native installation; compile with a time Program"
        )
    run_bind_gates(
        artifact,
        plan.layout,
        inputs.initial_state,
        plan.params,
        plan.aux,
        platform_manifest=artifact.platform_manifest, execution_context=plan.execution_context,
    )
    adapter = adapter_for(plan.target, plan.layout, n_blocks=len(plan.instances))
    return adapter.build_install_plan(plan)


__all__ = ["adapter_for", "install_plan"]


# --- Mesh layout lowering (engine config + AMR refinement seams) --------------------------------
# These lower an inert mesh layout onto the C++ engine config / refinement seams; that is
# runtime-adapter responsibility, so they live here (not in the codegen layer). The AMR helpers
# were moved verbatim from codegen.orchestration (behavior byte-identical); their error messages
# already speak the "pops.bind:" vocabulary and are preserved verbatim.


def _system_config_from_layout(layout: Any) -> Any:
    """Build a ``SystemConfig`` from a :class:`pops.mesh.layouts.Uniform` descriptor.

    Maps the inert Uniform layout onto the C++ runtime config the ``System`` constructor consumes:
    ``n`` / ``L`` / ``periodic`` from the single-level ``CartesianMesh`` (``layout.mesh``), mirroring
    :func:`_amr_config_from_layout` for the AMR route so the bound engine matches the Problem's grid
    instead of the ``SystemConfig`` defaults. Imported lazily so the runtime module stays
    import-light.
    """
    from pops._bootstrap import SystemConfig

    mesh = layout.mesh
    cfg = SystemConfig()
    cfg.n = int(mesh.n)
    cfg.L = float(mesh.L)
    cfg.periodic = bool(mesh.periodic)
    return cfg


def _flow_amr_layout(sim: Any, layout: Any, n_blocks: Any = 1, *,
                     bind_schema: Any = None, params: Any = None) -> Any:
    """Flow the AMR layout's typed refinement criterion onto @p sim BEFORE the blocks are installed.

    Mirrors the old string path: a ``Refine.on(subject).above(threshold)`` (or a ``TagUnion`` of
    them) becomes ``set_refinement(threshold, ...)`` and a ``gradient``-predicate on the potential
    becomes ``set_phi_refinement(threshold)``. The field problem (Poisson) is flowed through the
    unified ``install(solvers=...)`` seam (``AmrSystem._install_solver`` -> ``set_poisson``), which
    runs its field solvers BEFORE adding the blocks, so it is not duplicated here.

    @p n_blocks is the declared block count: the per-block variable / role selector is only wired in
    MULTI-BLOCK (>= 2 blocks, the union-of-tags runtime engine), so a single-block Problem keeps the
    component-0-only behaviour and a multi-block Problem forwards a non-density subject to
    ``set_refinement(threshold, variable=)`` (the C++ AmrSystem resolves it per block; a compiled
    block refuses a non-default selector, the honest native boundary).
    """
    criterion = getattr(layout, "refine", None)
    if criterion is not None:
        _apply_refine_criterion(
            sim,
            criterion,
            is_multiblock=n_blocks > 1,
            bind_schema=bind_schema,
            params=params,
        )


def _flow_bootstrap_tagging(sim: Any, bootstrap: Any, params: Any) -> None:
    """Lower the resolved owner-qualified threshold indicator without name heuristics."""
    from pops.mesh.amr import Above

    graph = bootstrap.tagging.graph
    if type(graph.refine) is not Above or graph.coarsen is not None:
        raise NotImplementedError(
            "pops.bind: native bootstrap currently lowers Above(density, threshold) "
            "without a coarsen root"
        )
    predicate = graph.refine
    if predicate.threshold not in params:
        raise ValueError("pops.bind: bootstrap tagging threshold is missing from resolved params")
    # The native runtime resolves the exact variable against the selected block descriptor.  Never
    # infer component zero from spelling (rho/n/ne) and never maintain a variable-name whitelist.
    if predicate.indicator.block_ref is None:
        raise ValueError("pops.bind: bootstrap tag indicator must be block-qualified")
    sim._set_bootstrap_refinement(
        predicate.indicator.block_ref.local_id,
        predicate.indicator.local_id,
        float(params[predicate.threshold]),
        bootstrap.tagging.qualified_id,
    )


def _apply_refine_criterion(sim: Any, criterion: Any, is_multiblock: bool = False, *,
                            bind_schema: Any = None, params: Any = None) -> Any:
    """Lower one typed refinement criterion to set_refinement / set_phi_refinement on @p sim.

    A ``Refine`` whose predicate is a gradient on the potential (``phi`` / ``grad phi``) lowers to
    ``set_phi_refinement(threshold)``; a density subject lowers to ``set_refinement(threshold)`` on
    component 0. A non-density subject lowers to ``set_refinement(threshold, variable=subject)`` when
    @p is_multiblock (the union-of-tags engine resolves the selector per block); in single-block it is
    refused (the AmrCouplerMP path refines on component 0 only). A ``TagUnion`` lowers to each call in
    turn. A criterion that is neither raises a clear error rather than silently dropping it."""
    from pops.mesh.amr import Refine, TagUnion

    if isinstance(criterion, TagUnion):
        for c in criterion.criteria:
            _apply_refine_criterion(
                sim,
                c,
                is_multiblock=is_multiblock,
                bind_schema=bind_schema,
                params=params,
            )
        return
    if not isinstance(criterion, Refine):
        raise TypeError(
            "pops.bind: AMR refine criterion must be a pops.mesh.amr.Refine / TagUnion (got %r)"
            % type(criterion).__name__)
    if not getattr(criterion, "references_authenticated", False):
        raise ValueError(
            "pops.bind: Refine criterion references were not authenticated by Problem.resolve; "
            "run it through pops.compile(problem, layout=...) instead of attaching a raw or "
            "canonical-looking Handle directly to a compiled/runtime layout")
    threshold = criterion.threshold
    if threshold is None:
        raise ValueError("pops.bind: Refine criterion has no threshold "
                         "(use Refine.on(subject).above(value))")
    threshold = _refine_threshold_value(threshold, bind_schema, params)
    from pops.model import Handle
    if not isinstance(criterion.subject, Handle):
        raise NotImplementedError(
            "pops.bind: [amr:expression_indicator unavailable] Refine subject %s is a semantic "
            "indicator expression. Its Handle leaves were validated and resolved at compile, but "
            "the current native AMR runtime only lowers direct declaration Handle selectors and "
            "the dedicated potential-gradient predicate. Add the expression-indicator backend "
            "capability before running this criterion; it is never flattened to a variable name."
            % type(criterion.subject).__name__)
    subject = _refine_subject_name(criterion.subject)
    # The potential-gradient tag (|grad phi| > threshold) is the AMR-specific ring-edge criterion.
    if criterion.predicate == "gradient_above" and subject in ("phi", "grad phi", "potential"):
        sim.set_phi_refinement(float(threshold))
        return
    # A density subject lowers to set_refinement(threshold) on component 0 (no selector).
    if _is_default_density_subject(subject):
        sim.set_refinement(float(threshold))
        return
    # A non-density subject is a per-block variable / role selector. It is only wired in MULTI-BLOCK
    # (the union-of-tags runtime engine); the single-block AmrCouplerMP path refines on component 0
    # only, so a selector there is refused with a clear message rather than silently dropped.
    if not is_multiblock:
        raise NotImplementedError(
            "pops.bind: refining on %r is a multi-block AMR feature; the single-block AMR route "
            "refines on the density (component 0) only. Refine on the density "
            "(Refine.on(Density).above(...)), or use the |grad phi| tag "
            "(Refine.on(phi).gradient_above(...))." % (subject,))
    # Forward the subject as the per-block variable name. The C++ AmrSystem::set_refinement resolves
    # it against each block's conserved variables (a native block) or refuses it (a compiled .so
    # block: component 0 only) -- the honest native boundary, not a silent drop here.
    sim.set_refinement(float(threshold), variable=subject)


def _refine_threshold_value(threshold: Any, schema: Any, params: Any) -> Any:
    """Resolve one canonical parameter threshold from the effective bind mapping."""
    from pops.ir import ValueExpr
    from pops.model import ParamHandle

    handle = threshold.handle if isinstance(threshold, ValueExpr) else threshold
    if not isinstance(handle, ParamHandle):
        return threshold
    if schema is None:
        raise ValueError("pops.bind: parameterized AMR threshold requires BindSchema")
    slot = schema.slot(handle)
    if slot.handle not in (params or {}):
        raise ValueError(
            "pops.bind: resolved params are missing AMR threshold %s" % slot.qid
        )
    return params[slot.handle]


def _refine_subject_name(subject: Any) -> Any:
    """Lower one canonical Handle to the native variable token at the runtime boundary."""
    from pops.model import Handle

    if not isinstance(subject, Handle):
        raise TypeError(
            "pops.bind: Refine subject must be a resolved pops.model.Handle, got %r; strings "
            "are not declaration identities" % type(subject).__name__)
    if not subject.is_resolved:
        raise ValueError(
            "pops.bind: Refine subject %s is still authoring-owned; compile must resolve every "
            "reference through Problem.resolve before runtime lowering" % subject.qualified_id)
    return subject.local_id


def _is_default_density_subject(subject: Any) -> Any:
    """True when a Refine subject names the density / component 0 (the single-block default).

    The native single-block AMR refines on component 0 (the historical density), so a Density-role
    or density-named subject maps to ``set_refinement(threshold)`` with no selector; ``None`` (an
    unnamed subject) is treated as the default too. Any other name is a non-default selector that
    the single-block route cannot honor."""
    if subject is None:
        return True
    return subject in ("Density", "density", "rho", "n", "ne")
