"""Public ``pops.compile`` / ``pops.bind`` snapshot boundary.

``compile`` resolves the mutable authoring graph exactly once, captures a complete
``AuthoringSnapshot`` (including explicit layout/time/libraries), and creates a short-lived
``ResolvedPlan``. Every block loader is built before the returned artifact is sealed. The artifact
retains only its snapshot, immutable metadata and ``InstallPlan``.

``bind`` consumes that ``InstallPlan`` plus runtime values. It never receives a ``Problem`` and has
no reconstruction/fallback path into model, registry or descriptor builders. Runtime imports remain
lazy so the codegen module-scope dependency graph stays acyclic.
"""

from __future__ import annotations

from typing import Any

from pops.codegen._orchestration_compile import (
    attach_install_plan as _attach_install_plan,
    attach_problem_snapshot as _attach_problem_snapshot,
    capture_field_solvers as _capture_field_solvers,
    capture_runtime_declarations as _capture_runtime_declarations,
    compile_install_models as _compile_install_models,
    prepare_problem_snapshot as _prepare_problem_snapshot,
    resolve_compile_libraries as _resolve_compile_libraries,
)


def compile(problem: Any, layout: Any = None, backend: Any = None, time: Any = None,
            **kwargs: Any) -> Any:
    """Compile one :class:`pops.problem.Problem` through an immutable resolved plan.

    The layout selects the target (``Uniform`` -> ``system``; ``AMR`` -> ``amr_system``). Uniform
    requires an explicit whole-system Program. AMR compiles the Program when present and otherwise
    uses its native per-block time policies. Both routes compile one runtime-ready model per block
    and return an artifact whose ``install_plan`` is the sole bind authority.

    Args:
        problem: The :class:`pops.problem.Problem` assembly to lower.
        layout: The mesh layout (``Uniform`` / ``AMR``) to compile for (ADC-526). Required unless the
            Problem carries a constructor layout (back-compat); when both are given they must agree (a
            mismatch raises), so a compiled artifact is frozen to exactly one layout.
        backend: The typed codegen backend descriptor; defaults to ``Production()`` (ADC-545: the
            backend string was removed -- a bare ``backend="production"`` raises ``TypeError``). It
            lowers to the same token the driver always consumed, so the artifact is byte-identical.
            The AMR route requires ``Production()`` (``Model.compile`` enforces it).
        time: The ``pops.time.Program`` time scheme; falls back to ``problem._time``. On the Uniform
            route a missing scheme raises. On the AMR route it is OPTIONAL: given, the whole-system
            Program is compiled for ``target='amr_system'`` and installed on the hierarchy (ADC-634);
            omitted, each native block carries its own time policy.
        **kwargs: Extra keyword args forwarded verbatim to the compile driver (so_path / force / cxx /
            include / std / debug / libraries on the Uniform route and the AMR whole-system Program;
            include / cxx / std on the AMR per-block loaders).

    Returns:
        A sealed :class:`pops.CompiledArtifact` carrying ``authoring_snapshot`` and ``install_plan``.
    """
    # Lazy imports keep the codegen layer's module-scope import graph clean (no mesh / runtime).
    from pops.mesh.layouts import AMR
    from pops.codegen.backends import Production, _Backend

    # ADC-545: backend is a TYPED descriptor (None -> Production(), byte-identical lowering to the
    # same "production" token). A bare string is refused; the public string form was removed. The
    # descriptor is LOWERED to its canonical token right here: the internal drivers (Model.compile,
    # compile_problem) speak tokens, and the per-block dsl.compiled record keeps its historical
    # ("production", target) shape.
    if backend is None:
        backend = Production()
    elif not isinstance(backend, _Backend):
        raise TypeError(
            "pops.compile: backend must be a typed pops.codegen backend descriptor "
            "(pops.codegen.Production() -- the default), not %r" % (backend,))
    from pops.codegen.backends import lower_backend
    backend = lower_backend(backend)

    # ADC-526: resolve + validate the effective layout FIRST (the Problem no longer carries a
    # mandatory layout); a missing / disagreeing layout, or an AMR envelope / Uniform-with-AMR-tags
    # violation, is refused here before any compile. Then the structural registry checks.
    resolved_layout = _resolve_layout(problem, layout)
    _validate_layout_for_compile(problem, resolved_layout)
    problem.validate()
    is_amr = isinstance(resolved_layout, AMR)
    target = "amr_system" if is_amr else "system"

    # Resolve the EFFECTIVE time scheme BEFORE the branch so the AMR route can read it too (ADC-634):
    # explicit @p time wins, else the Problem's recorded scheme. The Uniform route still REQUIRES one
    # (raise below); the AMR route treats it as optional (a whole-system Program to install on the
    # hierarchy when present, else the native per-block time policy).
    eff_time = time if time is not None else problem._time

    if eff_time is not None:
        from pops.time import Program
        if not isinstance(eff_time, Program):
            raise TypeError(
                "pops.compile: time must be a pops.Program (pops.lib.time presets must build and "
                "return one), not %s" % type(eff_time).__name__
            )

    if not is_amr and eff_time is None:
        raise NotImplementedError(
            "pops.compile: a time scheme is required; pass time=pops.time.Program(...) or set "
            "it on the problem with problem.time(...). There is no default time scheme.")
    if "problem_snapshot" in kwargs:
        raise TypeError(
            "pops.compile computes and freezes its own AuthoringSnapshot; problem_snapshot= is only "
            "accepted by the advanced pops.codegen.compile_problem driver")
    # Resolve every reference and detach every runtime descriptor *before* freeze.  From this point
    # onward compiler helpers receive only ResolvedPlan; no function retains or re-reads Problem.
    from pops.model.bind_schema import BindSchema
    bind_schema = BindSchema.from_problem(problem)
    from pops.problem._detached import detached_frozen
    from pops.codegen._plans import ResolvedBlock, ResolvedPlan

    detached_layout = detached_frozen(resolved_layout)
    resolved_blocks = tuple(
        ResolvedBlock(
            name=name,
            model=_resolve_problem_model(spec["model"]),
            spatial=detached_frozen(spec["spatial"]),
        )
        for name, spec in problem._blocks.items()
    )
    field_solvers = _capture_field_solvers(problem, detached_frozen)
    outputs, diagnostics = _capture_runtime_declarations(problem, detached_frozen)
    libraries, library_snapshot_values = _resolve_compile_libraries(
        tuple(kwargs.get("libraries", ()) or ()))
    if "libraries" in kwargs:
        kwargs = dict(kwargs)
        kwargs["libraries"] = libraries
    problem_snapshot = _prepare_problem_snapshot(
        problem, eff_time, layout=detached_layout, libraries=library_snapshot_values)
    plan = ResolvedPlan(
        snapshot=problem_snapshot,
        target=target,
        layout=detached_layout,
        time=eff_time,
        blocks=resolved_blocks,
        bind_schema=bind_schema,
        field_solvers=field_solvers,
        outputs=outputs,
        diagnostics=diagnostics,
        libraries=libraries,
    )

    if is_amr:
        # AMR route (single AND multi block): each block lowers to a target='amr_system' production
        # CompiledModel installed via the NATIVE add_native_block path at bind. When eff_time is a
        # whole-system Program, _compile_amr ALSO compiles it with compile_problem(target='amr_system')
        # and returns the CompiledProblem (ADC-634: AmrSystem.install_program drives it per level);
        # without one the byte-identical no-Program path returns the first block's CompiledModel. The
        # AMR branch is split into ``_orchestration_amr`` (ADC-550); import it lazily to keep the
        # module load order acyclic.
        from pops.codegen._orchestration_amr import _compile_amr
        return _compile_amr(plan, backend, **kwargs)

    install_models = _compile_install_models(plan, backend, kwargs)

    from pops.codegen.compile_drivers import compile_problem
    compiled = compile_problem(
        time=plan.time, model=plan.first_model, backend=backend, target=plan.target,
        problem_snapshot=problem_snapshot, **kwargs)
    _attach_install_plan(compiled, plan, install_models, has_program=True)
    _attach_problem_snapshot(compiled, problem_snapshot)
    return compiled


def bind(compiled: Any, *, initial_state: Any = None, state: Any = None, params: Any = None,
         aux: Any = None, solvers: Any = None, cadence: Any = None) -> Any:
    """Wire a compiled handle onto the runtime: the PUBLIC bind entry point.

    ``pops.bind`` is THE documented way to instantiate a runnable simulation from a compiled handle
    (``compiled = pops.compile(...)``). It builds the per-instance state mapping from the problem's
    blocks and the supplied initial state, derives the field solvers from the problem's field
    problems (an explicit @p solvers overrides), flows the Problem's output / checkpoint policies
    (C4 / ADC-509) so the bound sim's ``run(output_dir=...)`` fires them at each policy cadence, then
    delegates to an internal RUNTIME ADAPTER selected from the carried target
    (:func:`pops.runtime._bind_adapters.adapter_for`): ``layout=Uniform`` -> the Uniform adapter
    (``System``, whole-system compiled time Program), ``layout=AMR`` -> the AMR adapter
    (``AmrSystem`` derived from the layout, native per-block install). The adapter builds the internal
    engine, lowers the validated objects onto its INTERNAL ``_install_compiled`` seam and returns a
    ``BoundSimulation`` VIEW over that engine. The engines stay internal backends: the returned view
    exposes the run / data / diagnostic / io surface and hides the assembly setters
    (``add_block`` / ``set_poisson`` / ``set_refinement`` / ``install_program`` / ...). Call
    ``sim.run(...)`` to advance it.

    Args:
        compiled: A sealed artifact returned by :func:`compile`.
        initial_state: dict {block_name: array} of per-block initial state (alias: @p state).
        state: Alias for @p initial_state (only one may be given).
        params: dict {ParamHandle: value} of runtime parameter overrides. Model parameters must be
            block-qualified (``block[param]``); case-owned handles are already unambiguous.
        aux: dict {aux_name: array} of static aux inputs.
        solvers: dict {field: solver} overriding the per-field solvers from the problem.
        cadence: optional ``pops.CompiledTime`` macro-step cadence.

    Returns:
        A ``BoundSimulation`` view over the internal ``System`` / ``AmrSystem`` engine.
    """
    so_path = getattr(compiled, "so_path", None)
    if so_path is None:
        raise TypeError(
            "pops.bind: expected a compiled handle from pops.compile(...) (with .so_path); "
            "got %r" % type(compiled).__name__)
    if initial_state is not None and state is not None:
        raise TypeError("pops.bind: pass either initial_state= or state=, not both")
    initial = initial_state if initial_state is not None else state

    from pops.codegen._plans import require_install_plan
    plan = require_install_plan(compiled)
    target = plan.target
    layout = plan.layout

    # All defaults come from the immutable InstallPlan. An explicit solver selection is a genuine
    # bind-time input and may override that plan; there is no live Problem fallback.
    field_solvers = dict(plan.field_solvers)
    field_solvers.update(solvers or {})
    outputs = list(plan.outputs)
    diagnostics = list(plan.diagnostics)
    instances = plan.assemble_instances(initial or {})

    bind_schema = plan.bind_schema
    # One validation/materialisation point: this authenticates qualified handles, checks dtype/domain,
    # installs every explicit default and evaluates Bind-phase DerivedParam dependencies. Downstream
    # adapters receive the complete canonical mapping and never invent a fallback.
    resolved_params = bind_schema.resolve(params or {})

    # HARD REFUSAL GATES (ADC-537): reject a bad install with precise context BEFORE the native
    # artifact is loaded -- an initial state of the wrong shape/dtype/components/ghost depth, a
    # runtime param outside its typed domain, an aux a lowered operator requires but the state omits,
    # and an ABI/Kokkos/MPI/layout manifest mismatch. run_bind_gates raises ONE aggregated error on
    # any violation; there is no Python-runtime fallback when the native load fails.
    from pops.runtime._bind_validation import run_bind_gates
    run_bind_gates(compiled, layout, initial or {}, resolved_params, aux or {})

    # Delegate to the internal runtime adapter (lazy import: runtime edge, kept in-function so the
    # codegen module-scope import graph stays clean). adapter_for selects Uniform vs AMR from the
    # target the layout produced; the adapter builds the engine, installs, and wraps it in a
    # BoundSimulation view.
    from pops.runtime._bind_adapters import adapter_for

    adapter = adapter_for(target, layout, n_blocks=plan.n_blocks)
    return adapter.build(compiled, layout=layout, instances=instances, params=resolved_params,
                         aux=aux or {}, solvers=field_solvers, cadence=cadence, outputs=outputs,
                         diagnostics=diagnostics)


def _validate_layout_for_compile(problem: Any, layout: Any) -> None:
    """Validate the resolved layout at compile, once the layout is finally known (ADC-526).

    The Problem's ``validate()`` is layout-agnostic (it defers layout-specific checks to compile), so
    the layout's OWN ``validate`` runs HERE: an AMR envelope violation is refused, and a ``Uniform``
    layout that received active AMR criteria (a user recorded ``problem.amr.refine(...)`` then
    compiled with ``layout=Uniform(...)``) is refused with the clear "no level to refine onto"
    message rather than silently dropping the criteria. The field problems are also re-validated
    with the resolved layout in context so a solver can refuse a layout it cannot serve.
    """
    from pops.mesh.layouts import Uniform

    criteria = getattr(getattr(problem, "_constraints", None), "refinement", {}) or {}
    if criteria.get("refine") is not None and isinstance(layout, Uniform) \
            and getattr(layout, "ignore_amr", None) is None:
        criterion = criteria["refine"]
        sub = getattr(criterion, "criteria", None)
        names = [c.name for c in sub] if sub is not None else [criterion.name]
        raise ValueError(
            "pops.compile: layout=Uniform(...) but the Problem recorded active AMR criteria (%s) via "
            "problem.amr.refine(...); a single-level layout has no level to refine onto, and a "
            "criterion is never silently ignored. Compile with layout=AMR(...) to actually refine, "
            "or drop the refinement criteria." % ", ".join(names))
    context = {"layout": layout}
    layout.validate(context)
    for field in getattr(problem, "_fields", {}).items() if hasattr(
            getattr(problem, "_fields", None), "items") else []:
        field[1].validate(context)


def _resolve_layout(problem: Any, layout: Any) -> Any:
    """Resolve the effective compile layout into a detached compile-owned value (ADC-526/653).

    The layout lives on ``pops.compile(problem, layout=...)``: explicit @p layout wins, else a
    constructor layout the Problem may still carry; neither present is a loud error naming the
    spelling, and a disagreement between the two is refused (an artifact is frozen to one layout).
    The AMR criteria recorded via ``problem.amr.refine(...)`` sit only on the constraint registry.
    They are merged HERE into a fresh AMR descriptor, once the layout is known. If the layout and
    the Problem both declare the same policy slot, the two-authority configuration is rejected even
    when the values look equal: the user must choose exactly one declaration site. No user-owned
    layout is mutated or retained as the effective compile layout.
    """
    from pops.mesh.layouts import AMR, Uniform

    problem_layout = getattr(problem, "layout", None)
    if layout is None:
        if problem_layout is None:
            raise ValueError(
                "pops.compile: no layout given; pass layout=Uniform(...) or layout=AMR(...) to "
                "pops.compile(problem, layout=...). The Problem no longer carries a layout (ADC-526).")
        resolved = problem_layout
    else:
        if problem_layout is not None and layout is not problem_layout and layout != problem_layout:
            raise ValueError(
                "pops.compile: the explicit layout= (%r) disagrees with the problem's own layout "
                "(%r); a compiled artifact is frozen to one layout." % (layout, problem_layout))
        resolved = layout

    def resolve_references(value: Any) -> Any:
        protocol = getattr(value, "resolve_references", None)
        return protocol(problem.resolve) if callable(protocol) else value

    # A Uniform layout has no merge slots, but its optional refine descriptor still follows the same
    # typed-reference resolution protocol (including explicit IgnoreAMRCriteria authoring). Return a
    # fresh value so compile never retains a caller-owned mutable layout object.
    if isinstance(resolved, Uniform):
        return Uniform(
            mesh=resolved.mesh,
            embedded_boundary=resolved.embedded_boundary,
            refine=(resolve_references(resolved.refine)
                    if resolved.refine is not None else None),
            ignore_amr=resolved.ignore_amr,
        )

    if not isinstance(resolved, AMR):
        return resolved

    # Merge each layout-free Problem slot into a detached AMR descriptor. A double declaration is
    # an error, not last-writer-wins and not an idempotent special case.
    criteria = getattr(getattr(problem, "_constraints", None), "refinement", {}) or {}
    policies = {}
    for slot in ("refine", "regrid", "nesting", "patches"):
        layout_value = getattr(resolved, slot)
        problem_value = criteria.get(slot)
        if layout_value is not None and problem_value is not None:
            raise ValueError(
                "pops.compile: AMR %s is declared both on layout=AMR(%s=...) and through "
                "problem.amr.refine(..., %s=...); these are competing authorities. Keep the "
                "policy in exactly one place." % (slot, slot, slot))
        policies[slot] = problem_value if problem_value is not None else layout_value

    refine = policies["refine"]
    # ProblemAmrHandle already authenticated and canonicalised a registry-owned criterion exactly
    # once. A layout-owned criterion has not crossed that boundary yet and must be resolved here.
    if refine is not None and criteria.get("refine") is None:
        refine = resolve_references(refine)
    elif refine is not None and not getattr(refine, "references_authenticated", False):
        raise ValueError(
            "pops.compile: the ConstraintRegistry contains an unauthenticated AMR criterion; "
            "record it through problem.amr.refine(...) so every Handle leaf is resolved")
    output = resolved.output
    return AMR(
        base=resolved.base,
        max_levels=resolved.max_levels,
        ratio=resolved.ratio,
        regrid=policies["regrid"],
        patches=policies["patches"],
        refine=refine,
        nesting=policies["nesting"],
        checkpoint=resolved.checkpoint,
        output=resolve_references(output) if output is not None else None,
        clustering=resolved.clustering,
    )


def _resolve_problem_model(physics: Any) -> Any:
    """Resolve a block's physics to the model ``compile_problem`` accepts.

    A blackboard :class:`pops.physics.Model` exposes the underlying ``pops.dsl`` engine model
    via ``.dsl`` -- that is what ``compile_problem(model=...)`` wants. A ``pops.model.Module``
    or a raw ``pops.dsl`` model is forwarded as-is (``compile_problem`` lowers a ``Module``
    itself). ``None`` raises, so a block with no physics never reaches codegen.

    The operator-first :class:`pops.model.Module` is the CANONICAL compile IR (ADC-557): it is the
    validation + trace authority ``compile_problem`` captures (via ``lower_and_validate``) from the
    resolved model's ``.module``, so a facade ``Model`` never needs a manual ``m.to_module()`` in the
    standard flow. The model returned HERE is the one the kernel emitters consume (a dsl model / a
    facade Model duck-typed by the emitter); the Module IR authority is derived downstream. The
    Module -> dsl lowering that a raw Module needs stays INTERNAL to ``compile_problem``.
    """
    if physics is None:
        raise ValueError("pops.compile: the block has no physics model to resolve")
    dsl_model = getattr(physics, "dsl", None)
    if dsl_model is not None:
        return dsl_model
    return physics


# The AMR layout-lowering helpers moved to pops.runtime._bind_adapters (ADC-583): lowering a layout
# onto the AmrSystem config / refinement seams is runtime-adapter work, not codegen. They are still
# reachable as orchestration attributes via this LAZY forwarder (a function-scope import, so the
# codegen module-scope import graph stays runtime-free) for the few tests that reference them here.
_MOVED_TO_BIND_ADAPTERS = (
    "_amr_config_from_layout", "_flow_amr_layout", "_apply_refine_criterion",
    "_refine_subject_name", "_is_default_density_subject")


def __getattr__(name: str) -> Any:
    if name in _MOVED_TO_BIND_ADAPTERS:
        from pops.runtime import _bind_adapters
        return getattr(_bind_adapters, name)
    raise AttributeError("module %r has no attribute %r" % (__name__, name))


__all__ = ["compile", "bind"]
