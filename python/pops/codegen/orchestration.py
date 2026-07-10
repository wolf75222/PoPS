"""pops.codegen.orchestration -- thin pops.compile / pops.bind over the existing runtime.

These are the Spec 5 sec.11 lowering entry points for a :class:`pops.problem.Problem`:

* :func:`compile` validates the assembly, picks the compile target from the LAYOUT
  (``Uniform`` -> ``"system"``, ``AMR`` -> ``"amr_system"``; no user ``target=`` string),
  resolves each block's physics to the model the runtime wants, and compiles it. Both routes
  compile EACH block to its native ``.so`` loader; they differ in whether a whole-system time
  ``Program`` is ALSO compiled. A ``Uniform`` layout compiles the whole-system time ``Program`` once
  (``compile_problem``, BYTE-IDENTICAL to before). An ``AMR`` layout (single OR multi block) compiles
  EACH block to a ``target='amr_system'`` production ``CompiledModel`` (the native AMR ``.so``
  loader, ``add_native_block``) and carries the ``{block: CompiledModel}`` table on the handle; when
  the effective time is a whole-system ``Program`` it ALSO compiles that Program with
  ``compile_problem(target='amr_system')`` (ADC-634) -- the ``AmrSystem.install_program`` seam drives
  it per level over the hierarchy -- and returns the resulting ``CompiledProblem``. Without a
  Program (native per-block time policy) the AMR route stays byte-identical and returns the first
  block's ``CompiledModel``.
* :func:`bind` assembles the per-instance state mapping + field solvers + output policies, then
  delegates to an internal RUNTIME ADAPTER (:mod:`pops.runtime._bind_adapters`) selected from the
  carried target: ``layout=Uniform`` -> the Uniform adapter (``System``), ``layout=AMR`` -> the AMR
  adapter (``AmrSystem``). The adapter builds the engine, lowers the validated objects onto its
  INTERNAL ``_install_compiled`` seam, and returns a
  :class:`pops.runtime._bound_sim.BoundSimulation` view -- NOT the raw engine, so the legacy setters
  (``add_block`` / ``set_poisson`` / ``set_refinement`` / ``install_program`` / ...) are hidden.
  ``bind`` is the public entry point; the engines stay internal backends. No parallel runtime.

There is NO new codegen and NO new install machinery here: this module ORCHESTRATES the
proven pieces (``Model.compile`` for the per-block AMR loader, the runtime adapters for the
install). Every not-yet-wired route raises a clear ``NotImplementedError``.

Import-graph rule (Spec 4 / sec.4): ``codegen`` may import only ir / model / physics / time /
lib at module scope. The runtime (System / AmrSystem), the runtime adapters, mesh (AMR) and problem
types are pulled LAZILY inside the function bodies, so this module adds no forbidden cross-layer edge.
"""

from __future__ import annotations

from typing import Any

from pops.codegen._orchestration_instances import (
    assemble_instances as _assemble_instances,
    check_problem_not_mutated as _check_case_not_mutated,
    problem_field_solvers as _problem_field_solvers,
)


def compile(problem: Any, layout: Any = None, backend: Any = None, time: Any = None,
            **kwargs: Any) -> Any:
    """Lower a :class:`pops.problem.Problem` to a compiled handle.

    Validates @p problem, derives the compile target from the LAYOUT (``Uniform`` -> system,
    ``AMR`` -> amr_system) and lowers via the route the target selects. ADC-526: the layout is a
    ``pops.compile`` argument -- the Problem no longer owns a mandatory layout, so ONE Problem
    compiles under ``layout=Uniform(...)`` OR ``layout=AMR(...)``. A layout given to the Problem
    constructor (back-compat) is still honoured and must agree with @p layout; neither present is a
    loud error. The Problem's recorded AMR criteria (``problem.amr.refine(...)``) are applied to an
    ``AMR`` layout here:

    - ``Uniform``: compiles the whole-system time ``Program`` once with ``compile_problem``
      (BYTE-IDENTICAL to before), resolving each block's physics and carrying the full
      ``{block: model}`` table on ``_block_models`` so bind installs each block with its OWN model
      (multi-block Uniform Cases lower; C3). The time scheme is explicit here: @p time (a
      ``pops.time.Program``), else ``problem.time(...)``; a missing scheme raises (no silent default).
    - ``AMR`` (single OR multi block): compiles EACH block's resolved model to a
      ``target='amr_system'`` production ``CompiledModel`` (native AMR ``.so`` loader,
      ``add_native_block``) and carries ``{block: CompiledModel}`` on ``_block_compiled_models``.
      When the effective time is a whole-system ``Program`` (@p time, else ``problem._time``), it
      ALSO compiles that Program with ``compile_problem(target='amr_system')`` and returns the
      resulting ``CompiledProblem`` (ADC-634: ``AmrSystem.install_program`` drives it per level over
      the hierarchy). Without a Program the AMR route is byte-identical to before and returns the
      FIRST block's ``CompiledModel`` (native per-block time policy; @p time is not required).
      ``_target`` / ``_layout`` / ``_problem`` / ``_block_compiled_models`` are attached for bind's
      dispatch on both shapes.

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
        The compiled handle: a ``CompiledProblem`` on the Uniform route and on the AMR route WITH a
        whole-system Program (ADC-634), the first block's ``CompiledModel`` on the AMR route WITHOUT
        one. ``._problem`` / ``._target`` / ``._layout`` are set on every shape.
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
    layout = _resolve_layout(problem, layout)
    _validate_layout_for_compile(problem, layout)
    problem.validate()
    is_amr = isinstance(layout, AMR)
    target = "amr_system" if is_amr else "system"

    # Resolve the EFFECTIVE time scheme BEFORE the branch so the AMR route can read it too (ADC-634):
    # explicit @p time wins, else the Problem's recorded scheme. The Uniform route still REQUIRES one
    # (raise below); the AMR route treats it as optional (a whole-system Program to install on the
    # hierarchy when present, else the native per-block time policy).
    eff_time = time if time is not None else problem._time

    if not is_amr and eff_time is None:
        raise NotImplementedError(
            "pops.compile: a time scheme is required; pass time=pops.time.Program(...) or set "
            "it on the problem with problem.time(...). There is no default time scheme.")
    if "problem_snapshot" in kwargs:
        raise TypeError(
            "pops.compile computes and freezes its own ProblemSnapshot; problem_snapshot= is only "
            "accepted by the advanced pops.codegen.compile_problem driver")
    problem_snapshot = _prepare_problem_snapshot(problem, eff_time)

    if is_amr:
        # AMR route (single AND multi block): each block lowers to a target='amr_system' production
        # CompiledModel installed via the NATIVE add_native_block path at bind. When eff_time is a
        # whole-system Program, _compile_amr ALSO compiles it with compile_problem(target='amr_system')
        # and returns the CompiledProblem (ADC-634: AmrSystem.install_program drives it per level);
        # without one the byte-identical no-Program path returns the first block's CompiledModel. The
        # AMR branch is split into ``_orchestration_amr`` (ADC-550); import it lazily to keep the
        # module load order acyclic.
        from pops.codegen._orchestration_amr import _compile_amr
        return _compile_amr(
            problem, layout, backend, target, eff_time,
            problem_snapshot=problem_snapshot, **kwargs)

    time = eff_time

    # Resolve every block's physics model (C3): the whole-system time Program is compiled once with the
    # first block as the codegen representative (compiled.model -- the per-instance default at bind),
    # while _block_models carries the full {block_name: resolved model} table so bind()'s
    # _assemble_instances installs each block with its OWN model.
    block_models = {name: _resolve_problem_model(spec["model"])
                    for name, spec in problem._blocks.items()}
    _, model = next(iter(block_models.items()))

    from pops.codegen.compile_drivers import compile_problem
    compiled = compile_problem(
        time=time, model=model, backend=backend, target=target,
        problem_snapshot=problem_snapshot, **kwargs)
    compiled._problem = problem
    compiled._target = target
    compiled._block_models = block_models
    # COMPILE-TIME SNAPSHOT AUTHORITY (ADC-592): freeze WHAT the compile saw so bind() lowers from the
    # compile-time truth, not a LIVE re-read of a possibly-mutated Problem (mutating between compile and
    # bind would otherwise silently change what gets bound). We snap the per-block model + spatial, the
    # field solvers and the output policies at COMPILE time; bind() uses them as the authority and raises
    # a loud drift error if the live block-name set diverges.
    compiled._block_specs = {name: {"model": block_models[name], "spatial": spec["spatial"]}
                             for name, spec in problem._blocks.items()}
    compiled._field_solvers = _problem_field_solvers(problem)
    compiled._outputs, compiled._diagnostics = _capture_runtime_declarations(problem)
    # Declared diagnostic measures (ADC-542): carried on the compile-time snapshot exactly like the
    # output policies so bind() flows them onto the engine and run() fires them each cadence tick.
    # Carry the layout so bind()'s runtime adapter can derive the engine config from the mesh: the
    # Uniform adapter builds the System's SystemConfig (n / L / periodic) from the Uniform mesh,
    # mirroring how the AMR adapter derives the AmrSystemConfig (n / L / periodic / regrid / patch
    # settings) and flows the typed refinement. A handle NOT produced here carries no layout and
    # binds on the bare System() defaults.
    compiled._layout = layout
    _attach_problem_snapshot(compiled, problem_snapshot)
    return compiled


def _capture_runtime_declarations(problem: Any) -> tuple[list[Any], list[Any]]:
    """Detach and canonicalize output/diagnostic declarations for the compiled snapshot."""
    def resolved(value: Any, family: str) -> Any:
        resolve_references = getattr(value, "resolve_references", None)
        if not callable(resolve_references):
            raise TypeError(
                "%s declaration %r must implement resolve_references(resolver)"
                % (family, type(value).__name__))
        return resolve_references(problem.resolve)

    outputs = [resolved(value, "runtime output") for value in (problem._outputs or [])]
    diagnostics = [
        resolved(value, "runtime diagnostic") for value in (problem._diagnostics or [])]
    return outputs, diagnostics


def _prepare_problem_snapshot(problem: Any, time: Any) -> Any:
    """Freeze authoring before the driver computes or looks up an artifact identity."""
    from pops.problem._snapshot import prepare_compile_snapshot
    return prepare_compile_snapshot(problem, time)


def _attach_problem_snapshot(compiled: Any, snapshot: Any) -> None:
    """Attach the snapshot already used by the driver, then seal the public artifact."""
    from pops.problem._snapshot import attach_problem_snapshot
    attach_problem_snapshot(compiled, snapshot)


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
        compiled: A ``CompiledProblem`` from :func:`compile` (carries ``_problem`` / ``_target``).
        initial_state: dict {block_name: array} of per-block initial state (alias: @p state).
        state: Alias for @p initial_state (only one may be given).
        params: dict {param_name: value} of runtime parameter overrides.
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

    problem = getattr(compiled, "_problem", None)
    target = getattr(compiled, "_target", "system")
    layout = getattr(compiled, "_layout", None)

    # COMPILE-TIME SNAPSHOT AUTHORITY (ADC-592): drift-check the LIVE Problem against what compile froze,
    # then lower from the compile-time snapshot -- not a fresh live re-read -- so a Problem mutated between
    # compile and bind cannot silently change what gets bound. A block-name divergence is a loud error;
    # an explicit solvers= kwarg is a documented override, not drift.
    block_specs = getattr(compiled, "_block_specs", None)
    _check_case_not_mutated(problem, block_specs)

    # Field solvers from the COMPILE-TIME snapshot (compiled._field_solvers), NOT a live re-read; an
    # explicit @p solvers still overrides (documented user input).
    field_solvers_src: Any = (getattr(compiled, "_field_solvers", None)
                              if getattr(compiled, "_field_solvers", None) is not None
                              else _problem_field_solvers(problem))
    field_solvers = dict(field_solvers_src)
    field_solvers.update(solvers or {})

    # OUTPUT / CHECKPOINT policies (C4 / ADC-509) from the COMPILE-TIME snapshot (compiled._outputs),
    # so the bound sim's run() fires exactly the policies the compile saw. Empty for a Problem with no
    # .output(...) -- the install is unchanged.
    _outputs_src = getattr(compiled, "_outputs", None)
    outputs = list(_outputs_src if _outputs_src is not None
                   else (getattr(problem, "_outputs", []) or []))
    # Declared diagnostic measures (ADC-542) from the same compile-time snapshot, flowed onto the
    # engine so run() fires each at its cadence via the native reductions.
    _diag_src = getattr(compiled, "_diagnostics", None)
    diagnostics = list(_diag_src if _diag_src is not None
                       else (getattr(problem, "_diagnostics", []) or []))

    # The AMR install goes through the NATIVE per-block path (each instance carries its OWN
    # target='amr_system' CompiledModel from compile()'s _block_compiled_models); the Uniform install
    # carries the whole-system compiled time Program (@p compiled). _assemble_instances builds the
    # right per-block model table from the COMPILE-TIME block specs (models + spatial), so it is immune
    # to a post-compile Problem mutation (the Uniform route used to re-resolve spec["model"] live). A
    # legacy handle with no _block_specs falls back to the historical live-read path (AMR then routes
    # via _block_compiled_models, byte-identical to before).
    n_blocks = len(block_specs) if block_specs is not None else (
        len(problem._blocks) if problem is not None else 1)
    amr_models = (getattr(compiled, "_block_compiled_models", None)
                  if target == "amr_system" else None)
    instances = _assemble_instances(problem, initial or {}, block_specs=block_specs,
                                    models=amr_models)

    # HARD REFUSAL GATES (ADC-537): reject a bad install with precise context BEFORE the native
    # artifact is loaded -- an initial state of the wrong shape/dtype/components/ghost depth, a
    # runtime param outside its typed domain, an aux a lowered operator requires but the state omits,
    # and an ABI/Kokkos/MPI/layout manifest mismatch. run_bind_gates raises ONE aggregated error on
    # any violation; there is no Python-runtime fallback when the native load fails.
    from pops.runtime._bind_validation import run_bind_gates
    run_bind_gates(compiled, problem, layout, initial or {}, params or {}, aux or {})

    # Delegate to the internal runtime adapter (lazy import: runtime edge, kept in-function so the
    # codegen module-scope import graph stays clean). adapter_for selects Uniform vs AMR from the
    # target the layout produced; the adapter builds the engine, installs, and wraps it in a
    # BoundSimulation view.
    from pops.runtime._bind_adapters import adapter_for

    adapter = adapter_for(target, layout, n_blocks=n_blocks)
    return adapter.build(compiled, layout=layout, instances=instances, params=params or {},
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
