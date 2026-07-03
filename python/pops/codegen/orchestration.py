"""pops.codegen.orchestration -- thin pops.compile / pops.bind over the existing runtime.

These are the Spec 5 sec.11 lowering entry points for a :class:`pops.problem.Problem`:

* :func:`compile` validates the assembly, picks the compile target from the LAYOUT
  (``Uniform`` -> ``"system"``, ``AMR`` -> ``"amr_system"``; no user ``target=`` string),
  resolves each block's physics to the model the runtime wants, and compiles it. The TWO routes
  differ in WHAT is compiled: a ``Uniform`` layout compiles the whole-system time ``Program`` once
  (``compile_problem``, BYTE-IDENTICAL to before); an ``AMR`` layout (single OR multi block)
  compiles EACH block to a ``target='amr_system'`` production ``CompiledModel`` (the native AMR
  ``.so`` loader, ``add_native_block``) and carries the ``{block: CompiledModel}`` table on the
  handle -- there is NO whole-system time Program on AMR (``AmrSystem`` has no ``install_program``
  seam; the native blocks carry their own time policy).
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


def compile(problem, layout=None, backend="production", time=None, **kwargs):
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
      ``target='amr_system'`` production ``CompiledModel`` (the native AMR ``.so`` loader,
      ``add_native_block``) and carries the ``{block: CompiledModel}`` table on
      ``_block_compiled_models``. There is NO whole-system time Program on AMR (``AmrSystem`` has no
      ``install_program`` seam), so the AMR route does NOT require @p time -- the native blocks carry
      their own per-block time policy (set at bind from the block spec). The handle returned is the
      FIRST block's ``CompiledModel`` (it satisfies bind's ``.so_path`` guard); ``_target`` /
      ``_layout`` / ``_problem`` are attached for bind's AMR dispatch.

    Args:
        problem: The :class:`pops.problem.Problem` assembly to lower.
        layout: The mesh layout (``Uniform`` / ``AMR``) to compile for (ADC-526). Required unless the
            Problem carries a constructor layout (back-compat); when both are given they must agree (a
            mismatch raises), so a compiled artifact is frozen to exactly one layout.
        backend: The codegen backend (default "production"). The AMR route requires "production"
            (the only native AMR loader, ``Model.compile`` enforces it).
        time: The ``pops.time.Program`` time scheme (Uniform route only); falls back to
            ``problem._time``. Ignored on the AMR route (the native blocks carry their own time policy).
        **kwargs: Extra keyword args forwarded verbatim to the compile driver (so_path / force / cxx /
            include / std / debug / libraries on the Uniform route; include / cxx / std on the AMR route).

    Returns:
        The compiled handle (a ``CompiledProblem`` on the Uniform route, the first block's
        ``CompiledModel`` on the AMR route), with ``._problem`` / ``._target`` / ``._layout`` set.
    """
    # Lazy imports keep the codegen layer's module-scope import graph clean (no mesh / runtime).
    from pops.mesh.layouts import AMR

    # ADC-526: resolve the effective layout FIRST (the Problem no longer carries a mandatory layout),
    # applying the Problem's recorded AMR criteria to it. A missing layout / a layout= that disagrees
    # with a constructor layout is refused here.
    layout = _resolve_layout(problem, layout)
    # Validate the resolved layout NOW that it is known: an AMR(max_levels) / AMR(ratio) beyond the
    # native envelope (NATIVE_MAX_LEVELS / NATIVE_RATIOS), or a Uniform layout carrying active AMR
    # criteria (recorded via problem.amr.refine on a Uniform compile), is refused before any compile
    # with the existing clear message -- never silently clamped or dropped.
    _validate_layout_for_compile(problem, layout)
    # problem.validate() runs the structural registry checks (blocks / fields / collisions).
    problem.validate()
    is_amr = isinstance(layout, AMR)
    target = "amr_system" if is_amr else "system"

    if is_amr:
        # AMR route (single AND multi block): no whole-system time Program. Each block lowers to a
        # target='amr_system' production CompiledModel and installs through the NATIVE add_native_block
        # path at bind (compiled=None, instances carry the per-block CompiledModel). The AMR runtime has
        # no install_program seam, so time= is NOT required here -- the per-block time policy is set at
        # bind from the block spec. compile_problem is NOT called (the byte-identical Uniform path is
        # untouched).
        return _compile_amr(problem, layout, backend, target, **kwargs)

    time = time if time is not None else problem._time
    if time is None:
        raise NotImplementedError(
            "pops.compile: a time scheme is required; pass time=pops.time.Program(...) or set "
            "it on the problem with problem.time(...). There is no default time scheme.")

    # Resolve every block's physics model (C3): the whole-system time Program is compiled once with the
    # first block as the codegen representative (compiled.model -- the per-instance default at bind),
    # while _block_models carries the full {block_name: resolved model} table so bind()'s
    # _assemble_instances installs each block with its OWN model.
    block_models = {name: _resolve_problem_model(spec["model"])
                    for name, spec in problem._blocks.items()}
    _, model = next(iter(block_models.items()))

    from pops.codegen.compile_drivers import compile_problem
    compiled = compile_problem(time=time, model=model, backend=backend, target=target, **kwargs)
    compiled._problem = problem
    compiled._target = target
    compiled._block_models = block_models
    # COMPILE-TIME SNAPSHOT AUTHORITY (ADC-592): freeze WHAT the compile saw so bind() lowers from the
    # compile-time truth, not a LIVE re-read of a possibly-mutated Problem. Without this, bind() re-resolves
    # problem._blocks / problem._fields / problem._outputs from the live object, so mutating the Problem
    # between compile and bind would silently change what gets bound (the proven vulnerability). We snap
    # the per-block model + spatial, the field solvers and the output policies at COMPILE time; bind()
    # uses them as the authority and raises a loud drift error if the live block-name set diverges.
    compiled._block_specs = {name: {"model": block_models[name], "spatial": spec["spatial"]}
                             for name, spec in problem._blocks.items()}
    compiled._field_solvers = _problem_field_solvers(problem)
    compiled._outputs = list(problem._outputs or [])
    # Carry the layout so bind()'s runtime adapter can derive the engine config from the mesh: the
    # Uniform adapter builds the System's SystemConfig (n / L / periodic) from the Uniform mesh,
    # mirroring how the AMR adapter derives the AmrSystemConfig (n / L / periodic / regrid / patch
    # settings) and flows the typed refinement. A handle NOT produced here carries no layout and
    # binds on the bare System() defaults.
    compiled._layout = layout
    return compiled


def _compile_amr(problem, layout, backend, target, **kwargs):
    """Compile each AMR block to a ``target='amr_system'`` ``CompiledModel`` (single AND multi block).

    There is no whole-system time Program on AMR: each block's resolved physics model is compiled to a
    production native loader (``Model.compile(backend='production', target='amr_system')`` -> a
    ``CompiledModel`` whose adder is ``add_native_block``), and the ``{block: CompiledModel}`` table is
    carried on ``_block_compiled_models`` for bind's ``_assemble_instances`` to install each block via
    ``add_equation``. The returned handle is the FIRST block's ``CompiledModel`` -- it carries a real
    ``.so_path`` so bind's so_path guard passes, and ``_target`` / ``_layout`` / ``_problem`` /
    ``_block_compiled_models`` are attached for the AMR dispatch.

    Only the compile-driver kwargs ``Model.compile`` accepts are forwarded (include / cxx / std /
    name / require_metadata / hoist_reciprocals); a whole-system kwarg like ``force`` / ``debug`` /
    ``libraries`` has no per-block-loader equivalent and is dropped (the AMR route does not build a
    Program .so). An explicit ``so_path`` is forwarded ONLY for a single block: pinning the SAME path
    for several blocks would make their loaders collide (one .so per block), so a multi-block Problem lets
    each block fall back to its model-hash-keyed cache path.
    """
    compile_kwargs = {k: v for k, v in kwargs.items()
                      if k in ("include", "cxx", "std", "name", "require_metadata",
                               "hoist_reciprocals")}
    if "so_path" in kwargs and len(problem._blocks) == 1:
        compile_kwargs["so_path"] = kwargs["so_path"]
    block_compiled = {name: _compile_block_amr(name, spec["model"], backend, compile_kwargs)
                      for name, spec in problem._blocks.items()}
    _, compiled = next(iter(block_compiled.items()))
    compiled._problem = problem
    compiled._target = target
    compiled._block_compiled_models = block_compiled
    # COMPILE-TIME SNAPSHOT AUTHORITY (ADC-592, parity with the Uniform route): the AMR route is already
    # snapshot-safe for the per-block models (_block_compiled_models table), but the field solvers /
    # output policies / spatial are still re-read live at bind. Freeze them at compile so a Problem mutated
    # between compile and bind is caught by bind()'s drift check rather than silently rebound.
    compiled._block_specs = {name: {"model": block_compiled[name], "spatial": spec["spatial"]}
                             for name, spec in problem._blocks.items()}
    compiled._field_solvers = _problem_field_solvers(problem)
    compiled._outputs = list(problem._outputs or [])
    # Carry the AMR layout so bind() can rebuild the AmrSystemConfig (n / L / periodic / regrid /
    # patch settings) and flow the typed refinement + field problem onto the AmrSystem.
    compiled._layout = layout
    return compiled


def _compile_block_amr(name, physics, backend, compile_kwargs):
    """Compile one AMR block's physics to a ``target='amr_system'`` ``CompiledModel``.

    Resolves the block's physics to the underlying engine model (``_resolve_problem_model``: a
    blackboard ``pops.physics.Model`` -> its ``.dsl`` engine, a ``pops.dsl.Model`` as-is) and calls its
    ``.compile(backend=..., target='amr_system', **compile_kwargs)``. A model that exposes no such
    ``.compile`` (e.g. a raw ``pops.model.Module``, which has no per-block native loader) raises a clear
    error rather than a cryptic ``AttributeError``.
    """
    model = _resolve_problem_model(physics)
    block_compile = getattr(model, "compile", None)
    if not callable(block_compile):
        raise NotImplementedError(
            "pops.compile: block %r resolves to a %r, which has no .compile(...) producing a "
            "target='amr_system' CompiledModel; an AMR block needs a pops.dsl.Model / "
            "pops.physics.Model physics (a raw pops.model.Module has no per-block AMR loader)."
            % (name, type(model).__name__))
    return block_compile(backend=backend, target="amr_system", **compile_kwargs)


def bind(compiled, *, initial_state=None, state=None, params=None, aux=None,
         solvers=None, cadence=None):
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
    field_solvers = dict(getattr(compiled, "_field_solvers", None)
                         if getattr(compiled, "_field_solvers", None) is not None
                         else _problem_field_solvers(problem))
    field_solvers.update(solvers or {})

    # OUTPUT / CHECKPOINT policies (C4 / ADC-509) from the COMPILE-TIME snapshot (compiled._outputs),
    # so the bound sim's run() fires exactly the policies the compile saw. Empty for a Problem with no
    # .output(...) -- the install is unchanged.
    outputs = list(getattr(compiled, "_outputs", None)
                   if getattr(compiled, "_outputs", None) is not None
                   else (getattr(problem, "_outputs", []) or []))

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

    # Delegate to the internal runtime adapter (lazy import: runtime edge, kept in-function so the
    # codegen module-scope import graph stays clean). adapter_for selects Uniform vs AMR from the
    # target the layout produced; the adapter builds the engine, installs, and wraps it in a
    # BoundSimulation view.
    from pops.runtime._bind_adapters import adapter_for

    adapter = adapter_for(target, layout, n_blocks=n_blocks)
    return adapter.build(compiled, layout=layout, instances=instances, params=params or {},
                         aux=aux or {}, solvers=field_solvers, cadence=cadence, outputs=outputs)


def _validate_layout_for_compile(problem, layout):
    """Validate the resolved layout at compile, once the layout is finally known (ADC-526).

    The Problem's ``validate()`` is layout-agnostic (it defers layout-specific checks to compile), so
    the layout's OWN ``validate`` runs HERE: an AMR envelope violation is refused, and a ``Uniform``
    layout that received active AMR criteria (a user recorded ``problem.amr.refine(...)`` then
    compiled with ``layout=Uniform(...)``) is refused with the clear "no level to refine onto"
    message rather than silently dropping the criteria. The field problems are also re-validated
    with the resolved layout in context so a solver can refuse a layout it cannot serve.
    """
    from pops.mesh.layouts import AMR, Uniform

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


def _resolve_layout(problem, layout):
    """Resolve the effective compile layout, then apply the problem's AMR criteria (ADC-526).

    ADC-526 completes the move of the layout to ``pops.compile(problem, layout=...)``: the Problem no
    longer owns a mandatory layout. The effective layout is the explicit @p layout when given, else a
    layout the Problem may still carry from its constructor (back-compat); neither present is a loud
    error pointing at ``pops.compile(problem, layout=...)``. When BOTH are given they must agree (a
    compiled artifact is frozen to one layout), so a disagreement is refused rather than silently
    overridden.

    The AMR refinement criteria the user recorded via ``problem.amr.refine(...)`` live on the
    Problem's constraint registry (layout-free); they are applied to the resolved layout HERE, when
    the layout is finally known, so ONE Problem compiles under a plain ``Uniform`` or under an
    ``AMR`` that receives its refine / regrid / nesting / patches.
    """
    from pops.mesh.layouts import AMR

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

    # Apply the layout-free AMR criteria recorded on the Problem's constraint registry to the AMR
    # layout (ADC-526). A Uniform layout ignores them (its own validate refuses active criteria);
    # criteria the constructor-layout path already wrote are re-applied idempotently.
    criteria = getattr(getattr(problem, "_constraints", None), "refinement", {}) or {}
    if criteria and isinstance(resolved, AMR):
        for slot in ("refine", "regrid", "nesting", "patches"):
            if criteria.get(slot) is not None:
                setattr(resolved, slot, criteria[slot])
    return resolved


def _resolve_problem_model(physics):
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


def _assemble_instances(problem, initial, block_specs=None, models=None):
    """Build the ``sim.install`` instances mapping from the COMPILE-TIME block specs + initial state.

    Each block becomes ``{name: {"model": <model>, "spatial": spatial, "initial": state}}`` -- the
    shape the unified install consumes. The per-block model + spatial come from @p block_specs, the
    COMPILE-TIME snapshot (``compiled._block_specs``, ADC-592): the model is the resolved engine model
    (Uniform, handed to the compiled time Program) OR the block's own ``target='amr_system'``
    CompiledModel (AMR), captured at compile so a post-compile Problem mutation cannot change it. The
    per-block initial state comes from @p initial (keyed by block name); an unknown key raises so a
    typo is not silently dropped.

    @p block_specs of ``None`` is a legacy/degraded handle (produced without the ADC-592 snapshot):
    fall back to a LIVE re-read of ``problem._blocks`` (byte-identical to before for a non-mutated
    Problem). On that fallback path, @p models -- when given (the AMR route's
    ``compiled._block_compiled_models`` table) -- supplies each block's own ``target='amr_system'``
    CompiledModel (installed via ``add_native_block``) instead of the resolved engine model.
    """
    if problem is None and block_specs is None:
        raise TypeError("pops.bind: the compiled handle carries no problem assembly "
                        "(was it produced by pops.compile?)")
    declared = set(block_specs) if block_specs is not None else set(problem._blocks)
    unknown = sorted(set(initial) - declared)
    if unknown:
        raise ValueError("pops.bind: initial state for unknown block(s) %s; declared blocks: %s"
                         % (unknown, sorted(declared)))
    instances = {}
    if block_specs is not None:
        for name, snap in block_specs.items():
            entry = {"model": snap["model"], "spatial": snap["spatial"]}
            if name in initial:
                entry["initial"] = initial[name]
            instances[name] = entry
        return instances
    # Legacy fallback (no compile-time snapshot on the handle): live re-read of the problem blocks.
    for name, spec in problem._blocks.items():
        if models is not None:
            if name not in models:
                raise ValueError(
                    "pops.bind: block %r has no compiled model on the handle; the AMR handle must "
                    "carry one CompiledModel per block (was it produced by pops.compile?)" % (name,))
            model = models[name]
        else:
            model = _resolve_problem_model(spec["model"])
        entry = {"model": model, "spatial": spec["spatial"]}
        if name in initial:
            entry["initial"] = initial[name]
        instances[name] = entry
    return instances


def _check_case_not_mutated(problem, block_specs):
    """Raise a LOUD error when the LIVE Problem's blocks diverge from the compile-time snapshot (ADC-592).

    ``compiled._block_specs`` is the block-name set the compile FROZE; if the live ``problem._blocks``
    no longer matches (a block added / removed after ``pops.compile``), bind would silently bind a
    stale composition. We refuse it, naming the drift, and point at a recompile. A degraded handle
    (no snapshot, ``block_specs is None``) or a handle with no live problem is skipped (nothing to
    compare -- the legacy live-read path stays byte-identical for a non-mutated Problem)."""
    if block_specs is None or problem is None:
        return
    live = set(getattr(problem, "_blocks", {}) or {})
    frozen = set(block_specs)
    if live != frozen:
        added = sorted(live - frozen)
        removed = sorted(frozen - live)
        raise ValueError(
            "pops.bind: the Problem was mutated after pops.compile (blocks changed: added=%s removed=%s);"
            " a compiled artifact is frozen at compile time and is not affected by a later Problem "
            "mutation -- recompile the Problem (pops.compile(...)) before pops.bind(...)."
            % (added, removed))


def _problem_field_solvers(problem):
    """The {field_name: solver} mapping derived from the problem's field problems."""
    if problem is None:
        return {}
    return {name: fp.solver for name, fp in problem._fields.items() if fp.solver is not None}


# The AMR layout-lowering helpers moved to pops.runtime._bind_adapters (ADC-583): lowering a layout
# onto the AmrSystem config / refinement seams is runtime-adapter work, not codegen. They are still
# reachable as orchestration attributes via this LAZY forwarder (a function-scope import, so the
# codegen module-scope import graph stays runtime-free) for the few tests that reference them here.
_MOVED_TO_BIND_ADAPTERS = (
    "_amr_config_from_layout", "_flow_amr_layout", "_apply_refine_criterion",
    "_refine_subject_name", "_is_default_density_subject")


def __getattr__(name):
    if name in _MOVED_TO_BIND_ADAPTERS:
        from pops.runtime import _bind_adapters
        return getattr(_bind_adapters, name)
    raise AttributeError("module %r has no attribute %r" % (__name__, name))


__all__ = ["compile", "bind"]
