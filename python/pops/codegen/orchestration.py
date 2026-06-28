"""pops.codegen.orchestration -- thin pops.compile / pops.bind over the existing runtime.

These are the Spec 5 sec.11 lowering entry points for a :class:`pops.case.Case`:

* :func:`compile` validates the assembly, picks the compile target from the LAYOUT
  (``Uniform`` -> ``"system"``, ``AMR`` -> ``"amr_system"``; no user ``target=`` string),
  resolves each block's physics to the model the runtime wants, and compiles it. The TWO routes
  differ in WHAT is compiled: a ``Uniform`` layout compiles the whole-system time ``Program`` once
  (``compile_problem``, BYTE-IDENTICAL to before); an ``AMR`` layout (single OR multi block)
  compiles EACH block to a ``target='amr_system'`` production ``CompiledModel`` (the native AMR
  ``.so`` loader, ``add_native_block``) and carries the ``{block: CompiledModel}`` table on the
  handle -- there is NO whole-system time Program on AMR (``AmrSystem`` has no ``install_program``
  seam; the native blocks carry their own time policy).
* :func:`bind` dispatches ``System`` vs ``AmrSystem`` from the carried target, assembles the
  per-instance state mapping, and calls the INTERNAL ``sim._install_compiled(compiled, instances=,
  ...)`` seam (``pops.runtime._system_unified_install``). For ``amr_system`` it installs through the
  NATIVE path (``compiled=None`` with each instance carrying its ``target='amr_system'``
  ``CompiledModel``), sidestepping the blocked whole-system time-Program install. ``bind`` is the
  public entry point; the ``_install_compiled`` seam is undocumented / low-level. No parallel runtime.

There is NO new codegen and NO new install machinery here: this module ORCHESTRATES the
proven pieces (``Model.compile`` for the per-block AMR loader, ``add_equation`` for the native
install). Every not-yet-wired route raises a clear ``NotImplementedError``.

Import-graph rule (Spec 4 / sec.4): ``codegen`` may import only ir / model / physics / time /
lib at module scope. The runtime (System / AmrSystem), mesh (AMR) and case types are pulled
LAZILY inside the function bodies, so this module adds no forbidden cross-layer edge.
"""


def compile(problem, backend="production", time=None, **kwargs):
    """Lower a :class:`pops.case.Case` to a compiled handle.

    Validates @p problem, derives the compile target from its LAYOUT (``Uniform`` -> system,
    ``AMR`` -> amr_system) and lowers via the route the target selects:

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
        problem: The :class:`pops.case.Case` assembly to lower.
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

    # problem.validate() runs the layout's own check, so an AMR(max_levels) / AMR(ratio) beyond the
    # native envelope (NATIVE_MAX_LEVELS / NATIVE_RATIOS) is refused HERE with the existing clear
    # AMR.available message before any compile, never silently clamped.
    problem.validate()
    is_amr = isinstance(problem.layout, AMR)
    target = "amr_system" if is_amr else "system"

    if is_amr:
        # AMR route (single AND multi block): no whole-system time Program. Each block lowers to a
        # target='amr_system' production CompiledModel and installs through the NATIVE add_native_block
        # path at bind (compiled=None, instances carry the per-block CompiledModel). The AMR runtime has
        # no install_program seam, so time= is NOT required here -- the per-block time policy is set at
        # bind from the block spec. compile_problem is NOT called (the byte-identical Uniform path is
        # untouched).
        return _compile_amr(problem, backend, target, **kwargs)

    time = time if time is not None else problem._time
    if time is None:
        raise NotImplementedError(
            "pops.compile: a time scheme is required; pass time=pops.time.Program(...) or set "
            "it on the problem with problem.time(...). There is no default time scheme.")

    # Resolve every block's physics model (C3): the whole-system time Program is compiled once with the
    # first block as the codegen representative (compiled.model -- the per-instance default at bind),
    # while _block_models carries the full {block_name: resolved model} table so bind()'s
    # _assemble_instances installs each block with its OWN model.
    block_models = {name: _resolve_problem_model(spec["physics"])
                    for name, spec in problem._blocks.items()}
    _, model = next(iter(block_models.items()))

    from pops.codegen.compile_drivers import compile_problem
    compiled = compile_problem(time=time, model=model, backend=backend, target=target, **kwargs)
    compiled._problem = problem
    compiled._target = target
    compiled._block_models = block_models
    # Carry the AMR layout so bind() can rebuild the AmrSystemConfig (n / L / periodic / regrid /
    # patch settings) and flow the typed refinement + field problem onto the AmrSystem. None for a
    # Uniform layout (System bind reads no layout); set only on the AMR route.
    compiled._layout = None
    return compiled


def _compile_amr(problem, backend, target, **kwargs):
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
    for several blocks would make their loaders collide (one .so per block), so a multi-block Case lets
    each block fall back to its model-hash-keyed cache path.
    """
    compile_kwargs = {k: v for k, v in kwargs.items()
                      if k in ("include", "cxx", "std", "name", "require_metadata",
                               "hoist_reciprocals")}
    if "so_path" in kwargs and len(problem._blocks) == 1:
        compile_kwargs["so_path"] = kwargs["so_path"]
    block_compiled = {name: _compile_block_amr(name, spec["physics"], backend, compile_kwargs)
                      for name, spec in problem._blocks.items()}
    _, compiled = next(iter(block_compiled.items()))
    compiled._problem = problem
    compiled._target = target
    compiled._block_compiled_models = block_compiled
    # Carry the AMR layout so bind() can rebuild the AmrSystemConfig (n / L / periodic / regrid /
    # patch settings) and flow the typed refinement + field problem onto the AmrSystem.
    compiled._layout = problem.layout
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
    (``compiled = pops.compile(...)``); it dispatches ``System`` vs ``AmrSystem`` from the target
    carried on @p compiled (set by :func:`compile`), builds the per-instance state mapping from the
    problem's blocks and the supplied initial state, derives the field solvers from the problem's
    field problems (an explicit @p solvers overrides), flows the Case's output / checkpoint policies
    (C4 / ADC-509) so the bound sim's ``run(output_dir=...)`` fires them at each policy cadence, and
    calls the INTERNAL ``sim._install_compiled(compiled, instances=, params=, aux=, solvers=,
    cadence=, outputs=)`` seam -- the low-level install lowering, not a public entry. For
    ``target='amr_system'`` (ADC-503) the install goes through the NATIVE path
    (``_install_compiled(compiled=None, ...)`` with each instance carrying its own
    ``target='amr_system'`` ``CompiledModel`` from the handle's ``_block_compiled_models``), since the
    AMR runtime has no whole-system ``install_program`` seam. Returns the bound simulation (the
    ``System`` / ``AmrSystem`` is the Simulation facade for now): call ``sim.run(...)`` to advance it.

    Args:
        compiled: A ``CompiledProblem`` from :func:`compile` (carries ``_problem`` / ``_target``).
        initial_state: dict {block_name: array} of per-block initial state (alias: @p state).
        state: Alias for @p initial_state (only one may be given).
        params: dict {param_name: value} of runtime parameter overrides.
        aux: dict {aux_name: array} of static aux inputs.
        solvers: dict {field: solver} overriding the per-field solvers from the problem.
        cadence: optional ``pops.time.CompiledTime`` macro-step cadence.

    Returns:
        The bound ``System`` / ``AmrSystem`` simulation handle.
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

    field_solvers = _problem_field_solvers(problem)
    field_solvers.update(solvers or {})

    # OUTPUT / CHECKPOINT policies (C4 / ADC-509): flow the Case's stored output policies onto the
    # install seam so the bound sim's run() fires them at each policy cadence (the existing
    # write()/checkpoint writers). Empty for a Case with no .output(...) -- the install is unchanged.
    outputs = list(getattr(problem, "_outputs", []) or [])

    from pops.runtime.system import AmrSystem, System
    if target == "amr_system":
        # Build the AmrSystem from an AmrSystemConfig DERIVED from the AMR layout (n / L / periodic
        # from the base CartesianMesh, regrid cadence from AMR.regrid, patch settings from
        # AMR.patches), then flow the typed refinement (problem.amr / AMR.refine -> set_refinement /
        # set_phi_refinement) BEFORE install, mirroring the old string path. A missing layout (an AMR
        # target with no carried descriptor) is a bug.
        if layout is None:
            raise TypeError(
                "pops.bind: an AMR target carries no layout descriptor; the compiled handle must "
                "come from pops.compile(problem_with_AMR_layout, ...)")
        n_blocks = len(problem._blocks) if problem is not None else 1
        sim = AmrSystem(_amr_config_from_layout(layout))
        _flow_amr_layout(sim, layout, n_blocks=n_blocks)
        # AMR install goes through the NATIVE path (compiled=None): each instance carries its OWN
        # target='amr_system' CompiledModel (from compile()'s _block_compiled_models), which
        # _install_compiled wires with add_equation -> add_native_block. There is NO whole-system time
        # Program install on AMR (AmrSystem rejects compiled != None), so the handle's per-block table
        # is the install payload, not the handle itself.
        instances = _assemble_instances(problem, initial or {},
                                        models=getattr(compiled, "_block_compiled_models", None))
        sim._install_compiled(compiled=None, instances=instances, params=params or {},
                              aux=aux or {}, solvers=field_solvers, cadence=cadence, outputs=outputs)
        return sim

    sim = System()
    instances = _assemble_instances(problem, initial or {})
    sim._install_compiled(compiled, instances=instances, params=params or {}, aux=aux or {},
                          solvers=field_solvers, cadence=cadence, outputs=outputs)
    return sim


def _amr_config_from_layout(layout):
    """Build an ``AmrSystemConfig`` from a :class:`pops.mesh.layouts.AMR` descriptor.

    Maps the inert AMR layout onto the C++ runtime config the ``AmrSystem`` constructor consumes:

      - ``n`` / ``L`` / ``periodic`` from the base ``CartesianMesh`` (``layout.base``);
      - ``regrid_every`` from ``layout.regrid``: a ``RegridEvery(n)`` -> ``n``, a ``FrozenRegrid``
        (or no regrid policy) -> ``0`` (a frozen hierarchy, bit-identical);
      - ``distribute_coarse`` / ``coarse_max_grid`` from ``layout.patches`` (a ``PatchLayout``),
        else the C++ defaults.

    The native AMR route is fixed at ``NATIVE_MAX_LEVELS`` levels / ratio ``NATIVE_RATIOS`` (the
    config carries no ``max_levels`` / ``ratio`` field); ``layout.max_levels`` / ``layout.ratio``
    are validated against that envelope by ``compile`` (and ``AMR.available`` / ``validate``), not
    flowed as config knobs. Imported lazily so this codegen module stays mesh-import-free.
    """
    from pops._bootstrap import AmrSystemConfig
    from pops.mesh.amr import FrozenRegrid, PatchLayout, RegridEvery

    base = layout.base
    cfg = AmrSystemConfig()
    cfg.n = int(base.n)
    cfg.L = float(base.L)
    cfg.periodic = bool(base.periodic)

    regrid = layout.regrid
    if isinstance(regrid, RegridEvery):
        cfg.regrid_every = int(regrid.steps)
    elif regrid is None or isinstance(regrid, FrozenRegrid):
        cfg.regrid_every = 0
    else:
        raise TypeError(
            "pops.bind: AMR.regrid must be a pops.mesh.amr.RegridEvery(n) / FrozenRegrid() "
            "(got %r)" % type(regrid).__name__)

    patches = layout.patches
    if isinstance(patches, PatchLayout):
        cfg.distribute_coarse = bool(patches.distribute_coarse)
        cfg.coarse_max_grid = int(patches.coarse_max_grid)
    elif patches is not None:
        raise TypeError(
            "pops.bind: AMR.patches must be a pops.mesh.amr.PatchLayout(...) (got %r)"
            % type(patches).__name__)
    return cfg


def _flow_amr_layout(sim, layout, n_blocks=1):
    """Flow the AMR layout's typed refinement criterion onto @p sim BEFORE the blocks are installed.

    Mirrors the old string path: a ``Refine.on(subject).above(threshold)`` (or a ``TagUnion`` of
    them) becomes ``set_refinement(threshold, ...)`` and a ``gradient``-predicate on the potential
    becomes ``set_phi_refinement(threshold)``. The field problem (Poisson) is flowed through the
    unified ``install(solvers=...)`` seam (``AmrSystem._install_solver`` -> ``set_poisson``), which
    runs its field solvers BEFORE adding the blocks, so it is not duplicated here.

    @p n_blocks is the declared block count: the per-block variable / role selector is only wired in
    MULTI-BLOCK (>= 2 blocks, the union-of-tags runtime engine), so a single-block Case keeps the
    component-0-only behaviour and a multi-block Case forwards a non-density subject to
    ``set_refinement(threshold, variable=)`` (the C++ AmrSystem resolves it per block; a compiled
    block refuses a non-default selector, the honest native boundary).
    """
    criterion = getattr(layout, "refine", None)
    if criterion is not None:
        _apply_refine_criterion(sim, criterion, is_multiblock=n_blocks > 1)


def _apply_refine_criterion(sim, criterion, is_multiblock=False):
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
            _apply_refine_criterion(sim, c, is_multiblock=is_multiblock)
        return
    if not isinstance(criterion, Refine):
        raise TypeError(
            "pops.bind: AMR refine criterion must be a pops.mesh.amr.Refine / TagUnion (got %r)"
            % type(criterion).__name__)
    threshold = criterion.threshold
    if threshold is None:
        raise ValueError("pops.bind: Refine criterion has no threshold "
                         "(use Refine.on(subject).above(value))")
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


def _refine_subject_name(subject):
    """The plain string name of a Refine subject (a string, or an object carrying ``.name``)."""
    if isinstance(subject, str):
        return subject
    name = getattr(subject, "name", None)
    return name if isinstance(name, str) else None


def _is_default_density_subject(subject):
    """True when a Refine subject names the density / component 0 (the single-block default).

    The native single-block AMR refines on component 0 (the historical density), so a Density-role
    or density-named subject maps to ``set_refinement(threshold)`` with no selector; ``None`` (an
    unnamed subject) is treated as the default too. Any other name is a non-default selector that
    the single-block route cannot honor."""
    if subject is None:
        return True
    return subject in ("Density", "density", "rho", "n", "ne")


def _resolve_problem_model(physics):
    """Resolve a block's physics to the model ``compile_problem`` accepts.

    A blackboard :class:`pops.physics.Model` exposes the underlying ``pops.dsl`` engine model
    via ``.dsl`` -- that is what ``compile_problem(model=...)`` wants. A ``pops.model.Module``
    or a raw ``pops.dsl`` model is forwarded as-is (``compile_problem`` lowers a ``Module``
    itself). ``None`` raises, so a block with no physics never reaches codegen.
    """
    if physics is None:
        raise ValueError("pops.compile: the block has no physics model to resolve")
    dsl_model = getattr(physics, "dsl", None)
    if dsl_model is not None:
        return dsl_model
    return physics


def _assemble_instances(problem, initial, models=None):
    """Build the ``sim.install`` instances mapping from the problem's blocks + initial state.

    Each block becomes ``{name: {"model": <model>, "spatial": spatial, "initial": state}}`` -- the
    shape the unified install consumes. The per-block initial state comes from @p initial (keyed by
    block name); an unknown key raises so a typo is not silently dropped.

    @p models, when given (the AMR route), is the ``{block: CompiledModel}`` table from
    ``compile()``'s ``_block_compiled_models``: each block's model becomes its OWN
    ``target='amr_system'`` ``CompiledModel`` (installed via ``add_equation`` -> ``add_native_block``)
    instead of the resolved engine model the System route hands to a compiled time Program. A block
    missing from the table is a compile/bind mismatch and raises.
    """
    if problem is None:
        raise TypeError("pops.bind: the compiled handle carries no problem assembly "
                        "(was it produced by pops.compile?)")
    unknown = sorted(set(initial) - set(problem._blocks))
    if unknown:
        raise ValueError("pops.bind: initial state for unknown block(s) %s; declared blocks: %s"
                         % (unknown, sorted(problem._blocks)))
    instances = {}
    for name, spec in problem._blocks.items():
        if models is not None:
            if name not in models:
                raise ValueError(
                    "pops.bind: block %r has no compiled model on the handle; the AMR handle must "
                    "carry one CompiledModel per block (was it produced by pops.compile?)" % (name,))
            model = models[name]
        else:
            model = _resolve_problem_model(spec["physics"])
        entry = {"model": model, "spatial": spec["spatial"]}
        if name in initial:
            entry["initial"] = initial[name]
        instances[name] = entry
    return instances


def _problem_field_solvers(problem):
    """The {field_name: solver} mapping derived from the problem's field problems."""
    if problem is None:
        return {}
    return {name: fp.solver for name, fp in problem._fields.items() if fp.solver is not None}


__all__ = ["compile", "bind"]
