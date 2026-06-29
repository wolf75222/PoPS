"""pops.codegen.orchestration -- thin pops.compile / pops.bind over the existing runtime.

These are the Spec 5 sec.11 lowering entry points for a :class:`pops.case.Case`:

* :func:`compile` validates the assembly, picks the compile target from the LAYOUT
  (``Uniform`` -> ``"system"``, ``AMR`` -> ``"amr_system"``; no user ``target=`` string),
  resolves each block's physics to the model the runtime wants, and compiles the complete
  ``model + Program`` problem through ``compile_problem``. Uniform and AMR use the same public
  contract; the layout only selects the generated C++ install ABI.
* :func:`bind` dispatches ``System`` vs ``AmrSystem`` from the carried target, assembles the
  per-instance state mapping, and calls the INTERNAL ``sim._install_compiled(compiled, instances=,
  ...)`` seam (``pops.runtime._system_unified_install``). ``bind`` is the public entry point; the
  ``_install_compiled`` seam is undocumented / low-level.

There is no per-block AMR fallback here. If AMR needs a feature, the Program/codegen/runtime AMR
route must implement it rather than bypassing the complete problem contract.

Import-graph rule (Spec 4 / sec.4): ``codegen`` may import only ir / model / physics / time /
lib at module scope. The runtime (System / AmrSystem), mesh (AMR) and case types are pulled
LAZILY inside the function bodies, so this module adds no forbidden cross-layer edge.
"""


def compile(problem, backend=None, time=None, **kwargs):
    """Lower a :class:`pops.case.Case` to a compiled handle.

    Validates @p problem, derives the compile target from its LAYOUT (``Uniform`` -> system,
    ``AMR`` -> amr_system) and lowers via the route the target selects:

    Uniform and AMR both compile the whole time ``Program`` once with ``compile_problem``.
    ``compile_problem`` receives ``layout=problem.layout`` and therefore emits the correct C++
    install ABI. The time scheme is explicit here: @p time (a ``pops.time.Program``), else
    ``problem.time(...)``; a missing scheme raises (no silent default). The handle carries
    ``_block_models`` so bind installs each block with its own resolved model.

    Args:
        problem: The :class:`pops.case.Case` assembly to lower.
        backend: Typed codegen backend descriptor (default ``Production()`` inside
            ``compile_problem``).
        time: The ``pops.time.Program`` time scheme; falls back to ``problem._time``.
        **kwargs: compile-driver options only: ``so_path``, ``force``, ``cxx``, ``include``,
            ``std``, ``debug`` and ``libraries``. Route selectors such as ``target=`` are not
            accepted; the layout chooses the runtime ABI.

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

    allowed_kwargs = {"so_path", "force", "cxx", "include", "std", "debug", "libraries"}
    unknown = sorted(set(kwargs) - allowed_kwargs)
    if unknown:
        if "target" in unknown:
            raise TypeError(
                "pops.compile: target= is not a public API; choose the runtime with "
                "layout=Uniform(mesh) or layout=AMR(...), not a string target")
        raise TypeError(
            "pops.compile: unknown compile option(s) %s; allowed options are %s"
            % (unknown, sorted(allowed_kwargs)))

    time = time if time is not None else problem._time
    if time is None:
        raise ValueError(
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
    compiled = compile_problem(time=time, model=model, backend=backend, layout=problem.layout,
                               **kwargs)
    compiled._problem = problem
    compiled._target = target
    compiled._block_models = block_models
    # Carry the AMR layout so bind() can rebuild the AmrSystemConfig (n / L / periodic / regrid /
    # patch settings) and flow the typed refinement + field problem onto the AmrSystem. None for a
    # Uniform layout (System bind reads no layout); set only on the AMR route.
    compiled._layout = problem.layout if is_amr else None
    return compiled


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
    ``target='amr_system'`` the same compiled problem handle is passed to the AMR install seam.
    Returns the bound simulation (the ``System`` / ``AmrSystem`` is the Simulation facade for now):
    call ``sim.run(...)`` to advance it.

    Args:
        compiled: A ``CompiledProblem`` from :func:`compile` (carries ``_problem`` / ``_target``).
        initial_state: dict {block_name: array} of per-block initial state (alias: @p state).
        state: Alias for @p initial_state (only one may be given).
        params: dict {param_name: value} of runtime parameter overrides.
        aux: dict {aux_name: array} of static aux inputs.
        solvers: dict {field: solver} overriding the per-field solvers from the problem.
        cadence: optional internal compiled-Program macro-step cadence.

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
        # AMR installs the same compiled problem handle as Uniform. The block instances carry their
        # resolved physical models; the compiled Program .so is installed afterward by
        # AmrSystem._install_compiled through the native AMR program entry.
        instances = _assemble_instances(problem, initial or {},
                                        models=getattr(compiled, "_block_models", None))
        sim._install_compiled(compiled=compiled, instances=instances, params=params or {},
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

    @p n_blocks is the declared block count. Non-density subjects are forwarded as typed
    variable selectors for both single-block and multi-block layouts; the C++ AMR runtime resolves
    them against the block's conservative variables before it builds the tag predicate.
    """
    criterion = getattr(layout, "refine", None)
    if criterion is not None:
        _apply_refine_criterion(sim, criterion, is_multiblock=n_blocks > 1)


def _apply_refine_criterion(sim, criterion, is_multiblock=False):
    """Lower one typed refinement criterion to set_refinement / set_phi_refinement on @p sim.

    A ``Refine`` whose predicate is a gradient on the potential (``phi`` / ``grad phi``) lowers to
    ``set_phi_refinement(threshold)``; a density subject lowers to ``set_refinement(threshold)`` on
    component 0. A non-density subject lowers to ``set_refinement(threshold, variable=subject)`` and
    is resolved by the native C++ builder. A ``TagUnion`` lowers to each call in turn. A criterion
    that is neither raises a clear error rather than silently dropping it."""
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
    # Forward the subject as the variable name. The C++ AmrSystem::set_refinement resolves it against
    # each block's conserved variables, in both single-block and multi-block AMR.
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

    A blackboard :class:`pops.physics.Model` lowers to a ``pops.model.Module`` through
    ``to_module()`` / ``lower()``; the private codegen engine is not a public escape hatch.
    A ``pops.model.Module`` or a raw low-level model is forwarded as-is. ``None`` raises,
    so a block with no physics never reaches codegen.
    """
    if physics is None:
        raise ValueError("pops.compile: the block has no physics model to resolve")
    to_module = getattr(physics, "to_module", None)
    if callable(to_module):
        return to_module()
    lower = getattr(physics, "lower", None)
    if callable(lower):
        return lower()
    return physics


def _assemble_instances(problem, initial, models=None):
    """Build the ``sim.install`` instances mapping from the problem's blocks + initial state.

    Each block becomes ``{name: {"model": <model>, "spatial": spatial, "initial": state}}`` -- the
    shape the unified install consumes. The per-block initial state comes from @p initial (keyed by
    block name); an unknown key raises so a typo is not silently dropped.

    @p models, when given, is the ``{block: resolved_model}`` table from ``compile()``'s
    ``_block_models``. A block missing from the table is a compile/bind mismatch and raises.
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
                    "pops.bind: block %r has no resolved model on the compiled handle; "
                    "was it produced by pops.compile?" % (name,))
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
