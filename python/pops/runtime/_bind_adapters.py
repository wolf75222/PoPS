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
        engine = AmrSystem(_amr_config_from_layout(layout))
        _flow_amr_layout(engine, layout, n_blocks=self._n_blocks)
        return engine

    def install(self, engine, compiled, *, instances, params, aux, solvers, cadence, outputs,
                diagnostics):
        # A whole-system compiled time Program (compiled.program is not None, ADC-634) installs on the
        # AMR hierarchy via AmrSystem._install_compiled -> _finish_program_install -> install_program
        # (the ADC-508 per-level driver); a native per-block handle (compiled.program is None) installs
        # with compiled=None, each instance carrying its OWN target='amr_system' CompiledModel wired
        # with add_equation -> add_native_block. Discriminate by the duck-typed getattr signal, exactly
        # like System._install_compiled -- never by the handle class. A native CompiledModel has no
        # .program attribute, so getattr(..., None) selects compiled=None for it.
        program = compiled if getattr(compiled, "program", None) is not None else None
        engine._install_compiled(compiled=program, instances=instances, params=params, aux=aux,
                                 solvers=solvers, cadence=cadence, outputs=outputs,
                                 diagnostics=diagnostics)


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
    return _UniformRuntimeAdapter()


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


def _amr_config_from_layout(layout: Any) -> Any:
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
    flowed as config knobs. Imported lazily so the runtime module stays import-light.
    """
    from pops._bootstrap import AmrSystemConfig
    from pops.mesh.amr import FrozenRegrid, PatchClustering, PatchLayout, RegridEvery

    base = layout.base
    cfg: Any = AmrSystemConfig()
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

    # ADC-616: Berger-Rigoutsos clustering params. None -> the native ClusterParams default (0.7 / 1 /
    # 32), left as 0 on the config so the C++ keeps that default (bit-identical).
    clustering = getattr(layout, "clustering", None)
    if isinstance(clustering, PatchClustering):
        cfg.cluster_min_efficiency = float(clustering.min_efficiency)
        cfg.cluster_min_box_size = int(clustering.min_box_size)
        cfg.cluster_max_box_size = int(clustering.max_box_size)
    elif clustering is not None:
        raise TypeError(
            "pops.bind: AMR.clustering must be a pops.mesh.amr.PatchClustering(...) (got %r)"
            % type(clustering).__name__)
    return cfg


def _flow_amr_layout(sim: Any, layout: Any, n_blocks: Any = 1) -> Any:
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
        _apply_refine_criterion(sim, criterion, is_multiblock=n_blocks > 1)


def _apply_refine_criterion(sim: Any, criterion: Any, is_multiblock: bool = False) -> Any:
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
    if not getattr(criterion, "references_authenticated", False):
        raise ValueError(
            "pops.bind: Refine criterion references were not authenticated by Problem.resolve; "
            "run it through pops.compile(problem, layout=...) instead of attaching a raw or "
            "canonical-looking Handle directly to a compiled/runtime layout")
    threshold = criterion.threshold
    if threshold is None:
        raise ValueError("pops.bind: Refine criterion has no threshold "
                         "(use Refine.on(subject).above(value))")
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
