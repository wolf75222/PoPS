"""AmrSystem : the refined runtime coupler (Spec-4 PR-F composed class).

``AmrSystem`` carries one or several blocks on an AMR hierarchy. Its lines are split into the
``_amr_system_equation`` (add_equation + named-aux), ``_amr_system_io`` (write / checkpoint /
restart), ``_amr_system_program`` (compiled time-Program install / params / cadence, ADC-508)
and ``_amr_system_install`` (the ``pops.bind`` install seam + field-solver / aux helpers)
mixins to satisfy the <=500-line cap ; this module composes them and keeps the constructor + the
native-add_block / coupling / diagnostics glue.
"""
from __future__ import annotations

from typing import Any

from pops._bootstrap import AmrSystemConfig, _AmrSystem
from pops.runtime import threading as _threading
from pops.runtime._lifecycle import (
    FROZEN_STRUCTURAL as _FROZEN_STRUCTURAL, freeze_error as _freeze_error,
    guard_assembling as _guard_assembling, _LifecycleMixin)
from pops.runtime._numeric import native_real
from pops.runtime.bricks import Spatial, Explicit
from pops.runtime.defaults import (
    NEWTON_DEFAULT_ABS_TOL,
    NEWTON_DEFAULT_DAMPING,
    NEWTON_DEFAULT_FAIL_POLICY,
    NEWTON_DEFAULT_FD_EPS,
    NEWTON_DEFAULT_MAX_ITERS,
    NEWTON_DEFAULT_REL_TOL,
)
from pops.runtime._amr_system_equation import _AmrSystemEquation
from pops.runtime._amr_system_install import _AmrSystemInstall
from pops.runtime._amr_system_io import _AmrSystemIO
from pops.runtime._amr_system_program import _AmrSystemProgram
from pops.runtime.profile import PerformanceSummary, Profile


def _profile_payload(system: Any) -> Any:
    """Structured profiler payload when the native extension exposes it, else legacy text."""
    snapshot = getattr(system, "profile_snapshot", None)
    if callable(snapshot):
        return snapshot()
    return system.profile_report()


class _AmrProfileSession:
    """The typed profiling context manager AmrSystem.profile() returns (Spec 5 sec.12.5).

    Mirror of :class:`pops.runtime.system._ProfileSession` for the AMR runtime: ``__enter__``
    resets + enables the native profiler ; ``__exit__`` snapshots the report into a
    :class:`PerformanceSummary` and disables the profiler. ``summary().by_amr_mpi()`` surfaces the
    AMR / MPI phase timings (regrid / fill_boundary / average_down) + counters (criterion 43). Lives
    here rather than importing from system.py to avoid a circular import (system imports amr_system).
    """

    def __init__(self, system: Any, profile: Any) -> None:
        self._system = system
        self._profile = profile
        self._summary = None

    def __enter__(self) -> Any:
        self._system.reset_profiling()
        self._system.enable_profiling()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> Any:
        self._summary = PerformanceSummary(_profile_payload(self._system), self._profile)
        self._system.disable_profiling()
        return False

    def summary(self) -> Any:
        """Return a :class:`PerformanceSummary` (live report inside the block, snapshot after)."""
        if self._summary is not None:
            return self._summary
        return PerformanceSummary(_profile_payload(self._system), self._profile)


class AmrSystem(_AmrSystemEquation, _AmrSystemInstall, _AmrSystemIO, _AmrSystemProgram,
                _LifecycleMixin):
    """Refined counterpart of System : one or SEVERAL blocks carried on an AMR hierarchy.

    SINGLE-BLOCK (1 add_block) : historical AmrCouplerMP path (dynamic regrid, reflux). MULTI-BLOCK
    (>= 2 add_block) : N blocks co-located on ONE SHARED AMR hierarchy (AmrRuntime engine),
    SYSTEM Poisson with co-located SUMMED right-hand side (Sum_b q_b n_b), conservation PER BLOCK. The
    blocks may have DIFFERENT SPATIAL SCHEMES, a per-block TEMPORAL TREATMENT (explicit /
    imex), MULTIRATE (substeps / stride), COUPLED inter-species SOURCES and the multi-block production
    DSL. In multi-block the block NAME indexes set_density(name) / mass(name) / density(name).

    UNION-OF-TAGS REGRID (multi-block + regrid_every > 0) : the shared hierarchy is re-gridded from
    the UNION of the tags of all blocks. Two criteria compose (cell-by-cell OR) :

    - PER-BLOCK VARIABLE (set_refinement(threshold, variable=, role=)) : refine where the SELECTED
      variable of a block exceeds threshold. Default = component 0 (historical density), bit-identical ;
      ADC-296 lets you select it per block by name (variable=) or physical role (role=), resolved against
      the block's conserved variables (a block lacking the name/role raises, no silent component-0
      fallback). Non-default selector is multi-block only (mono-block / compiled .so : component 0 only) ;
    - ``grad phi`` (set_phi_refinement(grad_threshold)) : refine where the norm of the gradient of the
      electrostatic potential exceeds grad_threshold (diocotron ring edge). Disabled by default
      (grad_threshold <= 0). MULTI-BLOCK only.

    regrid_every == 0 -> FROZEN hierarchy (regrid never called, bit-identical).
    """

    def __init__(self, config: Any = None, **cfg_kw: Any) -> None:
        if config is None:
            config = AmrSystemConfig()
            for k, v in cfg_kw.items():
                setattr(config, k, v)
        # cf. System.__init__ : _AmrSystem(config) triggers the Kokkos init (lazy). set_threads
        # has no more effect after this point.
        _threading._first_system_built = True
        self._s = _AmrSystem(config)
        self._L = float(config.L)  # side of [0, L]^2 (for patch_rectangles : index -> physical)
        # Regrid cadence (checkpoint/restart ADC-65) : a BIT-IDENTICAL resume requires regrid_every == 0
        # (otherwise the post-restart regrid would re-diverge the hierarchy). Memorized for the restart guard.
        self._regrid_every = int(config.regrid_every)
        # ADC-291: block name -> {aux field name -> channel component}, filled by add_equation from a
        # CompiledModel.aux_extra_names (component of the k-th name = AUX_NAMED_BASE + k). Drives
        # set_aux_field(block, name, array). Empty for blocks without a named aux field. Mirror of
        # System._aux_field_index.
        self._aux_field_index = {}
        # RUNTIME FREEZE LIFECYCLE (ADC-592, parity with System): "assembling" until _finalize_bind
        # flips it to "bound" (the LAST act of _install_compiled). The Python flag enforces the freeze
        # even under a prebuilt .so with no native mark_bound; _bound_snapshot is the BoundSnapshot of
        # what was bound (None until bind).
        self._lifecycle = "assembling"
        self._bound_snapshot = None
        self._last_run_manifest = None
        self._last_run_identity = None
        self._last_restart_identity = None
        self._program_cadence_cfl = None

    def run(self, t_end, cfl=None, max_steps=1_000_000, output_dir=None, strategy=None):
        """Advance up to ``t_end``; RuntimeInstance alone publishes ConsumerGraph effects."""
        from pops.runtime._step_strategy import (
            AdaptiveCFL, resolve_run_strategy, run_step_attempt)
        strategy = resolve_run_strategy(self, strategy, cfl)
        manifest_cfl = strategy.cfl if isinstance(strategy, AdaptiveCFL) else 0.0
        from pops.runtime._run_manifest import begin_run
        begin_run(self, t_end=t_end, cfl=manifest_cfl, max_steps=max_steps, output_dir=output_dir)
        steps = 0
        while self._s.time() < t_end and steps < max_steps:
            run_step_attempt(self, self._s, strategy, t_end=float(t_end))
            steps += 1
        return steps

    def profile(self, profile: Any = None) -> Any:
        """Typed AMR / MPI profiling context manager (Spec 5 sec.12.5, criterion 43).

        Usage::

            sim.set_refinement(threshold)  # regrid_every > 0 in the config
            with sim.profile(pops.Profile.Basic()) as prof:
                for _ in range(n_steps):
                    sim.step_cfl(0.4)
            print(prof.summary().by_amr_mpi())  # regrid / fill_boundary / average_down timings

        @p profile is a :class:`pops.Profile` level ; with no argument it comes from ``POPS_PROFILE``
        (unset / ``off`` -> Basic()). The manager enables the native AMR profiler on entry and
        disables it on exit (off-by-default contract). ``prof.summary().by_amr_mpi()`` surfaces the
        AMR phase timings + counters as soon as a regrid / solve fired under the multi-block engine.
        """
        if profile is None:
            profile = Profile.from_env(default=Profile.Basic())
        elif not isinstance(profile, Profile):
            raise TypeError(
                "AmrSystem.profile: expected a pops.Profile (Profile.Basic()/Advanced()), got %r"
                % type(profile).__name__)
        return _AmrProfileSession(self, profile)

    def patch_rectangles(self) -> Any:
        """Physical rectangles (x0, y0, width, height) of the current fine patches, in [0, L]^2.

        Converts patch_boxes() (index space, inclusive corners) into physical coordinates. The level
        spacing is dx = L / (n << level) (ratio 2 per level) ; a patch [ilo..ihi] x [jlo..jhi]
        covers (ihi - ilo + 1) cells in x from x0 = ilo * dx (and likewise in y). Grid convention
        ne[j, i] -> index 0 = x (i), index 1 = y (j), consistent with density() and an imshow
        with extent [0, L, 0, L]. Convenient to plot the REAL patches (e.g. matplotlib Rectangle) without
        rebuilding a density proxy. Returns a list of (x0, y0, w, h), one per fine patch (all
        fine levels combined). Query (between steps) : triggers the lazy build like
        n_patches(), no cost on the hot path.
        """
        n, L = self._s.nx(), self._L
        rects = []
        for level, ilo, jlo, ihi, jhi in self._s.patch_boxes():
            dx = L / (n << level)
            rects.append((ilo * dx, jlo * dx, (ihi - ilo + 1) * dx, (jhi - jlo + 1) * dx))
        return rects

    def coarse_local_boxes(self) -> Any:
        """Number of coarse (base) boxes owned by this MPI rank (ADC-319 diagnostic).

        The base level is a MultiFab whose boxes are spread across ranks by a DistributionMapping.
        Returns this rank's owned-fab count (level-0 local_size()). With distribute_coarse=True the base
        is split into several boxes round-robin, so each rank owns a strict subset and the coarse
        transport is distributed; a replicated or single-box base owns the full count on every rank.
        Compare with coarse_total_boxes() and pops.n_ranks() to confirm MPI strong-scaling of the base.
        Triggers the lazy build like n_patches().
        """
        return self._s.coarse_local_boxes()

    def coarse_total_boxes(self) -> Any:
        """Total number of coarse (base) boxes across all ranks (ADC-319 diagnostic).

        Identical on every rank (BoxArray size, no communication). With distribute_coarse=True this is
        the number of round-robin base tiles; with a single-box or replicated base it is 1. A rank
        distributes the coarse transport when coarse_local_boxes() < coarse_total_boxes().
        Triggers the lazy build like n_patches().
        """
        return self._s.coarse_total_boxes()

    def add_block(self, name: Any, model: Any, spatial: Any = None, time: Any = None) -> Any:
        """Installs an evolved block composed of NATIVE BRICKS on the shared AMR hierarchy.

        Low-level runtime seam. The documented PUBLIC path is the typed ``pops.Case`` assembly
        lowered by ``pops.compile(problem, layout=AMR(...))`` and wired by ``pops.bind`` (which calls
        this internally); ``add_block`` stays for that seam and the tests.

        Refined counterpart of System.add_block. The 1st add_block opens the single-block path
        (AmrCouplerMP : dynamic regrid, reflux) ; each subsequent add_block co-locates one more block
        on THE SAME hierarchy (AmrRuntime engine, system Poisson with summed right-hand side).
        In multi-block the name indexes set_density(name) / mass(name) / density(name). The arguments
        are marshaled to the C++ facade (AmrSystem::add_block), which validates the block against the model.
        For a compiled DSL model (.so) or a dispatch on the model type, use add_equation.

        @param name unique name of the block.
        @param model an pops.Model(...) (ModelSpec : composed native bricks).
        @param spatial spatial discretization, an pops.Spatial(...) / pops.FiniteVolume(...) (default
            minmod + rusanov + conservative). Limiter (none / minmod / vanleer / weno5 ; weno5 = 3
            ghosts, the coupler allocates its levels at Limiter::n_ghost and the regrid inherits n_grow()),
            Riemann flux (rusanov / hll / hllc / roe) and reconstructed variables
            (conservative / primitive).
        @param time temporal treatment, an pops.Explicit (default) / pops.IMEX / pops.SourceImplicit.
            Carries substeps, stride (multirate hold-then-catch-up), the implicit mask (implicit_vars
            / implicit_roles) and the Newton options, threaded to the C++. newton_diagnostics is
            wired in native multi-block and rejected at the C++ build in single-block (the coupler does not
            aggregate a report).
        spatial.positivity_floor > 0 (ADC-259) floors the Density-role face states AND the
        coarse-fine fine ghost means to >= floor on the AMR transport (Zhang-Shu, parity with the
        uniform System). Guarantee = face / ghost-state Density positivity only (order-1 fallback),
        NOT updated-mean nor pressure positivity. A model without a Density role rejects it at the
        first step. The COMPILED .so path carries it too now (ADC-322): a loader regenerated against
        the current headers marshals the floor (add_equation on a CompiledModel, add_native_block).
        """
        _guard_assembling(self, "add_block")  # frozen once pops.bind completes (ADC-592)
        spatial = spatial if spatial is not None else Spatial()
        time = time if time is not None else Explicit()
        # positivity_floor (ADC-259) IS now wired on the AMR transport (Density-role face states +
        # C/F fine ghost means). Threaded to AmrSystem::add_block below; the compiled .so path carries
        # it too (ADC-322, regenerated loader). The C++ side rejects it on a model without a Density role.
        # wave_speed_cache (ADC-199) is NOT wired on the AMR path (AmrSystem::add_block does not
        # transport it) : explicit rejection rather than a silently ignored cache.
        if getattr(spatial, "wave_speed_cache", False):
            raise ValueError(
                "AmrSystem.add_block : wave_speed_cache not supported on the AMR path (separate "
                "work item) ; remove wave_speed_cache, or declare layout=Uniform(...) on the "
                "pops.Case (the uniform route wires the cache).")
        # weno_epsilon (ADC-645) is NOT wired on the AMR transport (AmrSystem::add_block does not
        # transport it; the AMR advance keeps the default kWenoEpsilon): explicit rejection rather
        # than a silently ignored regulariser. The uniform System path wires it.
        if getattr(spatial, "weno_epsilon", None) is not None:
            raise ValueError(
                "AmrSystem.add_block : weno_epsilon (WENO5(epsilon=...)) not supported on the AMR "
                "path (separate work item) ; remove epsilon, or declare layout=Uniform(...) on the "
                "pops.Case (the uniform route wires it).")
        # We thread substeps/stride (multirate, capstone iv), the partial IMEX mask, the Newton OPTIONS
        # AND newton_diagnostics (wave 3, settle). Resolved / validated on the C++ side (AmrSystem::add_block)
        # against the block names/roles : empty -> full backward-Euler. The options are wired in single-block
        # (coupler) AND multi-block ; newton_diagnostics is wired in native MULTI-BLOCK and REJECTED at the
        # C++ build in single-block (the coupler does not aggregate a report) -- no facade-side filtering here
        # (the facade does not yet know the total number of blocks : the single/multi decision is at build).
        self._s.add_block(name, model, spatial.limiter, spatial.flux, spatial.recon, time.kind,
                          getattr(time, "substeps", 1), getattr(time, "stride", 1),
                          getattr(time, "implicit_vars", []), getattr(time, "implicit_roles", []),
                          getattr(time, "newton_max_iters", NEWTON_DEFAULT_MAX_ITERS),
                          native_real(getattr(time, "newton_rel_tol", NEWTON_DEFAULT_REL_TOL),
                                      where="AmrSystem.add_block.newton_rel_tol"),
                          native_real(getattr(time, "newton_abs_tol", NEWTON_DEFAULT_ABS_TOL),
                                      where="AmrSystem.add_block.newton_abs_tol"),
                          native_real(getattr(time, "newton_fd_eps", NEWTON_DEFAULT_FD_EPS),
                                      where="AmrSystem.add_block.newton_fd_eps"),
                          native_real(getattr(time, "newton_damping", NEWTON_DEFAULT_DAMPING),
                                      where="AmrSystem.add_block.newton_damping"),
                          getattr(time, "newton_fail_policy", NEWTON_DEFAULT_FAIL_POLICY),
                          getattr(time, "newton_diagnostics", False),
                          native_real(getattr(spatial, "positivity_floor", 0.0),
                                      where="AmrSystem.add_block.positivity_floor"))

    def field(self, name: Any) -> Any:
        """Return the solved potential of a NAMED elliptic field (ADC-428) as an (n, n) array.

        Read-back of a second elliptic field declared via m.elliptic_field and lowered on the AMR layout:
        solves the hierarchy fields if needed (so it is current even before any step) then reads the
        field's coarse potential. AMR counterpart of reading System.aux_field(block, name) for an elliptic
        field. @throws if the field is unregistered (or the system runs the single-block coupler, which
        carries no named field)."""
        return self._s.named_field_values(name)

    def add_coupling(self, coupling: Any) -> Any:
        """Add a generic inter-species COUPLED SOURCE (pops.dsl.CoupledSource(...).compile(...))
        on the SHARED AMR hierarchy (MULTI-BLOCK), refined counterpart of System.add_coupling. The source
        is transported as bytecode and interpreted on the C++ side (AmrSystem.add_coupled_source; no
        per-cell Python callback). The coupling frequency (CoupledSource.frequency) is honored:
        constant -> dt bound dt <= cfl/mu; Expr -> PER-CELL frequency mu(U) evaluated on the COARSE grid at
        each step_cfl (the freq_prog_* vectors are forwarded). Must be called BEFORE the first
        step (the source is frozen then injected at the lazy build of the runtime engine)."""
        _guard_assembling(self, "add_coupling")  # frozen once pops.bind completes (ADC-592)
        # Late import (the multispecies module imports this package: avoid the cycle).
        from pops.physics.multispecies import CompiledCoupledSource
        from pops.physics.coupling_presets import lower_named_coupling, coupling_operator_args

        if isinstance(coupling, CompiledCoupledSource):
            args = coupling_operator_args(coupling, getattr(coupling, "conserved_roles", ()),
                                          getattr(coupling, "created_roles", ()))
            self._s.add_coupling_operator(*args)
            return
        # Named preset (ADC-595): lower to the generic coupled source (bit-identical to System), so the
        # AMR path gains the named couplings as typed operators too. ThermalExchange needs a per-block
        # gamma; AmrSystem does not expose block_gamma, so a preset that requires it raises clearly.
        preset = lower_named_coupling(coupling, self._amr_block_gamma)
        if preset is None:
            raise TypeError("AmrSystem.add_coupling expects pops.Ionization / Collision / "
                            "ThermalExchange or a CompiledCoupledSource "
                            "(pops.dsl.CoupledSource(...).compile(...))")
        preset.source.verify_declared_contract(conserved=preset.conserved, created=preset.created)
        args = coupling_operator_args(preset.source.compile(), preset.conserved, preset.created,
                                      frequency=preset.frequency)
        self._s.add_coupling_operator(*args)

    def _amr_block_gamma(self, name: Any) -> Any:
        """Per-block adiabatic index for the ThermalExchange preset (ADC-595). AmrSystem does not expose
        a block_gamma accessor, so a ThermalExchange on AMR raises a clear error pointing at the generic
        CoupledSource path (build the pressure closure with an explicit gamma via pops.dsl.CoupledSource)."""
        raise NotImplementedError(
            "AmrSystem: the ThermalExchange preset needs a per-block gamma, which AMR does not expose; "
            "author the thermal exchange as a generic pops.dsl.CoupledSource with an explicit gamma "
            "param, or use it on a uniform System.")

    @property
    def amr(self) -> Any:
        """The live AMR runtime inspection handle (Spec 5 sec.8.12), an
        :class:`pops.runtime.amr.AmrRuntimeView`.

        Bound to THIS built hierarchy: ``sim.amr.patch_table()`` /
        ``sim.amr.hierarchy_snapshot()`` / ``sim.amr.explain_regrid()`` /
        ``explain_ghosts()`` / ``explain_reflux()`` / ``explain_checkpoint()`` return short, inert
        reports of the patches that actually exist, the regrid cadence in force, and the
        ghost / reflux / checkpoint route limitations. The view READS the runtime (the box
        accessors + the retained config); it builds / allocates / steps NOTHING.

        ``System.amr`` does not exist: the inspection surface is AMR-specific (a uniform System
        carries no hierarchy). Use ``pops.inspect(layout)`` for the STATIC authoring report.
        """
        from pops.runtime.amr import AmrRuntimeView  # lazy: keeps the constructor import-light.

        return AmrRuntimeView(self)

    def __str__(self) -> Any:
        """Short, array-free summary: block names on the AMR hierarchy (Spec 5 sec.12.1).

        Field/patch data stays out of the summary -- it prints the block registry only.
        """
        try:
            blocks = list(self._s.block_names())
        except Exception:  # pragma: no cover - defensive: _AmrSystem not fully wired
            blocks = []
        return "AmrSystem(blocks=%s)" % (blocks,)

    def explain_bind(self, compiled: Any) -> Any:
        """A printable :class:`pops.codegen.inspect_report.BindReport` of @p compiled vs this AMR sim
        (Spec 5 sec.12.1, criterion #15). INERT parity with ``System.explain_bind``: reads the
        artifact's DECLARED bind inputs (``compiled.arguments()``) and the blocks / named aux wired on
        this AmrSystem, then reuses ADC-463 :func:`collect_missing_arguments` to report PROVIDED vs
        still-REQUIRED per group. It binds nothing and mutates nothing -- a read-only bind plan."""
        from pops.codegen.inspect_report import build_bind_report
        return build_bind_report(self, compiled)

    def inspect(self) -> Any:
        """Structured, array-free AMR runtime inspection report (ADC-591)."""
        from pops.runtime.inspection import build_runtime_inspection
        return build_runtime_inspection(self, runtime="amr_system")

    def program_report(self) -> Any:
        """Structured report of the compiled-Program runtime subsystem (ADC-594).

        Same value object as ``System.program_report`` -- the SHARED Program subsystem (the AMR runtime
        uses the common subset: no dt bound, no scheduler cache / history rings wired, so those
        sections stay empty). Metadata only; installed=False with empty sections on a runtime with no
        program installed."""
        from pops.runtime.program_report import build_program_report
        return build_program_report(self)

    def __getattr__(self, attr: Any) -> Any:
        # RUNTIME FREEZE (ADC-592): once bound, refuse a native STRUCTURAL setter reached through the
        # passthrough (sim._engine.set_refinement / install_program / ...) with the bind-vocabulary
        # RuntimeError, so the bypass is closed even under a prebuilt .so whose C++ setters are not yet
        # frozen. The data / param / diagnostic passthrough is untouched.
        if attr in _FROZEN_STRUCTURAL and getattr(self, "_lifecycle", "assembling") != "assembling":
            raise _freeze_error(attr)
        return getattr(self._s, attr)
