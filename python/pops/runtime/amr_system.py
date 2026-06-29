"""AmrSystem : the refined runtime coupler (Spec-4 PR-F composed class).

``AmrSystem`` carries one or several blocks on an AMR hierarchy. Its lines are split into private
runtime mixins for block lowering, IO and compiled-problem attach / params / cadence. The public
runtime route is ``sim.install(compiled, ...)``.
"""

from pops._bootstrap import AmrSystemConfig, _AmrSystem
from pops.runtime import threading as _threading
from pops.runtime.bricks import Spatial, Explicit, Split
from pops.runtime._amr_system_equation import _AmrSystemEquation
from pops.runtime._amr_system_io import _AmrSystemIO
from pops.runtime._amr_system_program import _AmrSystemProgram
from pops.runtime._system_unified_install import validate_install_arguments
from pops.runtime.profile import PerformanceSummary, Profile


class _AmrProfileSession:
    """Profiling context manager for the AMR runtime."""

    def __init__(self, system, profile):
        self._system = system
        self._profile = profile
        self._summary = None

    def __enter__(self):
        self._system.reset_profiling()
        self._system.enable_profiling()
        return self

    def __exit__(self, exc_type, exc, tb):
        self._summary = PerformanceSummary(self._system.profile_report(), self._profile)
        self._system.disable_profiling()
        return False

    def summary(self):
        """Return a :class:`PerformanceSummary` (live report inside the block, snapshot after)."""
        if self._summary is not None:
            return self._summary
        return PerformanceSummary(self._system.profile_report(), self._profile)


class AmrSystem(_AmrSystemEquation, _AmrSystemIO, _AmrSystemProgram):
    """Refined counterpart of System: one or several blocks on a shared AMR hierarchy."""

    def __init__(self, config=None, **cfg_kw):
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
        # ADC-291: block name -> {aux field name -> channel component}, filled by sim.install from a
        # CompiledModel.aux_extra_names (component of the k-th name = AUX_NAMED_BASE + k). Drives
        # set_aux_field(block, name, array). Empty for blocks without a named aux field. Mirror of
        # System._aux_field_index.
        self._aux_field_index = {}
        self._program_cadence_cfl = None
        self._output_policies = []

    def run(self, t_end, cfl=None, max_steps=1_000_000, output_dir=None):
        """Advance up to ``t_end`` by AMR CFL steps and fire output/checkpoint policies.

        Mirrors :meth:`pops.runtime.system.System.run`: Python only orchestrates macro-steps by
        calling the C++ ``step_cfl`` method; all cell work remains in the native runtime. When a
        compiled artifact cadence pins ``cfl="program"``, the wrapper calls ``step_cfl(1.0)`` and the
        installed artifact's dt-bound hook tightens the step inside C++.
        """
        if cfl is None:
            cfl = self._program_cadence_cfl if self._program_cadence_cfl is not None else 0.4
        if cfl == "program":
            cfl = 1.0
        policies = getattr(self, "_output_policies", [])
        out_dir = output_dir if output_dir is not None else "."
        steps = 0
        if policies:
            self._fire_outputs(policies, steps, out_dir, phase="start")
        while self.time() < t_end and steps < max_steps:
            self.step_cfl(cfl)
            steps += 1
            if policies:
                self._fire_outputs(policies, steps, out_dir, phase="step")
        if policies:
            self._fire_outputs(policies, steps, out_dir, phase="end")
        return steps

    def _fire_outputs(self, policies, step, output_dir, phase="step"):
        from pops.runtime._output_driver import fire_output_policies
        return fire_output_policies(self, policies, step, output_dir, phase=phase)

    def profile(self, profile=None):
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

    def patch_rectangles(self):
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

    def coarse_local_boxes(self):
        """Number of coarse (base) boxes owned by this MPI rank (ADC-319 diagnostic).

        The base level is a MultiFab whose boxes are spread across ranks by a DistributionMapping.
        Returns this rank's owned-fab count (level-0 local_size()). With distribute_coarse=True the base
        is split into several boxes round-robin, so each rank owns a strict subset and the coarse
        transport is distributed; a replicated or single-box base owns the full count on every rank.
        Compare with coarse_total_boxes() and pops.n_ranks() to confirm MPI strong-scaling of the base.
        Triggers the lazy build like n_patches().
        """
        return self._s.coarse_local_boxes()

    def coarse_total_boxes(self):
        """Total number of coarse (base) boxes across all ranks (ADC-319 diagnostic).

        Identical on every rank (BoxArray size, no communication). With distribute_coarse=True this is
        the number of round-robin base tiles; with a single-box or replicated base it is 1. A rank
        distributes the coarse transport when coarse_local_boxes() < coarse_total_boxes().
        Triggers the lazy build like n_patches().
        """
        return self._s.coarse_total_boxes()

    def _add_block(self, name, model, spatial=None, time=None):
        """Installs an evolved block composed of NATIVE BRICKS on the shared AMR hierarchy.

        Low-level runtime seam. The documented PUBLIC path is a typed model/program compiled by
        ``pops.compile_problem(...)`` and wired with ``sim.install(compiled, ...)``. This helper is
        private plumbing for that install path.

        Refined private counterpart of System._add_block. The first block opens the single-block path
        (AmrCouplerMP : dynamic regrid, reflux) ; each subsequent block co-locates one more block
        on THE SAME hierarchy (AmrRuntime engine, system Poisson with summed right-hand side).
        In multi-block the name indexes set_density(name) / mass(name) / density(name). The arguments
        are marshaled to the C++ facade (AmrSystem::add_block), which validates the block against the model.
        For a compiled model artifact or a dispatch on the model type, use the private
        ``_add_equation`` seam through ``sim.install``.

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
        @throws TypeError if time is an pops.Split / pops.Strang (Schur-condensed source stage) :
            not wired by this private primitive; use a compiled split problem installed through
            ``sim.install``.
        spatial.positivity_floor > 0 (ADC-259) floors the Density-role face states AND the
        coarse-fine fine ghost means to >= floor on the AMR transport (Zhang-Shu, parity with the
        uniform System). Guarantee = face / ghost-state Density positivity only (order-1 fallback),
        NOT updated-mean nor pressure positivity. A model without a Density role rejects it at the
        first step. The COMPILED .so path carries it too now (ADC-322): a loader regenerated against
        the current headers marshals the floor through the private compiled-model block seam.
        """
        spatial = spatial if spatial is not None else Spatial()
        time = time if time is not None else Explicit()
        # pops.Split / pops.Strang (Schur-condensed source stage) is only wired by the private
        # equation seam (which
        # connects set_source_stage + set_time_scheme AFTER adding the block) : we reject it HERE rather
        # than playing only the transport and SILENTLY LOSING the source (same guard as System.add_block).
        if isinstance(time, Split):
            raise TypeError(
                "AmrSystem._add_block: pops.Split / pops.Strang (Schur-condensed source stage) is "
                "not wired by this private native primitive; use sim.install(compiled, ...) with a "
                "compiled problem artifact carrying the split program.")
        # positivity_floor (ADC-259) IS now wired on the AMR transport (Density-role face states +
        # C/F fine ghost means). Threaded to AmrSystem::add_block below; the compiled .so path carries
        # it too (ADC-322, regenerated loader). The C++ side rejects it on a model without a Density role.
        # wave_speed_cache (ADC-199) is NOT wired on the AMR path (AmrSystem::add_block does not
        # transport it) : explicit rejection rather than a silently ignored cache.
        if getattr(spatial, "wave_speed_cache", False):
            raise ValueError(
                "AmrSystem.add_block : wave_speed_cache not supported on the AMR path (separate "
                "work item) ; remove wave_speed_cache or use the uniform System.")
        # We thread substeps/stride (multirate, capstone iv), the partial IMEX mask, the Newton OPTIONS
        # AND newton_diagnostics (wave 3, settle). Resolved / validated on the C++ side (AmrSystem::add_block)
        # against the block names/roles : empty -> full backward-Euler. The options are wired in single-block
        # (coupler) AND multi-block ; newton_diagnostics is wired in native MULTI-BLOCK and REJECTED at the
        # C++ build in single-block (the coupler does not aggregate a report) -- no facade-side filtering here
        # (the facade does not yet know the total number of blocks : the single/multi decision is at build).
        self._s.add_block(name, model, spatial.limiter, spatial.flux, spatial.recon, time.kind,
                          getattr(time, "substeps", 1), getattr(time, "stride", 1),
                          getattr(time, "implicit_vars", []), getattr(time, "implicit_roles", []),
                          getattr(time, "newton_max_iters", 2),
                          getattr(time, "newton_rel_tol", 0.0),
                          getattr(time, "newton_abs_tol", 0.0),
                          getattr(time, "newton_fd_eps", 1e-7),
                          getattr(time, "newton_damping", 1.0),
                          getattr(time, "newton_fail_policy", "none"),
                          getattr(time, "newton_diagnostics", False),
                          getattr(spatial, "positivity_floor", 0.0))

    def _install_compiled(self, compiled=None, *, instances=None, params=None, aux=None,
                          solvers=None, cadence=None, outputs=None):
        """Shared AMR install seam for native blocks and compiled problem handles."""
        instances = instances or {}
        params = params or {}
        aux = aux or {}
        solvers = solvers or {}

        # (0) EARLY VALIDATION (shared with System._install_compiled): reject a compiled install missing a
        # required declared argument BEFORE any native mutation. Inert (reads arguments() metadata).
        validate_install_arguments(self, compiled, instances, params, aux, solvers)

        # COMPILED vs NATIVE. COMPILED: `compiled` carries a combined .so artifact attached in step 5
        # plus a physical Module (the per-block model an instance falls back on). NATIVE:
        # `compiled is None` -- each instance carries its own native model. Validate the handle up
        # front, BEFORE any native mutation (no half-configured AMR hierarchy).
        so_path = None
        compiled_model = None
        if compiled is not None:
            so_path = getattr(compiled, "so_path", None)
            if so_path is None:
                raise TypeError(
                    "sim.install: compiled handle has no .so_path (got %r); pass a CompiledProblem "
                    "returned by pops.compile_problem(...)." % type(compiled).__name__)
            compiled_model = getattr(compiled, "model", None)
        # (1) FIELD SOLVERS first (parity with System): configure solvers before adding blocks and
        # before attaching the compiled artifact because native requirement validation reads the
        # configured solver. Declared named elliptic fields (ADC-428), collected from the per-instance
        # models, widen the accepted solver-field set beyond the default Poisson names: a solver
        # selection for a model-declared named field routes (the native loader wired
        # register_elliptic_field), a typo is rejected against the declared set.
        declared_fields = self._declared_elliptic_fields(instances)
        for field, solver_brick in solvers.items():
            self._install_solver(field, solver_brick, declared_fields)

        # (2) INSTANCES: add each named block, then set its initial density. The block model is the
        # per-instance "model" if given, else the physical Module carried by the compiled handle
        # (compiled.model), never the handle itself.
        for name, spec in instances.items():
            if not isinstance(spec, dict):
                raise TypeError("sim.install: instances[%r] must be a dict "
                                "(initial/spatial/time/model); got %r"
                                % (name, type(spec).__name__))
            model = spec.get("model", compiled_model)
            if model is None:
                raise ValueError(
                    "sim.install: instance %r has no block model -- supply "
                    "instances[%r]['model'] (a compiled/native block model), or pass a compiled handle that carries one "
                    "(compile_problem(model=...))." % (name, name))
            spatial = spec.get("spatial")
            time = spec.get("time")
            self._add_equation(name, model, spatial=spatial, time=time)

        # (3) AUX fields: B_z -> set_magnetic_field; named -> set_aux_field. After the blocks exist
        # (a named aux resolves against the block's declared aux table) and before artifact attach.
        for field_name, field in aux.items():
            self._install_aux(field_name, field)

        # (4) INITIAL state per instance (set_density on the AMR coarse base level).
        for name, spec in instances.items():
            initial = spec.get("initial")
            if initial is not None:
                self.set_density(name, initial)

        # (5/5b/6) COMPILED problem: attach the artifact on the AMR hierarchy, route runtime params
        # and apply the global cadence (or reject params= / cadence= on a NATIVE install). Extracted
        # into the _AmrSystemProgram mixin (_finish_problem_install) to keep this module small.
        self._finish_problem_install(compiled, so_path, params, cadence)
        if outputs:
            self._output_policies = list(outputs)

    def install(self, compiled, *, instances=None, params=None, aux=None,
                solvers=None, cadence=None, outputs=None):
        """Public Spec-5 install entry point for the AMR runtime.

        Installs a combined ``CompiledProblem`` on the AMR runtime. The native per-block route remains
        private/internal; user code enters through ``pops.compile_problem(..., layout=AMR(...))`` and
        this method.
        """
        if compiled is None:
            raise TypeError(
                "sim.install requires a CompiledProblem from pops.compile_problem(...); "
                "the public runtime route is compiled = pops.compile_problem(...), "
                "sim = pops.AmrSystem(...), sim.install(compiled, ...). Native AMR per-block wiring "
                "is private/internal and is not exposed through sim.install.")
        return self._install_compiled(
            compiled,
            instances=instances,
            params=params,
            aux=aux,
            solvers=solvers,
            cadence=cadence,
            outputs=outputs,
        )

    # Field names the default AMR Poisson route already serves (the shared coarse elliptic solve).
    _DEFAULT_POISSON_FIELDS = ("phi", "poisson", "charge_density", "default")

    def _install_solver(self, field, solver_brick, declared_fields=frozenset()):
        """Lower a declared AMR field solver to set_poisson; reject typos before runtime."""
        if field not in self._DEFAULT_POISSON_FIELDS and field not in declared_fields:
            declared = ", ".join(sorted(declared_fields)) or "(none declared)"
            raise ValueError(
                "sim.install: solver selection names field %r, which is neither the default Poisson "
                "field (%s) nor a named elliptic field any installed model declares (declared: %s). "
                "Declare it with m.elliptic_field(%r, rhs=...), or fix the field name."
                % (field, ", ".join(self._DEFAULT_POISSON_FIELDS), declared, field))
        if isinstance(solver_brick, str):
            raise TypeError(
                "sim.install: solver selections must be typed descriptors such as "
                "pops.solvers.GeometricMG(); got legacy token %r" % solver_brick)
        token = getattr(solver_brick, "scheme", None) or getattr(solver_brick, "name", None)
        if token is None:
            raise TypeError("sim.install: solver must be a pops.solvers.<Solver>(...) descriptor; got %r"
                            % type(solver_brick).__name__)
        self._set_poisson(solver=token)

    def _set_poisson(self, rhs="charge_density", solver="geometric_mg", bc="auto",
                     wall="none", wall_radius=0.0, epsilon=1.0, abs_tol=0.0):
        """Private lowering seam for the native AMR Poisson solve."""
        from pops.runtime._system_install import _lower_bc, _lower_wall
        bc = _lower_bc(bc)
        lowered = _lower_wall(wall)
        if lowered is not None:
            wall, wall_radius = lowered
        self._s.set_poisson(rhs=rhs, solver=solver, bc=bc, wall=wall,
                            wall_radius=wall_radius, epsilon=epsilon, abs_tol=abs_tol)

    @staticmethod
    def _declared_elliptic_fields(instances):
        """Collect named elliptic fields declared by per-instance AMR models."""
        names = set()
        for spec in (instances or {}).values():
            if not isinstance(spec, dict):
                continue
            model = spec.get("model")
            if model is None:
                continue
            explicit = getattr(model, "elliptic_field_names", None)
            if explicit is not None:
                names.update(explicit)
                continue
            raw = getattr(model, "_elliptic_fields", None)
            if raw:
                names.update(raw)
        return names

    def field(self, name):
        """Return the solved coarse potential of a named elliptic field."""
        return self._s.named_field_values(name)

    def _install_aux(self, field_name, field):
        """Lower an aux entry on AMR: 'B_z' -> set_magnetic_field; 'T_e' rejected (derived); any
        other name -> set_aux_field on the block that declares it. Mirror of System._install_aux."""
        if field_name == "B_z":
            self.set_magnetic_field(field)
            return
        if field_name == "T_e":
            raise ValueError(
                "sim.install: aux 'T_e' is DERIVED from a fluid block, not a static aux "
                "field; use set_electron_temperature_from(block).")
        block = None
        for blk, table in self._aux_field_index.items():
            if field_name in table:
                block = blk
                break
        if block is None:
            raise ValueError(
                "sim.install: aux field %r is not declared by any installed instance; add the "
                "instance with a model declaring m.aux_field(%r)." % (field_name, field_name))
        self._set_aux_field(block, field_name, field)

    def add_coupling(self, coupling):
        """Add a compiled inter-species coupled source on the shared AMR hierarchy."""
        # Late import (the multispecies module imports this package: avoid the cycle).
        from pops.physics.multispecies import CompiledCoupledSource

        if isinstance(coupling, CompiledCoupledSource):
            self._s.add_coupled_source(coupling.in_blocks, coupling.in_roles, coupling.consts,
                                       coupling.out_blocks, coupling.out_roles, coupling.prog_ops,
                                       coupling.prog_args, coupling.prog_lens,
                                       getattr(coupling, "frequency", 0.0), coupling.name,
                                       getattr(coupling, "freq_prog_ops", []),
                                       getattr(coupling, "freq_prog_args", []))
        else:
            raise TypeError("AmrSystem.add_coupling expects a CompiledCoupledSource "
                            "(pops.dsl.CoupledSource(...).compile(...)): the AMR coupled source is "
                            "MULTI-BLOCK and described in formulas")

    @property
    def amr(self):
        """The live AMR runtime inspection handle (Spec 5 sec.8.12), an
        :class:`pops.runtime.amr.AmrRuntimeView`.

        Bound to THIS built hierarchy: ``sim.amr.patch_table()`` /
        ``sim.amr.hierarchy_snapshot()`` / ``sim.amr.explain_regrid()`` /
        ``explain_ghosts()`` / ``explain_reflux()`` / ``explain_checkpoint()`` return short, inert
        reports of the patches that actually exist, the regrid cadence in force, and the
        ghost / reflux / checkpoint route limitations. The view READS the runtime (the box
        accessors + the retained config); it builds / allocates / steps NOTHING.

        ``System.amr`` does not exist: the inspection surface is AMR-specific (a uniform System
        carries no hierarchy). Use ``pops.inspect_amr(layout)`` for the STATIC authoring report.
        """
        from pops.runtime.amr import AmrRuntimeView  # lazy: keeps the constructor import-light.

        return AmrRuntimeView(self)

    def __str__(self):
        """Short, array-free summary: block names on the AMR hierarchy (Spec 5 sec.12.1).

        Field/patch data stays out of the summary -- it prints the block registry only.
        """
        try:
            blocks = list(self._s.block_names())
        except Exception:  # pragma: no cover - defensive: _AmrSystem not fully wired
            blocks = []
        return "AmrSystem(blocks=%s)" % (blocks,)

    def explain_bind(self, compiled):
        """A printable :class:`pops.codegen.inspect_report.BindReport` of @p compiled vs this AMR sim
        (Spec 5 sec.12.1, criterion #15). INERT parity with ``System.explain_bind``: reads the
        artifact's DECLARED bind inputs (``compiled.arguments()``) and the blocks / named aux wired on
        this AmrSystem, then reuses ADC-463 :func:`collect_missing_arguments` to report PROVIDED vs
        still-REQUIRED per group. It binds nothing and mutates nothing -- a read-only bind plan."""
        from pops.codegen.inspect_report import build_bind_report
        return build_bind_report(self, compiled)

    def __getattr__(self, attr):
        forbidden = {
            "add_block",
            "add_equation",
            "install_problem",
            "install_program",
            "initialize_compiled_program",
            "set_program_cadence",
            "set_param",
            "set_aux_field",
            "set_field_solver",
            "set_poisson",
        }
        if attr in forbidden:
            raise AttributeError(
                "AmrSystem.%s is not part of the public PoPS API; use sim.install(...) "
                "with a compiled artifact and typed descriptors instead." % attr)
        return getattr(self._s, attr)
