"""AmrSystem : the refined runtime coupler (Spec-4 PR-F composed class).

``AmrSystem`` carries one or several blocks on an AMR hierarchy. Its lines are split into the
``_amr_system_equation`` (add_equation), ``_amr_system_aux_state`` (owner-qualified
``InputAux`` / ``DerivedAux`` routes), ``_amr_system_io`` (private accepted-state
codec and restore transaction), ``_amr_system_program`` (compiled time-Program install / params / transaction)
and ``_amr_system_install`` (the ``pops.bind`` install seam + field-solver / aux helpers)
mixins; this module composes them and keeps the constructor plus coupling glue.
"""

from __future__ import annotations

from typing import Any

from pops._bootstrap import AmrSystemConfig, _AmrSystem
from pops.runtime import _threading
from pops.runtime._lifecycle import (
    FROZEN_STRUCTURAL as _FROZEN_STRUCTURAL,
    RETIRED_NATIVE_PASSTHROUGH as _RETIRED_NATIVE_PASSTHROUGH,
    freeze_error as _freeze_error,
    guard_assembling as _guard_assembling,
    _LifecycleMixin,
)
from pops.runtime._amr_system_equation import _AmrSystemEquation
from pops.runtime._amr_system_aux_state import _AmrSystemAuxState
from pops.runtime._amr_system_install import _AmrSystemInstall
from pops.runtime._amr_system_io import _AmrSystemIO
from pops.runtime._amr_system_program import _AmrSystemProgram
from pops.runtime._profile import PerformanceSummary, Profile
from pops.runtime._private_config_compat import private_constructor_config


def _profile_payload(system: Any) -> Any:
    """Structured profiler payload when the native extension exposes it, else legacy text."""
    snapshot = getattr(system, "profile_snapshot", None)
    if callable(snapshot):
        return snapshot()
    return system.profile_report()


class _AmrProfileSession:
    """The typed profiling context manager AmrSystem.profile() returns (Spec 5 sec.12.5).

    Mirror of :class:`pops.runtime._system._ProfileSession` for the AMR runtime: ``__enter__``
    resets + enables the native profiler ; ``__exit__`` snapshots the report into a
    :class:`PerformanceSummary` and disables the profiler. ``summary().by_amr_mpi()`` surfaces the
    AMR / MPI phase timings (regrid / fill_boundary / average_down) + counters (criterion 43). Lives
    here rather than importing from ``_system.py`` to avoid a circular import.
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


class AmrSystem(
    _AmrSystemEquation, _AmrSystemAuxState, _AmrSystemInstall, _AmrSystemIO,
    _AmrSystemProgram, _LifecycleMixin
):
    """Refined counterpart of System : one or SEVERAL blocks carried on an AMR hierarchy.

    One or several blocks are co-located on one shared AMR hierarchy through the same AmrRuntime,
    SYSTEM Poisson with co-located SUMMED right-hand side (Sum_b q_b n_b), conservation PER BLOCK. The
    blocks may have different spatial schemes, coupled inter-species sources and compiled production
    implementations. Their time descriptors are Program-authoring inputs only: one installed
    ProgramGraph owns stages, cadence and accepted clocks. In multi-block the block name indexes
    set_density(name) / mass(name) / density(name).

    UNION-OF-TAGS REGRID (regrid_every > 0): the shared hierarchy consumes the exact prepared
    AMRTagging graph resolved from the layout. State, field-gradient and logical nodes keep their
    block-qualified identities; the native engine never substitutes a component-zero threshold.

    regrid_every == 0 -> FROZEN hierarchy (regrid never called, bit-identical).
    """

    def __init__(self, config: Any = None, **cfg_kw: Any) -> None:
        if config is None:
            config = private_constructor_config(
                AmrSystemConfig, cfg_kw, runtime="AmrSystem", adaptive=True
            )
        # cf. System.__init__ : _AmrSystem(config) triggers the Kokkos init (lazy). set_threads
        # has no more effect after this point.
        _threading._first_system_built = True
        self._s = _AmrSystem(config)
        self._shape = tuple(config.shape)
        self._lower = tuple(config.lower)
        self._upper = tuple(config.upper)
        if not (
            len(self._shape) == len(self._lower) == len(self._upper)
            and len(self._shape) in (1, 2, 3)
        ):
            raise ValueError("AmrSystemConfig spatial arrays do not share one supported rank")
        self._lengths = tuple(
            high - low for low, high in zip(self._lower, self._upper, strict=True)
        )
        # Regrid cadence (checkpoint/restart ADC-65) : a BIT-IDENTICAL resume requires regrid_every == 0
        # (otherwise the post-restart regrid would re-diverge the hierarchy). Memorized for the restart guard.
        self._regrid_every = int(config.regrid_every)
        # RUNTIME FREEZE LIFECYCLE (ADC-592, parity with System): "assembling" until _finalize_bind
        # flips it to "bound" (the LAST act of _install_compiled). The Python flag enforces the freeze
        # even under a prebuilt .so with no native mark_bound; _bound_snapshot is the BoundSnapshot of
        # what was bound (None until bind).
        self._lifecycle = "assembling"
        self._bound_snapshot = None
        self._last_run_manifest = None
        self._last_run_identity = None
        self._last_restart_identity = None
        self._restart_lineage_identity = None
        self._step_strategy = None
        self._step_transaction_plan = None
        self._step_controller = None
        self._last_step_transaction_report = None
        from pops.runtime._temporal_restart import TemporalRestartState

        self._temporal_restart_state = TemporalRestartState()

    def _native_step_target(self) -> Any:
        """Return the transaction-free compiled target for runtime controllers."""
        return self._s

    def step(self, dt: Any) -> None:
        """Advance one fixed step and synchronize exactly one temporal envelope."""
        from pops.runtime._native_step_target import native_step_target
        from pops.runtime._step_strategy import prepare_program_run

        prepared_run = prepare_program_run(self)
        prepared_run.begin(
            self._temporal_restart_state,
            time=self._s.time(),
            macro_step=self._s.macro_step(),
        )
        prepared_run.run_step(
            native_step_target(self),
            t_end=float(self._s.time()) + float(dt),
        )

    def set_poisson(
        self,
        rhs: Any = "charge_density",
        solver: Any = "geometric_mg",
        *,
        bc: Any = None,
    ) -> Any:
        """Configure AMR Poisson with a typed physical-boundary selector.

        ``bc`` accepts a typed native boundary descriptor; omission keeps automatic selection.
        Embedded geometry is authored independently through the exact-ranked analytic level-set route.
        """
        from pops.runtime._system_install_lowering import _lower_bc

        bc_token = "auto" if bc is None else _lower_bc(bc)
        self._set_poisson_native(rhs=rhs, solver=solver, bc=bc_token)

    def _set_poisson_native(self, *, rhs: Any, solver: Any, bc: Any) -> Any:
        """Private token-level seam used by resolved AMR installation."""
        _guard_assembling(self, "set_poisson")
        if not isinstance(bc, str):
            raise TypeError("_set_poisson_native requires one native boundary token")
        self._s.set_poisson(rhs=rhs, solver=solver, bc=bc)

    def run(self, t_end, *, max_steps, output_dir=None, controls=None):
        """Advance up to ``t_end``; RuntimeInstance alone publishes ConsumerGraph effects."""
        from pops.runtime._step_strategy import prepare_program_run
        from pops.runtime._native_step_target import native_step_target

        prepared_run = prepare_program_run(self, controls)
        prepared_run.begin(
            self._temporal_restart_state, time=self._s.time(), macro_step=self._s.macro_step()
        )
        from pops.runtime._run_manifest import begin_run

        begin_run(
            self,
            t_end=t_end,
            step_transaction=prepared_run.control_payload,
            max_steps=max_steps,
            output_dir=output_dir,
        )
        step_target = native_step_target(self)
        steps = 0
        while self._s.time() < t_end and steps < max_steps:
            prepared_run.run_step(step_target, t_end=float(t_end))
            steps += 1
        return steps

    def profile(self, profile: Any = None) -> Any:
        """Typed AMR / MPI profiling context manager (Spec 5 sec.12.5, criterion 43).

        Usage::

            with sim.profile() as prof:
                for _ in range(n_steps):
                    sim.step_cfl(0.4)
            print(prof.summary().by_amr_mpi())  # regrid / fill_boundary / average_down timings

        ``profile`` is the private engine ``Profile`` value. With no argument it comes from
        ``POPS_PROFILE`` (unset / ``off`` -> Basic()). The manager enables the native AMR profiler
        on entry and disables it on exit. ``prof.summary().by_amr_mpi()`` surfaces AMR phase timings
        and counters as soon as a regrid or solve fired under the multi-block engine.
        """
        if profile is None:
            profile = Profile.from_env(default=Profile.Basic())
        elif not isinstance(profile, Profile):
            raise TypeError(
                "AmrSystem.profile: expected the private engine Profile value, got %r"
                % type(profile).__name__
            )
        return _AmrProfileSession(self, profile)

    def patch_bounds(self) -> Any:
        """Physical ``lower + extents`` tuples for the current ranked fine patches."""
        from pops.runtime._amr_bind_lowering import _physical_patch_bounds

        return _physical_patch_bounds(
            self._s.patch_boxes(),
            cells=self._shape,
            lengths=self._lengths,
            lower=self._lower,
        )

    def coarse_local_boxes(self) -> Any:
        """Number of coarse (base) boxes owned by this MPI rank (ADC-319 diagnostic).

        The base level is a MultiFab whose boxes are spread across ranks by a DistributionMapping.
        Returns this rank's owned-fab count (level-0 local_size()). With distribute_coarse=True the base
        is split into several boxes round-robin, so each rank owns a strict subset and the coarse
        transport is distributed; a replicated or single-box base owns the full count on every rank.
        Compare with coarse_total_boxes() and the runtime communicator size to confirm MPI
        strong-scaling of the base.
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

    def field(self, name: Any) -> Any:
        """Return the solved potential of a NAMED elliptic field as a ``(ny, nx)`` array.

        Read-back of a second elliptic field declared via m.elliptic_field and lowered on the AMR layout:
        solves the hierarchy fields if needed (so it is current even before any step) then reads the
        field's coarse potential. AMR counterpart of reading System.aux_field(block, name) for an elliptic
        field. @throws if the field is unregistered (or the system runs the single-block coupler, which
        carries no named field)."""
        return self._s.named_field_values(name)

    def add_coupling(self, coupling: Any) -> Any:
        """Reject coupling until one complete multi-block AMR provider owns its execution."""
        _guard_assembling(self, "add_coupling")  # frozen once pops.bind completes (ADC-592)
        raise NotImplementedError(
            "AMR coupling installation requires the atomic "
            "PreparedMultiBlockAmrHierarchy<Dim> coupling provider; the exact single-block "
            "AMR core publishes no coupled-source executor"
        )

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
        uses the common dt-bound contract but has no scheduler cache / history rings wired, so those
        sections stay empty). Metadata only; installed=False with empty sections on a runtime with no
        program installed."""
        from pops.runtime.program_report import build_program_report

        return build_program_report(self)

    def __getattr__(self, attr: Any) -> Any:
        if attr in _RETIRED_NATIVE_PASSTHROUGH:
            raise AttributeError(
                "AmrSystem.%s is not an authoring route; declare the block with "
                "pops.Case.block(...)" % attr
            )
        # RUNTIME FREEZE (ADC-592): once bound, refuse a native STRUCTURAL setter reached through the
        # passthrough (install_program / ...) with the bind-vocabulary
        # RuntimeError, so the bypass is closed even under a prebuilt .so whose C++ setters are not yet
        # frozen. The data / param / diagnostic passthrough is untouched.
        if attr in _FROZEN_STRUCTURAL and getattr(self, "_lifecycle", "assembling") != "assembling":
            raise _freeze_error(attr)
        return getattr(self._s, attr)
