"""System : the runtime coupler (Spec-4 PR-F composed class).

``System`` composes blocks, shares a Poisson and advances the whole. Its ~1300 lines of methods
are split into cohesive mixins (``_system_install`` / ``_system_unified_install`` /
``_system_aux_state`` / ``_system_diagnostics`` / ``_system_io``) to satisfy the per-file
<=500-line cap ; this module composes them and keeps the constructor + the delegation glue.
``AmrSystem`` lives in :mod:`pops.runtime.amr_system` and is re-exported here for the
``from pops.runtime.system import System, AmrSystem`` import in the slim ``pops`` hub.
"""
from __future__ import annotations

from typing import Any

from pops._bootstrap import SystemConfig, _System  # noqa: F401  (SystemConfig re-exported below)
# ADC-545: SystemConfig / AmrSystemConfig left the pops root; this module is their advanced home
# alongside System / AmrSystem, so `from pops.runtime.system import SystemConfig, AmrSystemConfig`
# resolves for the native/advanced tests that build the config POD directly.
from pops._bootstrap import AmrSystemConfig  # noqa: F401  (re-exported via this module)
from pops.runtime import threading as _threading
from pops.runtime._lifecycle import (
    FROZEN_STRUCTURAL as _FROZEN_STRUCTURAL, freeze_error as _freeze_error, _LifecycleMixin)
from pops.runtime.amr_system import AmrSystem  # noqa: F401  (re-exported via this module)
from pops.runtime._system_aux_state import _SystemAuxState
from pops.runtime._system_diagnostics import _SystemDiagnostics
from pops.runtime._system_install import _SystemInstall
from pops.runtime._system_io import _SystemIO
from pops.runtime._system_unified_install import _SystemUnifiedInstall
from pops.runtime.profile import PerformanceSummary, Profile


def _profile_payload(system: Any) -> Any:
    """Structured profiler payload when the native extension exposes it, else legacy text."""
    snapshot = getattr(system, "profile_snapshot", None)
    if callable(snapshot):
        return snapshot()
    return system.profile_report()


class _ProfileSession:
    """The typed profiling context manager System.profile() returns (Spec 5 sec.12.5).

    ``__enter__`` resets + enables the native profiler; ``__exit__`` snapshots the report into a
    :class:`PerformanceSummary` and disables the profiler. ``summary()`` works inside OR after the
    ``with`` block (it re-reads the live report while open, returns the closing snapshot after).
    The off-by-default contract holds: nothing here enables until the block is entered.
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
        """Return a :class:`PerformanceSummary` of the run.

        Inside the ``with`` block it reads the live native report; after the block it returns the
        snapshot taken on ``__exit__``.
        """
        if self._summary is not None:
            return self._summary
        return PerformanceSummary(_profile_payload(self._system), self._profile)


class System(_SystemInstall, _SystemUnifiedInstall, _SystemAuxState,
             _SystemDiagnostics, _SystemIO, _LifecycleMixin):
    """The system/coupler: composes blocks, shares a Poisson, advances the whole.

    Low-level runtime. The documented PUBLIC path is the typed ``pops.Problem`` assembly lowered by
    ``pops.compile`` and wired by ``pops.bind`` -> ``sim.run(...)``; the per-step ``step_cfl`` /
    ``step`` / ``step_adaptive`` methods (and ``add_block`` / ``add_equation`` / ``set_poisson``)
    are the low-level seam ``pops.bind`` builds on and the tests use, not the recommended front
    door.

    add_block takes a composed model (pops.Model(...)) + Spatial / Explicit / IMEX objects.
    Everything else (set_poisson, set_density, step, step_cfl, step_adaptive, diagnostics,
    primitives eval_rhs/get_state/set_state) is forwarded to the compiled facade.

    GEOMETRY: the choice lives in a MESH object passed as mesh= (pops.CartesianMesh / pops.PolarMesh),
    NOT in the scheme (pops.FiniteVolume stays reconstruction + Riemann + variables). Default (mesh=None
    or pops.CartesianMesh) = square domain, bit-identical to the history. pops.PolarMesh (global ring)
    is WIRED in System.step (Phase 2b): polar ExB transport + polar Poisson + aux in local basis
    (e_r, e_theta). Limits: scalar ExB transport, single-rank, no cart<->polar coupling."""

    def __init__(self, config: Any = None, mesh: Any = None, **cfg_kw: Any) -> None:
        if config is None:
            config = SystemConfig()
            for k, v in cfg_kw.items():
                setattr(config, k, v)
        # The mesh (if provided) carries the geometry CHOICE and overrides the corresponding fields
        # of the config. Applied AFTER cfg_kw: mesh= takes precedence over the n=/L= passed as keywords.
        if mesh is not None:
            if not hasattr(mesh, "_apply"):
                raise TypeError("System: mesh must be an pops.CartesianMesh / pops.PolarMesh (got %r)"
                                % type(mesh).__name__)
            mesh._apply(config)
        # Mark the Kokkos init as imminent: _System(config) allocates Fabs -> Kokkos initializes
        # (lazy) here. After this point, pops.set_threads has no further effect (warned by set_threads).
        _threading._first_system_built = True
        self._s = _System(config)  # geometry == 'polar' builds a global ring (Phase 2b, cf. PolarMesh)
        # Table of NAMED aux fields per block (ADC-70 phase 1): block -> {name: canonical component}.
        # Filled by add_equation from CompiledModel.aux_extra_names (the component of the k-th name =
        # dsl.AUX_NAMED_BASE + k). The FACADE holds the names: the C++ only manipulates component
        # indices (set_aux_field_component / aux_field_component). Empty for a block without a
        # named aux field. cf. set_aux_field / aux_field.
        self._aux_field_index = {}
        # CFL carried by an installed compiled-time cadence (CompiledTime(cfl=X)), or None when the
        # cadence pins no numeric cfl. run() with no explicit cfl= defaults to it, so a bare
        # sim.run(t_end) after bind(..., cadence=CompiledTime(cfl=X)) advances at X (not silently
        # ignored). Set by _install_cadence; None until a numeric-cfl cadence is installed.
        self._program_cadence_cfl = None
        # OUTPUT / CHECKPOINT policies (C4 / ADC-509) flowed by pops.bind through _install_compiled.
        # Empty until install; run(output_dir=...) fires each at its cadence via write()/checkpoint.
        self._output_policies = []
        # DECLARED diagnostic measures (ADC-542) flowed by pops.bind. Empty until install; run() fires
        # each DUE measure at its cadence, lowering it to a native collective reduction and recording
        # the scalar (readable via program_diagnostics). Previously the measures were dropped.
        self._diagnostic_measures = []
        # RUNTIME FREEZE LIFECYCLE (ADC-592): "assembling" while the composition is mutable, "bound"
        # once _finalize_bind runs (the LAST act of _install_compiled). The Python flag enforces the
        # freeze even under a prebuilt .so with no native mark_bound; the native lifecycle is defence
        # in depth. _bound_snapshot is the BoundSnapshot manifest of what was bound (None until bind).
        self._lifecycle = "assembling"
        self._bound_snapshot = None
        self._last_run_manifest = self._last_run_identity = self._last_restart_identity = None

    def run(self, t_end: Any, cfl: Any = None, max_steps: int = 1_000_000,
            output_dir: Any = None, strategy: Any = None) -> Any:
        """Advance up to t_end by CFL steps (sugar: `while time() < t_end: step_cfl(cfl)`).

        @p cfl: Courant number passed to step_cfl. When omitted (None) it defaults to the CFL pinned
        by an installed ``CompiledTime(cfl=X)`` cadence, else 0.4 -- so a numeric cadence cfl actually
        takes effect on a bare ``sim.run(t_end)`` rather than being silently ignored. @p max_steps:
        guard (avoids an infinite loop if dt -> 0). @p output_dir: when output / checkpoint policies
        were flowed onto this System (``pops.bind`` from a Problem with ``.output(policy)``), the
        directory the run writes them to; each policy fires at its own cadence through the existing
        write()/checkpoint writers (C4 / ADC-509). Defaults to the current directory when policies
        are present and output_dir is omitted. Returns the number of steps taken.
        cf. DSL_MODEL_DESIGN.md section 6."""
        from pops.runtime._step_strategy import (
            AdaptiveCFL, resolve_run_strategy, run_step_attempt)
        strategy = resolve_run_strategy(self, strategy, cfl)
        manifest_cfl = strategy.cfl if isinstance(strategy, AdaptiveCFL) else 0.0
        from pops.runtime._run_manifest import begin_run
        begin_run(self, t_end=t_end, cfl=manifest_cfl, max_steps=max_steps, output_dir=output_dir)
        policies = getattr(self, "_output_policies", [])
        measures = getattr(self, "_diagnostic_measures", [])
        out_dir = output_dir if output_dir is not None else "."
        # ConservationCheck anchors its drift to the FIRST-tick value; the run owns the baseline map so
        # it persists across the loop (the driver seeds an entry the first time a check fires).
        baselines = {}
        steps = 0
        while self.time() < t_end and steps < max_steps:
            run_step_attempt(self, self, strategy, t_end=float(t_end))
            steps += 1
            # on_end honesty: dt is CFL-driven, so the final step count is unknown a priori. This step
            # is the LAST one iff the loop is about to exit (t_end reached or the max_steps guard hit).
            last_step = steps if (not (self.time() < t_end) or steps >= max_steps) else None
            if policies:
                self._fire_outputs(policies, steps, out_dir, last_step)
            if measures:
                self._fire_diagnostics(measures, steps, last_step, baselines)
        return steps

    def _fire_outputs(self, policies: Any, step: Any, output_dir: Any, last_step: Any = None) -> Any:
        """Fire the DUE output / checkpoint policies at macro-step @p step (C4 run-loop hook).

        Delegates to :func:`pops.runtime._output_driver.fire_output_policies`, which maps each
        policy's typed cadence/format/fields onto the existing ``write`` / ``checkpoint`` writers.
        Kept tiny so the cadence logic lives in one host-testable place, not inline in run()."""
        from pops.runtime._output_driver import fire_output_policies
        return fire_output_policies(self, policies, step, output_dir, last_step=last_step)

    def _fire_diagnostics(self, measures, step, last_step, baselines):
        """Fire the DUE declared diagnostic measures at macro-step @p step (ADC-542 run-loop hook).

        Delegates to :func:`pops.runtime._diagnostics_driver.fire_diagnostics`, which lowers each due
        measure to a native collective reduction on this System and records the scalar via
        ``record_program_diagnostic``. Kept tiny (mirrors :meth:`_fire_outputs`) so the reduction
        mapping lives in one host-testable place."""
        from pops.runtime._diagnostics_driver import fire_diagnostics
        return fire_diagnostics(self, measures, step, last_step, baselines)

    def profile(self, profile: Any = None) -> Any:
        """Typed profiling context manager (Spec 5 sec.12.5, criteria 41-44).

        Usage::

            with sim.profile(pops.Profile.Basic()) as prof:
                sim.run(t_end=0.1)
            print(prof.summary())

        @p profile is a :class:`pops.Profile` level (``Profile.Basic()`` / ``Profile.Advanced()``);
        with no argument the level comes from ``POPS_PROFILE`` (unset / ``off`` -> Basic()). The
        manager enables the native profiler on entry and disables it on exit, so a plain run (no
        ``with sim.profile()``) leaves profiling off -- the off-by-default contract. ``prof.summary()``
        returns a :class:`pops.PerformanceSummary`.
        """
        if profile is None:
            profile = Profile.from_env(default=Profile.Basic())
        elif not isinstance(profile, Profile):
            raise TypeError(
                "System.profile: expected a pops.Profile (Profile.Basic()/Advanced()), got %r"
                % type(profile).__name__)
        return _ProfileSession(self, profile)

    def block_names(self) -> Any:
        """Names of the added blocks, in order (useful for a Python integrator).

        Delegates to the C++ block registry (single source), so it includes the blocks loaded via
        add_dynamic_block (.so JIT) and add_compiled_block (.so AOT), not only add_block.
        """
        return list(self._s.block_names())

    def inspect(self) -> Any:
        """Structured, array-free runtime inspection report (ADC-591)."""
        from pops.runtime.inspection import build_runtime_inspection
        return build_runtime_inspection(self, runtime="system")

    def program_report(self) -> Any:
        """Structured report of the compiled-Program runtime subsystem (ADC-594).

        Aggregates the bound Program accessors (installed step / hash, cadence, block map, param
        counts, diagnostics, histories, cache, profiler) into one inspectable value object. Metadata
        only; installed=False with empty sections on a runtime with no program installed."""
        from pops.runtime.program_report import build_program_report
        return build_program_report(self)

    def __str__(self) -> Any:
        """Short, array-free summary: the installed block names (Spec 5 sec.12.1).

        Deliberately field-data-free -- it prints the block registry, never a Fab dump.
        """
        try:
            blocks = self.block_names()
        except Exception:  # pragma: no cover - defensive: _System not fully wired
            blocks = []
        return "System(blocks=%s)" % (blocks,)

    @property
    def amr(self) -> Any:
        """The AMR runtime inspection surface does not apply to a uniform ``System``.

        ``System`` is single-level: it carries no AMR hierarchy, so ``sim.amr`` (the live
        patch / regrid / ghost / reflux / checkpoint reports of Spec 5 sec.8.12) applies only to a
        refined runtime. Declare ``layout=AMR(...)`` on the ``pops.Problem`` for a refined run, or use
        the STATIC authoring report ``pops.inspect_amr(layout)`` for a layout descriptor. Accessing
        it raises a clear ``AttributeError`` (sourced in ``__getattr__`` so the message is single).
        """
        # The AttributeError routes through __getattr__('amr'), which raises the clear message.
        raise AttributeError("amr")

    @staticmethod
    def abi_key() -> Any:
        """Module ABI key (compiler, C++ standard, signature of the pops headers). Compared to
        that of a native loader by add_native_block. Also exposed as a class attribute (the
        __getattr__ delegate only covers instances), so pops.System.abi_key() works."""
        native: Any = _System
        return native.abi_key()

    def __getattr__(self, attr: Any) -> Any:
        # 'amr' is an AmrSystem-only inspection handle; the System @property raises AttributeError,
        # which routes here -- intercept it so the clear message surfaces instead of the raw _pops
        # "object has no attribute 'amr'" delegation (Spec 5 sec.8.12).
        if attr == "amr":
            raise AttributeError(
                "System has no 'amr' inspection handle: System is a uniform single-level runtime "
                "with no AMR hierarchy. Declare layout=AMR(...) on the pops.Problem for a refined run "
                "(its sim.amr returns an AmrRuntimeView), or pops.inspect_amr(layout) for the "
                "static authoring report.")
        # RUNTIME FREEZE (ADC-592): once bound, refuse a native STRUCTURAL setter reached through the
        # passthrough (sim._engine.install_program / set_refinement / ...) with the bind-vocabulary
        # RuntimeError -- NOT AttributeError -- so the bypass is closed even under a prebuilt .so whose
        # C++ setters are not yet frozen. The data / param / diagnostic passthrough is untouched.
        if attr in _FROZEN_STRUCTURAL and getattr(self, "_lifecycle", "assembling") != "assembling":
            raise _freeze_error(attr)
        return getattr(self._s, attr)
