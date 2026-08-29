"""System : the runtime coupler (Spec-4 PR-F composed class).

``System`` composes blocks, shares a Poisson and advances the whole. Its ~1300 lines of methods
are split into cohesive mixins (``_system_install`` / ``_system_unified_install`` /
``_system_aux_state`` / ``_system_diagnostics`` / ``_system_io``); this module composes them and
keeps the constructor plus delegation glue.
``AmrSystem`` lives in :mod:`pops.runtime._amr_system` and is re-exported only through this private
engine module so compiler/runtime internals share one import seam. Neither engine is a public
authoring API; users obtain the installed runtime exclusively through ``pops.bind``.
"""
from __future__ import annotations

from typing import Any

from pops._bootstrap import SystemConfig, _System  # noqa: F401  (SystemConfig re-exported below)
# The config PODs remain private implementation details alongside their native engines.
from pops._bootstrap import AmrSystemConfig  # noqa: F401  (re-exported via this module)
from pops.runtime import _threading
from pops.runtime._lifecycle import (
    FROZEN_STRUCTURAL as _FROZEN_STRUCTURAL,
    RETIRED_NATIVE_PASSTHROUGH as _RETIRED_NATIVE_PASSTHROUGH,
    freeze_error as _freeze_error,
    _LifecycleMixin,
)
from pops.runtime._amr_system import AmrSystem  # noqa: F401  (re-exported via this module)
from pops.runtime._system_aux_state import _SystemAuxState
from pops.runtime._system_diagnostics import _SystemDiagnostics
from pops.runtime._system_install import _SystemInstall
from pops.runtime._system_io import _SystemIO
from pops.runtime._system_unified_install import _SystemUnifiedInstall
from pops.runtime._profile import PerformanceSummary, Profile
from pops.runtime._private_config_compat import private_constructor_config


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

    Low-level runtime. The documented PUBLIC path is the typed ``pops.Case`` assembly lowered by
    ``pops.compile`` and wired by ``pops.bind`` -> ``pops.run(sim, ...)``; the per-step native methods
    (and ``add_equation`` / ``set_poisson``)
    are the low-level seam ``pops.bind`` builds on and the tests use, not the recommended front
    door.

    ``add_equation`` dispatches a private native ``ModelSpec`` or a compiled production package.
    Public authoring uses ``pops.Model`` through ``pops.Case``; discretization and reusable
    integration Programs live in ``pops.numerics`` and ``pops.lib.time`` respectively.
    Everything else (set_poisson, set_density, step, step_cfl, diagnostics,
    primitives eval_rhs/get_state/set_state) is forwarded to the compiled facade.
    Temporal operations require the whole-system Program installed by ``pops.bind``; there is no
    native fallback. ``install_step_adaptive`` / ``step_adaptive`` install that Program from the
    oracle stride map; they do not resurrect a private native stepper.

    GEOMETRY: ordinary Cartesian authoring is lowered from ``CartesianGrid`` to the private exact-
    ranked ``SystemConfig`` before this engine is constructed. The historical ``mesh=`` bypass and
    its 2-D polar runtime were retired; non-Cartesian providers are refused during resolution rather
    than entering a second native engine."""

    _execution_context: Any

    def __init__(self, config: Any = None, mesh: Any = None, **cfg_kw: Any) -> None:
        if config is None:
            config = private_constructor_config(
                SystemConfig, cfg_kw, runtime="System", adaptive=False
            )
        if mesh is not None:
            raise NotImplementedError(
                "System(mesh=...) was retired with the legacy 2-D polar runtime; native "
                "System<Dim> accepts only the exact Cartesian layout resolved before bind "
                "(got %s)" % type(mesh).__name__
            )
        # Mark the Kokkos init as imminent: _System(config) allocates Fabs -> Kokkos initializes
        # (lazy) here. Runtime thread environment must therefore be fixed before this allocation.
        _threading._first_system_built = True
        self._s = _System(config)
        # Auxiliary storage is owned by the sealed global ProviderPack.  Python
        # carries no block/name-to-component table: callers use ComponentKey and
        # native code resolves its exact `{group, component}` address.
        self._pending_native_packages = 0
        self._batch_native_packages = False
        self._step_strategy = None
        self._step_transaction_plan = None
        self._step_controller = None
        self._last_step_transaction_report = None
        # RUNTIME FREEZE LIFECYCLE (ADC-592): "assembling" while the composition is mutable, "bound"
        # once _finalize_bind runs (the LAST act of _install_compiled). The Python flag enforces the
        # freeze even under a prebuilt .so with no native mark_bound; the native lifecycle is defence
        # in depth. _bound_snapshot is the BoundSnapshot manifest of what was bound (None until bind).
        self._lifecycle = "assembling"
        self._bound_snapshot = None
        self._last_run_manifest = self._last_run_identity = self._last_restart_identity = None
        self._restart_lineage_identity = None
        from pops.runtime._temporal_restart import TemporalRestartState
        self._temporal_restart_state = TemporalRestartState()

    def _native_step_target(self) -> Any:
        """Return the transaction-free compiled target for runtime controllers."""
        return self._s

    def _accepted_transaction_generation_(self) -> Any:
        """Expose the native accepted-transaction generation to aggregate readers."""
        return self._s.accepted_transaction_generation_()

    def step(self, dt: Any) -> None:
        """Advance once through the installed Program's prepared temporal authority."""
        from pops.runtime._step_strategy import prepare_program_run
        from pops.runtime._native_step_target import native_step_target

        prepared_run = prepare_program_run(self)
        prepared_run.begin(
            self._temporal_restart_state, time=self.time(), macro_step=self.macro_step())
        prepared_run.run_step(
            native_step_target(self), t_end=float(self.time()) + float(dt))

    def run(self, t_end: Any, *, max_steps: int, output_dir: Any = None,
            controls: Any = None) -> Any:
        """Advance with the Program-authenticated typed strategy and exact runtime controls."""
        from pops.runtime._step_strategy import prepare_program_run
        from pops.runtime._native_step_target import native_step_target
        prepared_run = prepare_program_run(self, controls)
        prepared_run.begin(
            self._temporal_restart_state, time=self.time(), macro_step=self.macro_step())
        from pops.runtime._run_manifest import begin_run
        begin_run(
            self, t_end=t_end, step_transaction=prepared_run.control_payload,
            max_steps=max_steps, output_dir=output_dir)
        step_target = native_step_target(self)
        steps = 0
        while self.time() < t_end and steps < max_steps:
            prepared_run.run_step(step_target, t_end=float(t_end))
            steps += 1
        return steps

    def profile(self, profile: Any = None) -> Any:
        """Typed profiling context manager (Spec 5 sec.12.5, criteria 41-44).

        Usage::

            with sim.profile() as prof:
                pops.run(sim, t_end=0.1, max_steps=1000)
            print(prof.summary())

        ``profile`` is the private engine ``Profile`` value. With no argument the level comes from
        ``POPS_PROFILE`` (unset / ``off`` -> Basic()). The manager enables the native profiler on
        entry and disables it on exit, so a plain run leaves profiling off. ``prof.summary()``
        returns the private immutable ``PerformanceSummary`` runtime record.
        """
        if profile is None:
            profile = Profile.from_env(default=Profile.Basic())
        elif not isinstance(profile, Profile):
            raise TypeError(
                "System.profile: expected the private engine Profile value, got %r"
                % type(profile).__name__)
        return _ProfileSession(self, profile)

    def _commit_pending_native_packages(self) -> None:
        """Materialize staged CompiledModel packages once the low-level add_equation burst ends."""
        if getattr(self, "_batch_native_packages", False) or not getattr(
            self, "_pending_native_packages", 0
        ):
            return
        from pops.runtime._amr_package_lane import ensure_system_native_package_lane

        ensure_system_native_package_lane(self)
        self._s._finalize_native_packages()
        self._pending_native_packages = 0

    def install_equation_cadence(self, handles: Any, **kwargs: Any) -> Any:
        """Lower recorded ``add_equation(..., time=Explicit(stride=M))`` to a Program."""
        from pops.runtime._cadence_install import install_equation_cadence

        return install_equation_cadence(self, handles, **kwargs)

    def install_step_adaptive(self, handles: Any, wave_speeds: Any, **kwargs: Any) -> Any:
        """Install the oracle adaptive-stride Program and return it."""
        from pops.runtime._cadence_install import install_step_adaptive

        return install_step_adaptive(self, handles, wave_speeds, **kwargs)

    def step_adaptive(
        self, cfl: Any, *, wave_speeds: Any, min_cell_spacing: Any = None
    ) -> Any:
        """Advance one oracle macro-step through the installed Program."""
        from pops.runtime._cadence_install import step_adaptive

        return step_adaptive(
            self, cfl, wave_speeds=wave_speeds, min_cell_spacing=min_cell_spacing
        )

    def block_names(self) -> Any:
        """Names of the installed native blocks, in deterministic registry order.

        Delegates to the C++ block registry (single source), so it includes blocks installed from a
        production package as well as direct native blocks. This is a low-level inspection surface;
        the compiled ``Program`` remains the time-integration authority.
        """
        self._commit_pending_native_packages()
        return list(self._s.block_names())

    def inspect(self) -> Any:
        """Structured, array-free runtime inspection report (ADC-591)."""
        from pops.runtime.inspection import build_runtime_inspection

        self._commit_pending_native_packages()
        return build_runtime_inspection(self, runtime="system")

    def program_report(self) -> Any:
        """Structured report of the compiled-Program runtime subsystem (ADC-594).

        Aggregates the bound Program accessors (installed step / hash, transaction, block map, param
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
        refined runtime. Declare ``layout=AMR(...)`` on the ``pops.Case`` for a refined run, or use
        the STATIC authoring report ``pops.inspect(layout)`` for a layout descriptor. Accessing
        it raises a clear ``AttributeError`` (sourced in ``__getattr__`` so the message is single).
        """
        # The AttributeError routes through __getattr__('amr'), which raises the clear message.
        raise AttributeError("amr")

    @staticmethod
    def abi_key() -> Any:
        """Module ABI key (compiler, C++ standard, signature of the pops headers). Compared to
        that of a native loader by add_native_block. This is an internal class attribute; the
        public lifecycle never exposes ``System``."""
        native: Any = _System
        return native.abi_key()

    def __getattr__(self, attr: Any) -> Any:
        # 'amr' is an AmrSystem-only inspection handle; the System @property raises AttributeError,
        # which routes here -- intercept it so the clear message surfaces instead of the raw _pops
        # "object has no attribute 'amr'" delegation (Spec 5 sec.8.12).
        if attr == "amr":
            raise AttributeError(
                "System has no 'amr' inspection handle: System is a uniform single-level runtime "
                "with no AMR hierarchy. Declare layout=AMR(...) on the pops.Case for a refined run "
                "(its sim.amr returns an AmrRuntimeView), or pops.inspect(layout) for the "
                "static authoring report.")
        if attr in _RETIRED_NATIVE_PASSTHROUGH:
            raise AttributeError(
                "System.%s is not an authoring route; declare the block with pops.Case.block(...)"
                % attr
            )
        # RUNTIME FREEZE (ADC-592): once bound, refuse a native STRUCTURAL setter reached through the
        # passthrough (instance.install_program / ...) with the bind-vocabulary
        # RuntimeError -- NOT AttributeError -- so the bypass is closed even under a prebuilt .so whose
        # C++ setters are not yet frozen. The data / param / diagnostic passthrough is untouched.
        if attr in _FROZEN_STRUCTURAL and getattr(self, "_lifecycle", "assembling") != "assembling":
            raise _freeze_error(attr)
        # Private names must raise AttributeError so getattr(self, "_x", default) works.
        # Finalizing on every underscore lookup recurses through _pending_native_packages.
        if not (isinstance(attr, str) and attr.startswith("_")):
            self._commit_pending_native_packages()
        return getattr(self._s, attr)
