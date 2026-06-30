"""Public AMR runtime methods mixed into :class:`pops.runtime.amr_system.AmrSystem`."""

import numpy as np

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
        """Return a :class:`PerformanceSummary`."""
        if self._summary is not None:
            return self._summary
        return PerformanceSummary(self._system.profile_report(), self._profile)


class _AmrSystemPublic:
    """Public readback, profiling, and inspection methods of the AMR runtime."""

    def run(self, t_end, cfl=None, max_steps=1_000_000, output_dir=None):
        """Advance up to ``t_end`` by AMR CFL steps and fire output/checkpoint policies."""
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
        """Typed AMR / MPI profiling context manager."""
        if profile is None:
            profile = Profile.from_env(default=Profile.Basic())
        elif not isinstance(profile, Profile):
            raise TypeError(
                "AmrSystem.profile: expected a pops.Profile (Profile.Basic()/Advanced()), got %r"
                % type(profile).__name__)
        return _AmrProfileSession(self, profile)

    def profile_summary(self, profile=None):
        """Return a typed snapshot of the AMR native profiling report."""
        if profile is None:
            profile = Profile.from_env(default=Profile.Basic())
        elif not isinstance(profile, Profile):
            raise TypeError(
                "AmrSystem.profile_summary: expected a pops.Profile (Profile.Basic()/Advanced()), got %r"
                % type(profile).__name__)
        return PerformanceSummary(self._s.profile_report(), profile)

    def get_recorded_scalars(self):
        """Return scalar diagnostics recorded by the installed compiled AMR problem."""
        return dict(self._s.program_diagnostics())

    def get_state(self, name, *, global_=False):
        """Public conservative-state readback on the AMR base level."""
        if not isinstance(name, str):
            raise TypeError("AmrSystem.get_state: name must be a block name string")
        n = int(self._s.nx())
        try:
            ncomp = int(self._s.block_n_vars(name))
            raw = (self._s.block_level_state_global(name, 0)
                   if global_ else self._s.block_level_state(name, 0))
        except RuntimeError as exc:
            msg = str(exc)
            if "block_level_state" not in msg and "MULTI-BLOCK only" not in msg:
                raise
            ncomp = int(self._s.n_vars())
            raw = self._s.level_state_global(0) if global_ else self._s.level_state(0)
        return np.asarray(raw, dtype=np.float64).reshape(ncomp, n, n)

    def get_current_fields(self, name=None, *, refresh=False):
        """Return the current canonical AMR coarse field bundle."""
        if name is not None and not isinstance(name, str):
            raise TypeError("AmrSystem.get_current_fields: name must be a block name string or None")
        if refresh:
            self._s.potential()
        return {"phi": np.asarray(self._s.potential(), dtype=np.float64)}

    def patch_rectangles(self):
        """Physical rectangles (x0, y0, width, height) of current fine patches."""
        n, L = self._s.nx(), self._L
        rects = []
        for level, ilo, jlo, ihi, jhi in self._s.patch_boxes():
            dx = L / (n << level)
            rects.append((ilo * dx, jlo * dx, (ihi - ilo + 1) * dx, (jhi - jlo + 1) * dx))
        return rects

    def coarse_local_boxes(self):
        """Number of coarse boxes owned by this MPI rank."""
        return self._s.coarse_local_boxes()

    def coarse_total_boxes(self):
        """Total number of coarse boxes across all ranks."""
        return self._s.coarse_total_boxes()

    @property
    def amr(self):
        """The live AMR runtime inspection handle."""
        from pops.runtime.amr import AmrRuntimeView
        return AmrRuntimeView(self)

    def __str__(self):
        """Short, array-free summary: block names on the AMR hierarchy."""
        try:
            blocks = list(self._s.block_names())
        except Exception:  # pragma: no cover - defensive: _AmrSystem not fully wired
            blocks = []
        return "AmrSystem(blocks=%s)" % (blocks,)

    def explain_bind(self, compiled):
        """Printable bind report for a compiled artifact vs this AMR sim."""
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


__all__ = ["_AmrSystemPublic"]
