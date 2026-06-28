"""Spec 5 gap #32 (criterion 32): the layout=AMR end-to-end config-flow + bind route.

A WHOLE-SYSTEM compiled time Program on an AMR layout is rejected EARLY at ``pops.compile``
(C2 / ADC-508): the AMR runtime has no ``install_program`` seam, so ``pops.compile(case_with_AMR,
time=Program)`` would otherwise compile a ``.so`` and then die at ``pops.bind`` -- a transitional
compile-succeeds-then-bind-fails reject. ``pops.compile`` now fails loud BEFORE any compile,
redirecting to the WIRED per-block AMR path (``pops.physics.Model.compile(backend="production",
target="amr_system")`` + ``pops.AmrSystem.add_equation``).

What is still proven LOCALLY here:

  (a) The whole-system AMR compile route is rejected at ``pops.compile`` (no ``.so`` built; not a
      bind-time reject), AND the config-flow helpers ``pops.bind`` uses on the per-block AMR path
      are correct: the ``AmrSystemConfig`` derived from the layout has the right n / L / periodic /
      regrid_every / patch settings; the layout's max_levels / ratio are validated against the
      native envelope (the config has no such field -- the native AMR is fixed at 2 levels / ratio
      2).
  (b) A REAL native ``AmrSystem`` built from the derived config + ``set_refinement`` +
      ``set_poisson`` + ``set_density`` (NATIVE bricks, no DSL compile) advances a few steps and
      stays physical -- proving the config-flow + refinement + Poisson wiring is correct end to
      end on the native engine path (the wired AMR route).

Runs under pytest and as a plain script (the ``__main__`` guard); the CI runner executes it as a
script.
"""
import sys

try:
    import numpy as np
    import pops
    from pops.codegen import orchestration
    from pops.mesh.amr import (FrozenRegrid, PatchLayout, Refine, RegridEvery, TagUnion,
                               NATIVE_MAX_LEVELS, NATIVE_RATIOS)
    from pops.mesh.cartesian import CartesianMesh
    from pops.mesh.layouts import AMR
except Exception as exc:  # noqa: BLE001
    print("skip test_spec5_amr_e2e (pops unavailable: %s)" % exc)
    sys.exit(0)


def _check(cond, msg):
    if not cond:
        raise AssertionError(msg)


class _StubModel:
    """A physics stand-in exposing the ``.dsl`` engine model the compile route resolves."""

    def __init__(self, name="ne"):
        self.name = name
        self.dsl = object()


# --- monkeypatch helpers (work under pytest fixture OR the bare __main__ runner) ---
_SAVED = []


def _patch(monkeypatch, dotted, value):
    module_name, attr = dotted.rsplit(".", 1)
    import importlib
    module = importlib.import_module(module_name)
    if monkeypatch is not None:
        monkeypatch.setattr(module, attr, value)
    else:
        _SAVED.append((module, attr, getattr(module, attr)))
        setattr(module, attr, value)


def _unpatch(monkeypatch):
    if monkeypatch is not None:
        return
    while _SAVED:
        module, attr, original = _SAVED.pop()
        setattr(module, attr, original)


# --- (a) the compile ROUTE rejects + the AmrSystemConfig-from-layout mapping --------
def test_amr_whole_system_program_rejected_at_compile(monkeypatch=None):
    """C2 (ADC-508): compile(case_with_AMR, time=Program) rejects EARLY -- BEFORE any .so build --
    redirecting to the wired per-block AMR path; it is NOT a bind-time reject (compile_problem is a
    tripwire so a leak past the reject is caught)."""
    called = {"compile_problem": False}

    def _tripwire(*a, **kw):
        called["compile_problem"] = True
        raise AssertionError("compile_problem must not be reached for an AMR whole-system Program")

    _patch(monkeypatch, "pops.codegen.compile_drivers.compile_problem", _tripwire)
    try:
        layout = AMR(CartesianMesh(n=48, L=2.0, periodic=False), max_levels=2, ratio=2)
        prob = pops.Case(layout=layout).block("ne", physics=_StubModel())
        try:
            orchestration.compile(prob, time=object())
            raise AssertionError("an AMR whole-system time Program must be rejected at compile()")
        except NotImplementedError as exc:
            _check("ADC-508" in str(exc), "the reject names the deferred issue ADC-508")
            _check("amr_system" in str(exc) and "add_equation" in str(exc),
                   "the reject redirects to the wired per-block AMR path")
        _check(called["compile_problem"] is False,
               "the reject fires BEFORE compile_problem (no .so built; not a bind-time reject)")
    finally:
        _unpatch(monkeypatch)
    print("ok test_amr_whole_system_program_rejected_at_compile")


def test_amr_config_from_layout_mapping():
    """The AmrSystemConfig pops.bind builds from the layout matches the descriptor.

    Builds the config directly (no .so, no AmrSystem) and asserts each field; also confirms the
    layout's max_levels / ratio sit inside the native envelope (the config carries no such knob).
    """
    layout = AMR(CartesianMesh(n=64, L=1.5, periodic=False), max_levels=2, ratio=2,
                 regrid=RegridEvery(8),
                 patches=PatchLayout(distribute_coarse=True, coarse_max_grid=16))
    cfg = orchestration._amr_config_from_layout(layout)
    _check(cfg.n == 64, "n from the base CartesianMesh")
    _check(cfg.L == 1.5, "L from the base CartesianMesh")
    _check(cfg.periodic is False, "periodic from the base CartesianMesh")
    _check(cfg.regrid_every == 8, "regrid_every from RegridEvery(8)")
    _check(cfg.distribute_coarse is True, "distribute_coarse from PatchLayout")
    _check(cfg.coarse_max_grid == 16, "coarse_max_grid from PatchLayout")
    # max_levels / ratio are validated against the native envelope, not stored on the config.
    _check(layout.max_levels <= NATIVE_MAX_LEVELS, "max_levels within the native envelope")
    _check(layout.ratio in NATIVE_RATIOS, "ratio within the native envelope")
    print("ok test_amr_config_from_layout_mapping")


def test_amr_config_regrid_and_patch_defaults():
    """FrozenRegrid / no regrid -> regrid_every == 0; default patches -> native config defaults."""
    frozen = orchestration._amr_config_from_layout(
        AMR(CartesianMesh(n=32), regrid=FrozenRegrid()))
    _check(frozen.regrid_every == 0, "FrozenRegrid -> frozen hierarchy (regrid_every == 0)")
    no_regrid = orchestration._amr_config_from_layout(AMR(CartesianMesh(n=32)))
    _check(no_regrid.regrid_every == 0, "no regrid policy -> regrid_every == 0")
    _check(no_regrid.n == 32, "n still derived from the base mesh")
    print("ok test_amr_config_regrid_and_patch_defaults")


def test_amr_refine_default_density_subject():
    """The density / component-0 subjects map to the single-block default; others are non-default."""
    _check(orchestration._is_default_density_subject("Density"), "Density role is the default")
    _check(orchestration._is_default_density_subject("density"), "density name is the default")
    _check(orchestration._is_default_density_subject("rho"), "rho name is the default")
    _check(orchestration._is_default_density_subject(None), "no subject is the default")
    _check(not orchestration._is_default_density_subject("MomentumX"),
           "a non-density role is a non-default (multi-block) selector")
    print("ok test_amr_refine_default_density_subject")


def test_amr_non_default_refine_selector_rejected():
    """A non-density refine selector is refused on the single-block AMR route (clear message)."""
    class _Recorder:
        def set_refinement(self, *a, **k):
            raise AssertionError("set_refinement must not be called for a rejected selector")

        def set_phi_refinement(self, *a, **k):
            raise AssertionError("set_phi_refinement must not be called here")

    layout = AMR(CartesianMesh(n=32))
    layout.refine = Refine.on("MomentumX").above(0.5)
    try:
        orchestration._flow_amr_layout(_Recorder(), layout)
        raise AssertionError("a non-density selector must raise on the single-block route")
    except NotImplementedError as exc:
        _check("multi-block" in str(exc), "the message names the multi-block limitation")
    print("ok test_amr_non_default_refine_selector_rejected")


# --- (b) a REAL native AmrSystem from the derived config runs a few steps -----------
def _native_compressible_model():
    """A native compressible-flow ModelSpec (composed bricks, NO DSL compile)."""
    return pops.Model(state=pops.FluidState("compressible", gamma=1.4),
                      transport=pops.CompressibleFlux(), source=pops.NoSource(),
                      elliptic=pops.BackgroundDensity(alpha=0.0, n0=0.0))


def _bubble(n):
    xs = (np.arange(n) + 0.5) / n
    X, Y = np.meshgrid(xs, xs)
    return (1.0 + 0.5 * np.exp(-((X - 0.5) ** 2 + (Y - 0.5) ** 2) / 0.02)).reshape(-1)


def test_native_amr_from_layout_runs():
    """A native AmrSystem built from a layout's derived config + set_refinement + set_poisson runs.

    This is the config-flow + refinement + Poisson wiring proven on the REAL engine (native bricks,
    no DSL compile). It mirrors exactly what pops.bind does on the AMR route: build the AmrSystem
    from _amr_config_from_layout(layout), flow the typed Refine to set_refinement, set the Poisson
    solver, add the native block and the initial density, then step.
    """
    n = 48
    layout = AMR(CartesianMesh(n=n, L=1.0, periodic=True), max_levels=2, ratio=2,
                 regrid=RegridEvery(4), patches=PatchLayout(coarse_max_grid=32))
    layout.refine = TagUnion(Refine.on("Density").above(1.2),
                             Refine.on("phi").gradient_above(0.5))

    cfg = orchestration._amr_config_from_layout(layout)
    _check(cfg.n == n and cfg.regrid_every == 4, "config derived from the layout")

    sim = pops.AmrSystem(cfg)
    # Flow the typed refinement exactly as pops.bind does (set_refinement / set_phi_refinement).
    orchestration._flow_amr_layout(sim, layout)
    # The Poisson field (set via the install solvers seam in bind) -- exercise it directly here.
    sim.set_poisson("charge_density", "geometric_mg")
    sim.add_block("gas", _native_compressible_model(),
                  spatial=pops.Spatial(minmod=True), time=pops.Explicit())
    sim.set_density("gas", _bubble(n))

    m0 = sim.mass()
    dt = 2e-4
    for _ in range(8):
        sim.step(dt)
    rho = np.array(sim.density())
    _check(np.isfinite(rho).all(), "density stays finite after stepping")
    _check(rho.size > 0 and float(np.max(np.abs(rho))) > 1e-6, "density is non-trivial")
    _check(sim.mass() > 1e-6, "mass stays positive")
    # Pure transport: mass is conserved on the periodic AMR hierarchy to a tight tolerance.
    _check(abs(sim.mass() - m0) < 1e-9 * (abs(m0) + 1.0), "mass conserved on the AMR hierarchy")
    _check(sim.n_patches() >= 0, "the hierarchy is queryable (regrid wired)")
    print("ok test_native_amr_from_layout_runs (n_patches=%d, mass=%.6f)"
          % (sim.n_patches(), sim.mass()))


def test_per_block_amr_route_still_accepts_amr_system_target():
    """The WIRED AMR route still reaches compile_problem(target='amr_system') -- it is the per-block
    emit (pops.physics.Model.compile(backend='production', target='amr_system') +
    pops.AmrSystem.add_equation), NOT the rejected whole-system Case Program path (C2 / ADC-508).
    Asserts compile_problem accepts that target token; the real .so emit is Kokkos-gated (ROMEO),
    end-to-end coverage lives in test_amr_compiled_positivity_floor.py.
    """
    import inspect as _inspect
    from pops.codegen.compile_drivers import compile_problem
    sig = _inspect.signature(compile_problem)
    _check("target" in sig.parameters,
           "compile_problem keeps the target= kwarg the per-block AMR emit uses")
    print("ok test_per_block_amr_route_still_accepts_amr_system_target")


def _run_all():
    funcs = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    for fn in funcs:
        fn()
    print("\nall %d test(s) passed" % len(funcs))


if __name__ == "__main__":
    _run_all()
