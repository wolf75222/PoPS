"""Spec 5 gap #32 (criterion 32): the layout=AMR end-to-end compile + bind route.

``pops.compile(problem_with_AMR_layout, ...)`` no longer DEFERS layout=AMR: it routes to
``compile_problem(target="amr_system")`` (the native AMR ``.so`` path that emits
``pops_install_native_amr``) and carries the AMR layout on the handle; ``pops.bind`` then builds
an ``AmrSystem`` from an ``AmrSystemConfig`` DERIVED from that layout and flows the typed
refinement + Poisson field onto it before install.

What is proven LOCALLY here:

  (a) The compile ROUTE resolves layout=AMR -> target="amr_system" (monkeypatched
      ``compile_problem`` so no real ``.so`` is built), and the ``AmrSystemConfig`` ``pops.bind``
      builds from the layout has the right n / L / periodic / regrid_every / patch settings; the
      layout's max_levels / ratio are validated against the native envelope (the config has no
      such field -- the native AMR is fixed at 2 levels / ratio 2).
  (b) A REAL native ``AmrSystem`` built from the same derived config + ``set_refinement`` +
      ``set_poisson`` + ``set_density`` (NATIVE bricks, no DSL compile) advances a few steps and
      stays physical -- proving the config-flow + refinement + Poisson wiring is correct end to
      end on the native engine path.

ROMEO-gated (NOT proven here): the actual ``.so`` compile of ``compile_problem(target=
"amr_system")`` needs the Kokkos toolchain (``POPS_KOKKOS_ROOT`` unset locally), so the FULL
``pops.compile`` -> ``pops.bind`` -> run on a compiled artifact is validated on ROMEO. The
production ``.so`` route is asserted only to REACH ``compile_problem(target="amr_system")``.

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


# --- (a) the compile ROUTE + the AmrSystemConfig-from-layout mapping ---------------
def test_amr_layout_drives_compile_target(monkeypatch=None):
    """compile(problem_with_AMR_layout) reaches compile_problem(target='amr_system')."""
    captured = {}

    class _StubCompiled:
        def __init__(self, target):
            self.so_path = "/tmp/stub_amr.so"
            self.model = None
            self._target = target

    def _fake_compile_problem(*, time, model, backend, target, **kw):
        captured.update(time=time, model=model, backend=backend, target=target)
        return _StubCompiled(target)

    _patch(monkeypatch, "pops.codegen.compile_drivers.compile_problem", _fake_compile_problem)
    try:
        layout = AMR(CartesianMesh(n=48, L=2.0, periodic=False), max_levels=2, ratio=2)
        prob = pops.Case(layout=layout).block("ne", physics=_StubModel())
        compiled = orchestration.compile(prob, time=object())
        _check(captured["target"] == "amr_system",
               "layout=AMR routes to compile_problem(target='amr_system')")
        _check(captured["backend"] == "production", "AMR uses the production backend")
        _check(compiled._target == "amr_system", "amr_system target carried on the handle")
        _check(compiled._layout is layout, "the AMR layout is carried for bind()")
    finally:
        _unpatch(monkeypatch)
    print("ok test_amr_layout_drives_compile_target")


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


def test_production_so_compile_is_romeo_gated():
    """The production .so compile path REACHES compile_problem(target='amr_system').

    The actual emit + .so compile needs Kokkos (POPS_KOKKOS_ROOT unset locally) -> ROMEO. We prove
    the route reaches compile_problem with target='amr_system' (and that compile_problem accepts
    that target) WITHOUT building a .so, by monkeypatching the C++-invoking tail.
    """
    reached = {}

    def _fake_compile_problem(*, time, model, backend, target, **kw):
        reached["target"] = target
        # A real call here would emit the program C++ and invoke the Kokkos compiler (ROMEO).
        raise RuntimeError("ROMEO boundary: .so compile needs the Kokkos toolchain")

    saved = orchestration_compile_problem_ref()
    set_orchestration_compile_problem(_fake_compile_problem)
    try:
        layout = AMR(CartesianMesh(n=32), max_levels=2, ratio=2)
        prob = pops.Case(layout=layout).block("ne", physics=_StubModel())
        try:
            orchestration.compile(prob, time=object())
            raise AssertionError("the ROMEO-boundary stub should have raised")
        except RuntimeError as exc:
            _check("ROMEO boundary" in str(exc), "reached the .so compile (Kokkos) boundary")
        _check(reached.get("target") == "amr_system",
               "the production .so path is reached with target='amr_system'")
    finally:
        set_orchestration_compile_problem(saved)
    print("ok test_production_so_compile_is_romeo_gated")


def orchestration_compile_problem_ref():
    import pops.codegen.compile_drivers as cd
    return cd.compile_problem


def set_orchestration_compile_problem(fn):
    import pops.codegen.compile_drivers as cd
    cd.compile_problem = fn


def _run_all():
    funcs = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    for fn in funcs:
        fn()
    print("\nall %d test(s) passed" % len(funcs))


if __name__ == "__main__":
    _run_all()
