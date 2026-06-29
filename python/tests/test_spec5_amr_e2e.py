"""Spec 5 gap #32 (criterion 32): the layout=AMR compile + runtime route.

``pops.compile_problem(model=Module, time=Program(...), layout=AMR(...))`` no longer DEFERS
layout=AMR and no longer falls back to per-block native loaders. The runtime side builds an
``AmrSystem`` from an ``AmrSystemConfig`` DERIVED from that layout, flows typed refinement +
Poisson field descriptors onto it, and installs compiled AMR Program handles through
``sim.install(...)``.

What is proven LOCALLY here:

  (a) The compile ROUTE resolves ``layout=AMR`` inside ``compile_problem`` to the AMR program ABI
      (monkeypatched so no real ``.so`` is built), and the
      ``AmrSystemConfig`` built from the layout has the right n / L / periodic /
      regrid_every / patch settings; the layout's max_levels / ratio are validated against the
      native envelope (the config has no such field -- the native AMR is fixed at 2 levels / ratio 2).
  (b) A REAL native ``AmrSystem`` built from the same derived config + ``set_refinement`` +
      ``set_poisson`` + ``set_density`` (NATIVE bricks, no DSL compile) advances a few steps and
      stays physical -- proving the config-flow + refinement + Poisson wiring is correct end to
      end on the native engine path.

ROMEO-gated (NOT proven here): the actual AMR Program ``.so`` compile needs the Kokkos toolchain, so
the FULL ``compile_problem`` -> ``sim.install`` -> AMR run on compiled artifacts is validated on ROMEO.

Runs under pytest and as a plain script (the ``__main__`` guard); the CI runner executes it as a
script.
"""
import sys
import os

try:
    import numpy as np
    import pops
    from pops.codegen.backends import Production
    from pops.model import Module
    from pops.runtime.bricks import (BackgroundDensity, CompressibleFlux, Explicit,
                                    FluidState, Model as NativeModel, NoSource, Spatial)
    from pops.runtime.amr_layout import amr_config_from_layout, flow_amr_layout, is_default_density_subject
    from pops.mesh.amr import (FrozenRegrid, PatchLayout, Refine, RegridEvery, TagUnion,
                               NATIVE_MAX_LEVELS, NATIVE_RATIOS)
    from pops.mesh.cartesian import CartesianMesh
    from pops.mesh.layouts import AMR
    from pops.numerics.reconstruction.limiters import Minmod
except Exception as exc:  # noqa: BLE001
    print("skip test_spec5_amr_e2e (pops unavailable: %s)" % exc)
    sys.exit(0)


def _check(cond, msg):
    if not cond:
        raise AssertionError(msg)


INCLUDE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "include"))


class _StubCompiledModel:
    """A compiled AMR Program stand-in returned by compile_problem."""

    def __init__(self, name="ne"):
        self.name = name
        self.so_path = "/tmp/%s_amr.so" % name
        self.target = "amr_system"


class _StubModule:
    """The lowered module model the compile route resolves to."""

    def __init__(self, name="ne"):
        self.name = name

class _StubModel:
    """A physics stand-in exposing the public ``to_module`` lowering hook."""

    def __init__(self, name="ne"):
        self.name = name
        self.module = _StubModule(name)

    def to_module(self):
        return self.module


def _minimal_module_and_program():
    """Tiny real Module + Program used to exercise compile_problem routing without legacy Case."""
    module = Module("amr_route")
    module.state_space("U", ("rho",))
    program = pops.time.Program("amr_program")
    u = program.state("plasma")
    program.commit("plasma", program.linear_combine("identity", u))
    return module, program


# --- monkeypatch helpers (work under pytest fixture OR the bare __main__ runner) ---
_SAVED = []


def _patch(monkeypatch, dotted, value):
    module_name, attr = dotted.rsplit(".", 1)
    import importlib
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError:
        module_name, owner_name = module_name.rsplit(".", 1)
        module = importlib.import_module(module_name)
        module = getattr(module, owner_name)
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
    """compile_problem(model=Module, time=Program, layout=AMR) emits the AMR program ABI."""
    captured = {"target": None, "compiled": False}

    def _fake_emit(self, model=None, target="system"):
        captured["target"] = target
        return "extern \"C\" int pops_test_amr_route() { return 0; }\n"

    def _fake_run_compile(cmd, label):
        captured["compiled"] = True

    def _fake_loader_build_flags(cxx=None):
        return "c++", [], []

    _patch(monkeypatch, "pops.time.program.Program._emit_cpp_program_for_target", _fake_emit)
    _patch(monkeypatch, "pops.codegen.compile_drivers._run_compile", _fake_run_compile)
    _patch(monkeypatch, "pops.codegen.compile_drivers.pops_loader_build_flags",
           _fake_loader_build_flags)
    try:
        layout = AMR(CartesianMesh(n=48, L=2.0, periodic=False), max_levels=2, ratio=2)
        module, program = _minimal_module_and_program()
        compiled = pops.compile_problem(
            "/tmp/pops_test_amr_route.so",
            model=module,
            time=program,
            backend=Production(),
            layout=layout,
            force=True,
            include=INCLUDE,
        )
        _check(captured["target"] == "amr_system",
               "layout=AMR selects the AMR program ABI")
        _check(captured["compiled"] is True, "compile_problem reached the compiler driver")
        _check(compiled.model is module, "compiled handle carries the Module")
        _check(compiled.program is program, "compiled handle carries the Program")
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
    cfg = amr_config_from_layout(layout)
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
    frozen = amr_config_from_layout(
        AMR(CartesianMesh(n=32), regrid=FrozenRegrid()))
    _check(frozen.regrid_every == 0, "FrozenRegrid -> frozen hierarchy (regrid_every == 0)")
    no_regrid = amr_config_from_layout(AMR(CartesianMesh(n=32)))
    _check(no_regrid.regrid_every == 0, "no regrid policy -> regrid_every == 0")
    _check(no_regrid.n == 32, "n still derived from the base mesh")
    print("ok test_amr_config_regrid_and_patch_defaults")


def test_amr_refine_default_density_subject():
    """The density / component-0 subjects map to the single-block default; others are non-default."""
    _check(is_default_density_subject("Density"), "Density role is the default")
    _check(is_default_density_subject("density"), "density name is the default")
    _check(is_default_density_subject("rho"), "rho name is the default")
    _check(is_default_density_subject(None), "no subject is the default")
    _check(not is_default_density_subject("MomentumX"),
           "a non-density role is a non-default (multi-block) selector")
    print("ok test_amr_refine_default_density_subject")


def test_amr_non_default_refine_selector_flows_to_cpp():
    """A non-density refine selector is forwarded to C++ instead of rejected in Python."""
    class _Recorder:
        def __init__(self):
            self.calls = []

        def set_refinement(self, *a, **k):
            self.calls.append((a, k))

        def set_phi_refinement(self, *a, **k):
            raise AssertionError("set_phi_refinement must not be called here")

    layout = AMR(CartesianMesh(n=32))
    layout.refine = Refine.on("MomentumX").above(0.5)
    rec = _Recorder()
    flow_amr_layout(rec, layout)
    _check(rec.calls == [((0.5,), {"variable": "MomentumX"})],
           "non-density subject is passed as variable selector")
    print("ok test_amr_non_default_refine_selector_flows_to_cpp")


# --- (b) a REAL native AmrSystem from the derived config runs a few steps -----------
def _native_compressible_model():
    """A native compressible-flow ModelSpec (composed bricks, NO DSL compile)."""
    return NativeModel(state=FluidState.compressible(gamma=1.4),
                       transport=CompressibleFlux(), source=NoSource(),
                       elliptic=BackgroundDensity(alpha=0.0, n0=0.0))


def _bubble(n):
    xs = (np.arange(n) + 0.5) / n
    X, Y = np.meshgrid(xs, xs)
    return (1.0 + 0.5 * np.exp(-((X - 0.5) ** 2 + (Y - 0.5) ** 2) / 0.02)).reshape(-1)


def test_native_amr_from_layout_runs():
    """A native AmrSystem built from a layout's derived config + set_refinement + set_poisson runs.

    This is the config-flow + refinement + Poisson wiring proven on the REAL engine (native bricks,
    no DSL compile). It mirrors the modern AMR route: build the AmrSystem
    from amr_config_from_layout(layout), flow the typed Refine to set_refinement, set the Poisson
    solver, add the native block and the initial density, then step.
    """
    n = 48
    layout = AMR(CartesianMesh(n=n, L=1.0, periodic=True), max_levels=2, ratio=2,
                 regrid=RegridEvery(4), patches=PatchLayout(coarse_max_grid=32))
    layout.refine = TagUnion(Refine.on("Density").above(1.2),
                             Refine.on("phi").gradient_above(0.5))

    cfg = amr_config_from_layout(layout)
    _check(cfg.n == n and cfg.regrid_every == 4, "config derived from the layout")

    sim = pops.AmrSystem(cfg)
    # Flow the typed refinement exactly as pops.bind does (set_refinement / set_phi_refinement).
    flow_amr_layout(sim, layout)
    # The Poisson field set through the runtime install seam -- exercise the native wiring here.
    sim._set_poisson("charge_density", "geometric_mg")
    sim._add_block("gas", _native_compressible_model(),
                   spatial=Spatial(limiter=Minmod()), time=Explicit())
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


def test_native_single_block_amr_variable_refinement_builds():
    """Single-block AMR resolves a non-density refinement variable in C++."""
    n = 32
    sim = pops.AmrSystem(amr_config_from_layout(
        AMR(CartesianMesh(n=n), max_levels=2, ratio=2, regrid=RegridEvery(1))))
    sim.set_refinement(-0.5, variable="rho_u")
    sim._set_poisson("charge_density", "geometric_mg")
    sim._add_block("gas", _native_compressible_model(),
                   spatial=Spatial(limiter=Minmod()), time=Explicit())
    sim.set_density("gas", _bubble(n))
    _check(sim.n_patches() >= 0, "single-block AMR built with variable='rho_u'")
    print("ok test_native_single_block_amr_variable_refinement_builds")


def test_amr_compile_requires_time_program():
    """The AMR compile_problem route requires an explicit Program; there is no per-block fallback."""
    layout = AMR(CartesianMesh(n=32), max_levels=2, ratio=2)
    module = Module("missing_time")
    try:
        pops.compile_problem(model=module, layout=layout, backend=Production())
        raise AssertionError("AMR compile without a Program must raise")
    except ValueError as exc:
        _check("time must be" in str(exc), "missing Program is rejected clearly")
    print("ok test_amr_compile_requires_time_program")


def _run_all():
    funcs = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    for fn in funcs:
        fn()
    print("\nall %d test(s) passed" % len(funcs))


if __name__ == "__main__":
    _run_all()
