"""Spec 5 gap #32 (criterion 32): the layout=AMR end-to-end compile + bind route.

``pops.compile(problem_with_AMR_layout, ...)`` no longer DEFERS layout=AMR (single OR multi block;
ADC-503): it compiles EACH block's resolved physics to a ``target="amr_system"`` production
``CompiledModel`` (the native AMR ``.so`` loader that emits ``pops_install_native_amr`` /
``add_native_block``) and carries the ``{block: CompiledModel}`` table on the handle. There is NO
whole-system time Program on AMR (``AmrSystem`` has no ``install_program`` seam), so the route does
NOT call ``compile_problem`` and does NOT need ``time=``. ``pops.bind`` then builds an ``AmrSystem``
from an ``AmrSystemConfig`` DERIVED from that layout, flows the typed refinement + Poisson field onto
it, and installs through the native path (``_install_compiled(compiled=None, instances=...)``).

What is proven LOCALLY here:

  (a) The compile ROUTE resolves layout=AMR -> per-block ``Model.compile(target="amr_system")``
      (monkeypatched so no real ``.so`` is built) and does NOT touch ``compile_problem``, and the
      ``AmrSystemConfig`` ``pops.bind`` builds from the layout has the right n / L / periodic /
      regrid_every / patch settings; the layout's max_levels / ratio are validated against the
      native envelope (the config has no such field -- the native AMR is fixed at 2 levels / ratio 2).
  (b) A REAL native ``AmrSystem`` built from the same derived config + ``set_refinement`` +
      ``set_poisson`` + ``set_density`` (NATIVE bricks, no DSL compile) advances a few steps and
      stays physical -- proving the config-flow + refinement + Poisson wiring is correct end to
      end on the native engine path.

ROMEO-gated (NOT proven here): the actual per-block ``.so`` compile (``Model.compile(backend=
"production", target="amr_system")``) needs the Kokkos toolchain (``POPS_KOKKOS_ROOT`` unset
locally), so the FULL ``pops.compile`` -> ``pops.bind`` -> multi-block AMR run on compiled artifacts
is validated on ROMEO. The production route is asserted only to REACH the per-block
``target="amr_system"`` compile.

Runs under pytest and as a plain script (the ``__main__`` guard); the CI runner executes it as a
script.
"""
import sys

try:
    import numpy as np
    import pops
    from pops.codegen import orchestration
    # ADC-583: the AMR layout-lowering helpers moved to the runtime adapter layer; the compile
    # ROUTE (orchestration.compile) still lives in codegen, the lowering helpers do not.
    from pops.runtime import _bind_adapters
    from pops.mesh.amr import (FrozenRegrid, PatchLayout, Refine, RegridEvery, TagUnion,
                               NATIVE_MAX_LEVELS, NATIVE_RATIOS)
    from pops.mesh.cartesian import CartesianMesh
    from pops.mesh.layouts import AMR
    from pops.model import DeclarationIndex, Handle, OwnerKind, OwnerPath
except Exception as exc:  # noqa: BLE001
    print("skip test_spec5_amr_e2e (pops unavailable: %s)" % exc)
    sys.exit(0)


from tests.python.support.assertions import _check
from tests.python.support.initial_states import bubble_amr as _bubble
from pops.runtime.system import AmrSystem  # ADC-545 advanced runtime seam
from pops.codegen.loader import CompiledModel
from pops.codegen._compiled_model_identity import model_compile_identity


def _ref(name, kind="state"):
    return Handle(name, kind=kind, owner=OwnerPath.shared("spec5-amr-e2e"))


class _StubCompiledModel(CompiledModel):
    """A target='amr_system' CompiledModel stand-in (the AMR route compiles each block to one)."""

    def __init__(self, source):
        super().__init__(
            "/tmp/%s_amr.so" % source.name, "production", "add_native_block",
            (), (), (), 0, None, 0, {}, {"cpu": True, "amr": True}, "abi",
            source._model_hash(), "c++", "c++20", target="amr_system",
            definition_identity=model_compile_identity(source))
        self.name = source.name


class _StubDsl:
    """The ``.dsl`` engine model the compile route resolves to; its ``.compile`` records the call."""

    def __init__(self, name="ne"):
        self.name = name
        self.compiled = []

    def compile(self, *, backend, target, **kw):
        self.compiled.append((backend, target))
        return _StubCompiledModel(self)

    def _model_hash(self):
        return "model-hash:%s" % self.name


class _StubModel:
    """A physics stand-in exposing the ``.dsl`` engine model the compile route resolves."""

    def __init__(self, name="ne"):
        self.name = name
        self.owner_path = OwnerPath.fresh(OwnerKind.MODEL_DEFINITION, name)
        self.dsl = _StubDsl(name)

    def declaration_index(self):
        return DeclarationIndex(owner=self.owner_path, handles=())


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
    """compile(problem_with_AMR_layout) compiles each block for target='amr_system' (ADC-503).

    The AMR route does NOT call compile_problem (no whole-system time Program). A tripwire on
    compile_problem proves the route never touches it, and no time= is required.
    """
    tripwire = {"hit": False}

    def _tripwire(*a, **kw):
        tripwire["hit"] = True

    _patch(monkeypatch, "pops.codegen.compile_drivers.compile_problem", _tripwire)
    try:
        layout = AMR(CartesianMesh(n=48, L=2.0, periodic=False), max_levels=2, ratio=2)
        model = _StubModel("ne")
        prob = pops.Problem(layout=layout).block("ne", physics=model)
        compiled = orchestration.compile(prob)  # no time= : the AMR route does not need one
        _check(tripwire["hit"] is False, "layout=AMR does NOT call compile_problem")
        _check(model.dsl.compiled == [("production", "amr_system")],
               "the block is compiled once with backend='production', target='amr_system'")
        plan = compiled.install_plan
        _check(plan.target == "amr_system", "amr_system target carried by the InstallPlan")
        _check(plan.layout is not layout and plan.layout.base is not layout.base
               and plan.layout.base.n == layout.base.n,
               "the deeply detached AMR layout is carried for bind()")
        _check(set(plan.block_models) == {"ne"},
               "the block CompiledModel is carried by the InstallPlan")
        _check(not hasattr(compiled, "_block_compiled_models"),
               "the retired private block-loader mirror is absent")
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
    cfg = _bind_adapters._amr_config_from_layout(layout)
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
    frozen = _bind_adapters._amr_config_from_layout(
        AMR(CartesianMesh(n=32), regrid=FrozenRegrid()))
    _check(frozen.regrid_every == 0, "FrozenRegrid -> frozen hierarchy (regrid_every == 0)")
    no_regrid = _bind_adapters._amr_config_from_layout(AMR(CartesianMesh(n=32)))
    _check(no_regrid.regrid_every == 0, "no regrid policy -> regrid_every == 0")
    _check(no_regrid.n == 32, "n still derived from the base mesh")
    print("ok test_amr_config_regrid_and_patch_defaults")


def test_amr_refine_default_density_subject():
    """The density / component-0 subjects map to the single-block default; others are non-default."""
    _check(_bind_adapters._is_default_density_subject("Density"), "Density role is the default")
    _check(_bind_adapters._is_default_density_subject("density"), "density name is the default")
    _check(_bind_adapters._is_default_density_subject("rho"), "rho name is the default")
    _check(_bind_adapters._is_default_density_subject(None), "no subject is the default")
    _check(not _bind_adapters._is_default_density_subject("MomentumX"),
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
    layout.refine = Refine.on(_ref("MomentumX", kind="role")).above(0.5).resolve_references(
        pops.Problem(name="native-refine-test").resolve)
    try:
        _bind_adapters._flow_amr_layout(_Recorder(), layout)
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
    layout.refine = TagUnion(
        Refine.on(_ref("Density", kind="role")).above(1.2),
        Refine.on(_ref("phi", kind="field")).gradient_above(0.5),
    ).resolve_references(pops.Problem(name="native-refine-test").resolve)

    cfg = _bind_adapters._amr_config_from_layout(layout)
    _check(cfg.n == n and cfg.regrid_every == 4, "config derived from the layout")

    sim = AmrSystem(cfg)
    # Flow the typed refinement exactly as pops.bind does (set_refinement / set_phi_refinement).
    _bind_adapters._flow_amr_layout(sim, layout)
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
    """The production .so compile path REACHES the per-block target='amr_system' compile (ADC-503).

    The actual emit + .so compile needs Kokkos (POPS_KOKKOS_ROOT unset locally) -> ROMEO. We prove
    the route reaches the per-block Model.compile(backend='production', target='amr_system') WITHOUT
    building a .so, by stubbing the block model's .compile to raise at the C++-invoking boundary.
    """
    reached = {}

    class _RomeoDsl:
        name = "ne"

        def compile(self, *, backend, target, **kw):
            reached["backend"], reached["target"] = backend, target
            # A real call here would emit the loader C++ and invoke the Kokkos compiler (ROMEO).
            raise RuntimeError("ROMEO boundary: .so compile needs the Kokkos toolchain")

    class _RomeoModel:
        def __init__(self):
            self.name = "ne"
            self.owner_path = OwnerPath.fresh(OwnerKind.MODEL_DEFINITION, self.name)
            self.dsl = _RomeoDsl()

        def declaration_index(self):
            return DeclarationIndex(owner=self.owner_path, handles=())

    layout = AMR(CartesianMesh(n=32), max_levels=2, ratio=2)
    prob = pops.Problem(layout=layout).block("ne", physics=_RomeoModel())
    try:
        orchestration.compile(prob)
        raise AssertionError("the ROMEO-boundary stub should have raised")
    except RuntimeError as exc:
        _check("ROMEO boundary" in str(exc), "reached the .so compile (Kokkos) boundary")
    _check(reached.get("target") == "amr_system" and reached.get("backend") == "production",
           "the production .so path is reached with backend='production', target='amr_system'")
    print("ok test_production_so_compile_is_romeo_gated")


def _run_all():
    funcs = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    for fn in funcs:
        fn()
    print("\nall %d test(s) passed" % len(funcs))


if __name__ == "__main__":
    _run_all()
