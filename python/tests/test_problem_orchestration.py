"""ADC-492 (Spec 5 sec.5.16 / sec.11): pops.Case assembly + pops.compile/bind + PhysicsModel.

These are PURE-PYTHON tests of the inert assembly, the alias, and the thin dispatch wiring.
The real ``.so`` compile (``compile_problem``) and the runtime install/run are Kokkos-gated
and validated on CI / ROMEO, so the dispatch tests MONKEYPATCH ``compile_problem`` /
``System`` / ``AmrSystem`` to assert routing WITHOUT a real compile. Every deferred route is
asserted to raise loudly (never fake success).

Runs both under pytest and as a plain script (``python3 test_problem_orchestration.py``); the
CI runner executes it as a script (the ``__main__`` guard below).
"""
import sys

try:
    import pops
    from pops.mesh.cartesian import CartesianMesh
    from pops.mesh.layouts import AMR, Uniform
    from pops.fields import FieldProblem
    from pops.math import laplacian
    from pops.codegen import orchestration
except Exception as exc:  # noqa: BLE001
    print("skip test_problem_orchestration (pops unavailable: %s)" % exc)
    sys.exit(0)


# --- tiny stand-ins (no compiler / no runtime) -----------------------------
class _StubModel:
    """A physics stand-in exposing the ``.dsl`` engine model pops.compile resolves."""

    def __init__(self, name="stub"):
        self.name = name
        self.dsl = object()  # what _resolve_problem_model returns


class _StubSolver:
    name = "GeometricMG"
    scheme = "geometric_mg"
    options = {}


class _StubCompiled:
    """A compiled-handle stand-in: only ``.so_path`` + the carried problem/target/layout."""

    def __init__(self, target="system", problem=None, layout=None):
        self.so_path = "/tmp/stub.so"
        self.model = None
        self._target = target
        self._problem = problem
        self._layout = layout


def _poisson_problem():
    """A minimal valid Poisson FieldProblem named 'phi' (the default-served field)."""
    return FieldProblem(name="phi", unknown="phi",
                        equation=(-laplacian("phi") == "charge_density"),
                        solver=_StubSolver())


def _check(cond, msg):
    if not cond:
        raise AssertionError(msg)


# --- assembly + chaining + inspect -----------------------------------------
def test_assembly_chaining_and_inspect():
    model = _StubModel("ne")
    prob = (pops.Case(name="plasma")
            .block("ne", physics=model, spatial=None)
            .param(pops.physics.ConstParam("alpha", 1.0))
            .aux("B_z", value=None))
    _check(prob is prob.block.__self__, "setters operate on the same problem")
    _check(prob.layout.name == "Uniform", "default layout is Uniform")
    info = prob.inspect()
    _check(info["name"] == "plasma", "name carried")
    _check(set(info["blocks"]) == {"ne"}, "block recorded")
    _check(info["params"]["alpha"]["default"] == 1.0, "param recorded")
    _check("B_z" in info["aux"], "aux recorded")
    _check(prob.options()["n_blocks"] == 1, "options report n_blocks")
    print("ok test_assembly_chaining_and_inspect")


def test_block_requires_physics_and_no_duplicate():
    prob = pops.Case()
    try:
        prob.block("ne", physics=None)
        raise AssertionError("block with no physics must raise")
    except ValueError:
        pass
    prob.block("ne", physics=_StubModel())
    try:
        prob.block("ne", physics=_StubModel())
        raise AssertionError("duplicate block must raise")
    except ValueError:
        pass
    print("ok test_block_requires_physics_and_no_duplicate")


def test_field_type_checked():
    prob = pops.Case()
    try:
        prob.field("not a FieldProblem")
        raise AssertionError("field must reject a non-FieldProblem")
    except TypeError:
        pass
    prob.field(_poisson_problem())
    _check("phi" in prob._fields, "field registered by name")
    print("ok test_field_type_checked")


def test_amr_property():
    # Uniform layout -> amr raises ValueError.
    try:
        pops.Case().amr
        raise AssertionError("amr on a Uniform layout must raise")
    except ValueError:
        pass
    # AMR layout -> returns a handle whose .refine chains back to the problem.
    prob = pops.Case(layout=AMR(CartesianMesh()))
    handle = prob.amr
    from pops.mesh.amr import RegridEvery
    returned = handle.refine(regrid=RegridEvery(20))
    _check(returned is prob, "amr.refine chains back to the problem")
    _check(prob.layout.regrid is not None, "refine recorded the regrid policy")
    print("ok test_amr_property")


# --- validate(): structural pass + each deferred case raising ---------------
def test_validate_structural_pass():
    prob = pops.Case().block("ne", physics=_StubModel()).field(_poisson_problem())
    _check(prob.validate() is True, "a single-block + Poisson-field problem validates")
    _check(bool(prob.available()) is True, "available() is yes for a valid problem")
    print("ok test_validate_structural_pass")


def test_validate_requires_a_block():
    try:
        pops.Case().validate()
        raise AssertionError("a problem with no block must not validate")
    except ValueError:
        pass
    print("ok test_validate_requires_a_block")


def test_validate_multi_block_deferred():
    prob = (pops.Case().block("ne", physics=_StubModel())
            .block("ni", physics=_StubModel()))
    try:
        prob.validate()
        raise AssertionError("multi-block must raise NotImplementedError")
    except NotImplementedError as exc:
        _check("multi-block" in str(exc), "multi-block message is explicit")
    print("ok test_validate_multi_block_deferred")


def test_validate_non_poisson_field_deferred():
    field = FieldProblem(name="temperature", unknown="T",
                         equation=(-laplacian("T") == "src"), solver=_StubSolver())
    prob = pops.Case().block("ne", physics=_StubModel()).field(field)
    try:
        prob.validate()
        raise AssertionError("a non-Poisson field must raise NotImplementedError")
    except NotImplementedError as exc:
        _check("non-Poisson" in str(exc), "non-Poisson message is explicit")
    print("ok test_validate_non_poisson_field_deferred")


def test_case_has_no_output_surface():
    """C4 (ADC-509): the decorative Case.output(...) surface is removed (no codegen / runtime).
    A Case exposes no output() / checkpoint() setter and no _outputs plumbing."""
    case = pops.Case().block("ne", physics=_StubModel())
    _check(not hasattr(case, "output"), "Case.output() removed (decorative API)")
    _check(not hasattr(case, "checkpoint"), "Case.checkpoint() removed (decorative API)")
    _check(not hasattr(case, "_outputs"), "Case._outputs plumbing removed")
    _check("n_outputs" not in case.options(), "options() drops n_outputs")
    _check("outputs" not in case.inspect(), "inspect() drops outputs")
    # validate() no longer carries an output reject; a plain single-block case validates.
    _check(case.validate() is True, "a case with no output surface still validates")
    print("ok test_case_has_no_output_surface")


def test_validate_name_collision():
    field = _poisson_problem()
    prob = pops.Case().block("phi", physics=_StubModel()).field(field)
    try:
        prob.validate()
        raise AssertionError("a block/field name collision must raise")
    except ValueError as exc:
        _check("share name" in str(exc), "collision message is explicit")
    print("ok test_validate_name_collision")


# --- PhysicsModel alias identity -------------------------------------------
def test_physics_model_alias_identity():
    _check(pops.physics.PhysicsModel is pops.physics.Model, "alias is the same class object")
    _check(pops.PhysicsModel is pops.physics.Model, "top-level alias is the same class object")
    _check(pops.physics.Model.__name__ == "Model", "class __name__ stays 'Model' (alias not rename)")
    # Existing pops.physics.Model consumers keep working.
    instance = pops.physics.Model("legacy")
    _check(type(instance).__name__ == "Model", "instances still report __name__ == 'Model'")
    _check(isinstance(instance, pops.PhysicsModel), "PhysicsModel is usable in isinstance")
    print("ok test_physics_model_alias_identity")


# --- compile(): layout-driven dispatch via a monkeypatched compile_problem --
def test_compile_layout_drives_target(monkeypatch=None):
    captured = {}

    def _fake_compile_problem(*, time, model, backend, target, **kw):
        captured.update(time=time, model=model, backend=backend, target=target)
        return _StubCompiled(target=target)

    _patch(monkeypatch, "pops.codegen.compile_drivers.compile_problem", _fake_compile_problem)
    try:
        prob = pops.Case(name="u").block("ne", physics=_StubModel())
        compiled = orchestration.compile(prob, time=object())
        _check(captured["target"] == "system", "Uniform layout routes to target='system'")
        _check(captured["backend"] == "production", "default backend forwarded")
        _check(compiled._target == "system", "target carried on the handle")
        _check(compiled._problem is prob, "problem carried on the handle")
    finally:
        _unpatch(monkeypatch)
    print("ok test_compile_layout_drives_target")


def test_compile_amr_time_program_rejected(monkeypatch=None):
    """C2 (ADC-508): a whole-system time Program on an AMR layout is rejected EARLY at compile()
    -- BEFORE any .so is built -- never at bind() (no transitional compile-succeeds-then-bind-fails).
    The reject names the wired per-block AMR path. compile_problem is monkeypatched to a tripwire so
    a leak past the early reject (an actual compile attempt) is caught."""
    called = {"compile_problem": False}

    def _tripwire(*a, **kw):
        called["compile_problem"] = True
        return _StubCompiled(target="amr_system")

    _patch(monkeypatch, "pops.codegen.compile_drivers.compile_problem", _tripwire)
    try:
        layout = AMR(CartesianMesh())
        prob = pops.Case(layout=layout).block("ne", physics=_StubModel())
        try:
            orchestration.compile(prob, time=object())
            raise AssertionError("an AMR whole-system time Program must be rejected at compile()")
        except NotImplementedError as exc:
            _check("ADC-508" in str(exc), "C2 reject names the deferred issue ADC-508")
            _check("amr_system" in str(exc) and "add_equation" in str(exc),
                   "C2 reject redirects to the wired per-block AMR path")
        _check(called["compile_problem"] is False,
               "the reject fires BEFORE compile_problem (no .so built; not a bind-time reject)")
    finally:
        _unpatch(monkeypatch)
    print("ok test_compile_amr_time_program_rejected")


def test_compile_amr_max_levels_beyond_native_raises():
    # max_levels above the native envelope is refused at compile (problem.validate() runs the
    # layout's AMR.available check), with the existing clear message, never silently clamped.
    from pops.mesh.amr import NATIVE_MAX_LEVELS
    prob = (pops.Case(layout=AMR(CartesianMesh(), max_levels=NATIVE_MAX_LEVELS + 1))
            .block("ne", physics=_StubModel()))
    try:
        orchestration.compile(prob, time=object())
        raise AssertionError("AMR(max_levels beyond native) must be refused")
    except ValueError as exc:
        _check("max_levels" in str(exc), "max-levels message is explicit")
    print("ok test_compile_amr_max_levels_beyond_native_raises")


def test_compile_missing_time_raises():
    prob = pops.Case().block("ne", physics=_StubModel())
    try:
        orchestration.compile(prob)  # no time= and no problem.time(...)
        raise AssertionError("missing time scheme must raise (no silent default)")
    except NotImplementedError as exc:
        _check("time scheme is required" in str(exc), "missing-time message is explicit")
    print("ok test_compile_missing_time_raises")


def test_compile_problem_time_setter_honored(monkeypatch=None):
    captured = {}

    def _fake_compile_problem(*, time, model, backend, target, **kw):
        captured.update(time=time, target=target)
        return _StubCompiled(target=target)

    _patch(monkeypatch, "pops.codegen.compile_drivers.compile_problem", _fake_compile_problem)
    try:
        sentinel = object()
        prob = pops.Case().block("ne", physics=_StubModel()).time(sentinel)
        orchestration.compile(prob)  # time taken from problem._time
        _check(captured["time"] is sentinel, "problem.time(...) is honored when time= omitted")
    finally:
        _unpatch(monkeypatch)
    print("ok test_compile_problem_time_setter_honored")


def test_compile_multi_block_raises(monkeypatch=None):
    prob = (pops.Case().block("ne", physics=_StubModel())
            .block("ni", physics=_StubModel()))
    try:
        orchestration.compile(prob, time=object())
        raise AssertionError("multi-block compile must raise")
    except NotImplementedError:
        pass
    print("ok test_compile_multi_block_raises")


# --- bind(): System vs AmrSystem dispatch via a monkeypatched runtime -------
class _RecordingSim:
    """A System / AmrSystem stand-in that records the _install_compiled(...) seam call."""

    last = {}

    def _install_compiled(self, compiled=None, *, instances=None, params=None, aux=None,
                          solvers=None, cadence=None):
        _RecordingSim.last = {"compiled": compiled, "instances": instances, "params": params,
                              "aux": aux, "solvers": solvers, "cadence": cadence}


def _bind_with_stub_runtime(target, layout=None):
    """Run bind() with System/AmrSystem replaced by recording stubs; return the chosen class.

    The AmrSystem stub mirrors the real constructor (it accepts the derived ``AmrSystemConfig``)
    and records the refinement flow (set_refinement / set_phi_refinement) bind() applies before
    install."""
    import pops.runtime.system as rtsys

    class _StubSystem(_RecordingSim):
        pass

    class _StubAmrSystem(_RecordingSim):
        def __init__(self, config=None):
            self.config = config
            self.refinement = None
            self.phi_refinement = None

        def set_refinement(self, threshold, variable="", role=""):
            self.refinement = (threshold, variable, role)

        def set_phi_refinement(self, threshold):
            self.phi_refinement = threshold

    orig_sys, orig_amr = rtsys.System, rtsys.AmrSystem
    rtsys.System, rtsys.AmrSystem = _StubSystem, _StubAmrSystem
    try:
        prob = pops.Case(layout=layout) if layout is not None else pops.Case()
        prob = prob.block("ne", physics=_StubModel()).field(_poisson_problem())
        compiled = _StubCompiled(target=target, problem=prob, layout=layout)
        sim = orchestration.bind(compiled, initial_state={"ne": [1.0]})
        return type(sim), _RecordingSim.last, _StubSystem, _StubAmrSystem, sim
    finally:
        rtsys.System, rtsys.AmrSystem = orig_sys, orig_amr


def test_bind_system_dispatch():
    sim_class, last, stub_system, _, _ = _bind_with_stub_runtime("system")
    _check(sim_class is stub_system, "target='system' binds a System")
    _check(last["compiled"].so_path == "/tmp/stub.so", "compiled handle passed to install")
    _check(set(last["instances"]) == {"ne"}, "the block becomes one install instance")
    _check(last["instances"]["ne"]["initial"] == [1.0], "initial state routed by block name")
    _check("phi" in last["solvers"], "the Poisson field solver derived from the problem")
    print("ok test_bind_system_dispatch")


def test_bind_amr_dispatch():
    from pops.mesh.amr import RegridEvery, PatchLayout, Refine
    layout = AMR(CartesianMesh(n=48, L=2.0, periodic=False), max_levels=2, ratio=2,
                 regrid=RegridEvery(4), patches=PatchLayout(distribute_coarse=True,
                                                            coarse_max_grid=16))
    layout.refine = Refine.on("density").above(1.5)
    sim_class, last, _, stub_amr, sim = _bind_with_stub_runtime("amr_system", layout=layout)
    _check(sim_class is stub_amr, "target='amr_system' binds an AmrSystem")
    cfg = sim.config
    _check(cfg.n == 48 and cfg.L == 2.0 and cfg.periodic is False,
           "AmrSystemConfig n/L/periodic derived from the base CartesianMesh")
    _check(cfg.regrid_every == 4, "regrid_every derived from RegridEvery(4)")
    _check(cfg.distribute_coarse is True and cfg.coarse_max_grid == 16,
           "patch settings derived from PatchLayout")
    _check(sim.refinement == (1.5, "", ""),
           "the density Refine criterion flowed to set_refinement (component 0) before install")
    _check(last["compiled"].so_path == "/tmp/stub.so", "compiled handle passed to AMR install")
    print("ok test_bind_amr_dispatch")


def test_bind_amr_frozen_and_phi_refinement():
    from pops.mesh.amr import FrozenRegrid, TagUnion, Refine
    layout = AMR(CartesianMesh(n=32), regrid=FrozenRegrid())
    layout.refine = TagUnion(Refine.on("Density").above(2.0),
                             Refine.on("phi").gradient_above(0.5))
    _, _, _, _, sim = _bind_with_stub_runtime("amr_system", layout=layout)
    _check(sim.config.regrid_every == 0, "FrozenRegrid -> regrid_every == 0")
    _check(sim.refinement == (2.0, "", ""),
           "a Density-role subject flows to set_refinement on component 0")
    _check(sim.phi_refinement == 0.5,
           "the grad-phi tag flows to set_phi_refinement")
    print("ok test_bind_amr_frozen_and_phi_refinement")


def test_bind_rejects_non_compiled():
    try:
        orchestration.bind(object())
        raise AssertionError("bind must reject a handle without .so_path")
    except TypeError:
        pass
    print("ok test_bind_rejects_non_compiled")


def test_bind_unknown_initial_state_raises():
    prob = pops.Case().block("ne", physics=_StubModel())
    compiled = _StubCompiled(problem=prob)
    import pops.runtime.system as rtsys
    orig = rtsys.System
    rtsys.System = _RecordingSim
    try:
        orchestration.bind(compiled, initial_state={"nope": [0.0]})
        raise AssertionError("an unknown block in initial_state must raise")
    except ValueError as exc:
        _check("unknown block" in str(exc), "unknown-block message is explicit")
    finally:
        rtsys.System = orig
    print("ok test_bind_unknown_initial_state_raises")


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
        return  # pytest restores it
    while _SAVED:
        module, attr, original = _SAVED.pop()
        setattr(module, attr, original)


def _run_all():
    funcs = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    for fn in funcs:
        fn()
    print("\nall %d test(s) passed" % len(funcs))


if __name__ == "__main__":
    _run_all()
