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
class _StubCompiledModel:
    """A target='amr_system' CompiledModel stand-in: ``.so_path`` + the adder/target metadata
    AmrSystem.add_equation dispatches on (no real .so)."""

    def __init__(self, name="stub", so_path="/tmp/stub_amr.so"):
        self.name = name
        self.so_path = so_path
        self.target = "amr_system"
        self.adder = "add_native_block"


class _StubDsl:
    """The ``.dsl`` engine model a physics block resolves to: its ``.compile(backend, target)``
    returns a stub CompiledModel and records the call (no compiler)."""

    def __init__(self, name="stub"):
        self.name = name
        self.compiled = []  # (backend, target) per compile call

    def compile(self, *, backend, target, **kw):
        self.compiled.append((backend, target))
        return _StubCompiledModel(name=self.name, so_path="/tmp/%s_amr.so" % self.name)


class _StubModel:
    """A physics stand-in exposing the ``.dsl`` engine model pops.compile resolves. The Uniform
    route reads ``.dsl`` as an opaque token for compile_problem; the AMR route calls
    ``.dsl.compile(backend, target='amr_system')``."""

    def __init__(self, name="stub"):
        self.name = name
        self.dsl = _StubDsl(name)  # what _resolve_problem_model returns


class _StubSolver:
    name = "GeometricMG"
    scheme = "geometric_mg"
    options = {}


class _StubCompiled:
    """A compiled-handle stand-in: only ``.so_path`` + the carried problem/target/layout.

    The AMR route also carries ``_block_compiled_models`` (the {block: CompiledModel} table); a
    System handle leaves it unset (the install reads it only for target='amr_system')."""

    def __init__(self, target="system", problem=None, layout=None, block_compiled=None):
        self.so_path = "/tmp/stub.so"
        self.model = None
        self._target = target
        self._problem = problem
        self._layout = layout
        if block_compiled is not None:
            self._block_compiled_models = block_compiled


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


def test_validate_multi_block_uniform_lowers():
    # C3: a multi-block assembly on a Uniform layout now VALIDATES (the >1-block reject is removed);
    # each block lowers as its own instance at bind.
    prob = (pops.Case().block("ne", physics=_StubModel())
            .block("ni", physics=_StubModel()))
    _check(prob.validate() is True, "a multi-block Uniform Case validates (C3)")
    _check(prob.options()["n_blocks"] == 2, "options report two blocks")
    print("ok test_validate_multi_block_uniform_lowers")


def test_validate_named_field_lowers():
    # C1-System: a named non-Poisson field now VALIDATES (the _POISSON_FIELD_NAMES whitelist reject
    # is removed); an undeclared field name is caught downstream at install (_install_solver).
    field = FieldProblem(name="temperature", unknown="T",
                         equation=(-laplacian("T") == "src"), solver=_StubSolver())
    prob = pops.Case().block("ne", physics=_StubModel()).field(field)
    _check(prob.validate() is True, "a Case with a named non-Poisson field validates (C1-System)")
    print("ok test_validate_named_field_lowers")


def test_validate_outputs_lower():
    # C4 / ADC-509: a valid OutputPolicy / CheckpointPolicy now VALIDATES (the NotImplementedError
    # deferral is removed); a non-policy object is rejected loud (it is a typo, not a deferral).
    from pops.output import OutputPolicy, CheckpointPolicy, HDF5
    from pops.time.schedule import every
    prob = (pops.Case().block("ne", physics=_StubModel())
            .output(OutputPolicy(format=HDF5(), cadence=every(20)))
            .output(CheckpointPolicy(cadence=every(100), restartable=True)))
    _check(prob.validate() is True, "a Case with valid output/checkpoint policies validates (C4)")

    class _NotAPolicy:
        name = "nope"
    bad = pops.Case().block("ne", physics=_StubModel()).output(_NotAPolicy())
    try:
        bad.validate()
        raise AssertionError("a non-policy output object must raise")
    except TypeError as exc:
        _check("OutputPolicy" in str(exc), "non-policy reject names the expected type")
    print("ok test_validate_outputs_lower")


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


def test_compile_amr_routes_to_amr_system():
    # ADC-503: single-block AMR lowers via the NATIVE per-block CompiledModel path -- compile() calls
    # the block's .compile(backend='production', target='amr_system') and does NOT touch
    # compile_problem (the whole-system time Program). No time= is required on the AMR route.
    tripwire = {"hit": False}

    def _tripwire(*a, **kw):
        tripwire["hit"] = True
        return _StubCompiled(target="amr_system")

    saved = []

    def _patch_tripwire():
        import pops.codegen.compile_drivers as cd
        saved.append((cd, "compile_problem", cd.compile_problem))
        cd.compile_problem = _tripwire

    def _restore_tripwire():
        while saved:
            mod, attr, orig = saved.pop()
            setattr(mod, attr, orig)

    _patch_tripwire()
    try:
        layout = AMR(CartesianMesh())
        model = _StubModel("ne")
        prob = pops.Case(layout=layout).block("ne", physics=model)
        compiled = orchestration.compile(prob)  # no time= : the AMR route does not need one
        _check(tripwire["hit"] is False,
               "the AMR route does NOT call compile_problem (no whole-system time Program)")
        _check(model.dsl.compiled == [("production", "amr_system")],
               "the block was compiled once for target='amr_system'")
        _check(compiled._target == "amr_system", "amr_system target carried on the handle")
        _check(compiled._layout is layout, "AMR layout carried on the handle for bind()")
        _check(set(compiled._block_compiled_models) == {"ne"},
               "compile carries one CompiledModel per block (_block_compiled_models)")
        _check(getattr(compiled, "so_path", None) is not None,
               "the handle carries a .so_path so bind's so_path guard passes")
    finally:
        _restore_tripwire()
    print("ok test_compile_amr_routes_to_amr_system")


def test_compile_multi_block_amr_routes_natively():
    # ADC-503: a 2-block AMR Case lowers -- compile() compiles EACH block to target='amr_system'
    # (twice), carries the {block: CompiledModel} table, and NEVER calls compile_problem.
    tripwire = {"hit": False}
    saved = []

    def _tripwire(*a, **kw):
        tripwire["hit"] = True
        return _StubCompiled(target="amr_system")

    import pops.codegen.compile_drivers as cd
    saved.append((cd, "compile_problem", cd.compile_problem))
    cd.compile_problem = _tripwire
    try:
        m_ne, m_ni = _StubModel("ne"), _StubModel("ni")
        prob = (pops.Case(layout=AMR(CartesianMesh()))
                .block("ne", physics=m_ne)
                .block("ni", physics=m_ni))
        compiled = orchestration.compile(prob)
        _check(tripwire["hit"] is False, "multi-block AMR does NOT call compile_problem")
        _check(m_ne.dsl.compiled == [("production", "amr_system")],
               "block 'ne' compiled once for target='amr_system'")
        _check(m_ni.dsl.compiled == [("production", "amr_system")],
               "block 'ni' compiled once for target='amr_system'")
        _check(set(compiled._block_compiled_models) == {"ne", "ni"},
               "compile carries a CompiledModel per block")
        _check(all(cm.target == "amr_system" for cm in compiled._block_compiled_models.values()),
               "every carried CompiledModel targets the AMR system")
    finally:
        while saved:
            mod, attr, orig = saved.pop()
            setattr(mod, attr, orig)
    print("ok test_compile_multi_block_amr_routes_natively")


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


def test_fft_field_solver_on_amr_layout_rejected_at_validate():
    # Spec 6 sec.8/9 (ADC-516): a field whose solver cannot serve the chosen mesh LAYOUT is
    # refused at Case.validate (so pops.compile refuses it before any .so build), with the
    # solver's PRECISE message -- here the spectral FFT on an AMR hierarchy.
    from pops.solvers.elliptic import FFT, GeometricMG

    def _field(solver):
        return FieldProblem(name="phi", unknown="phi",
                            equation=(-laplacian("phi") == "charge_density"), solver=solver)

    # FFT on an AMR layout -> rejected with the precise sec.8 message.
    amr_fft = (pops.Case(layout=AMR(CartesianMesh())).block("ne", physics=_StubModel())
               .field(_field(FFT())))
    try:
        amr_fft.validate()
        raise AssertionError("FFT on layout=AMR must be refused at validate")
    except ValueError as exc:
        _check("FFT requires Uniform(periodic=True), got AMR. Use GeometricMG()." in str(exc),
               "FFT-on-AMR rejection must carry the precise sec.8 message; got: %s" % exc)

    # GeometricMG on the SAME AMR layout validates (it is AMR-capable).
    (pops.Case(layout=AMR(CartesianMesh())).block("ne", physics=_StubModel())
     .field(_field(GeometricMG()))).validate()

    # FFT on a Uniform layout still validates (its route 'partial' is not a hard 'no').
    (pops.Case(layout=Uniform(CartesianMesh())).block("ne", physics=_StubModel())
     .field(_field(FFT()))).validate()
    print("ok test_fft_field_solver_on_amr_layout_rejected_at_validate")


def test_layout_solver_check_scoped_to_amr_no_false_positive():
    # The layout-solver check is SCOPED to an AMR route (Spec 6 no-false-positive): a solver whose
    # available() returns a hard "no" for a NON-layout reason (an unresolved external brick) on a
    # Uniform layout is NOT refused, and a solver whose available() reads a context key Case does
    # not supply never crashes Case.validate (the call is guarded).
    from pops.descriptors import Availability

    class _UnresolvableSolver:
        name = "external_elliptic"

        def available(self, context=None):
            return Availability.no("compiled brick could not be resolved")

    class _ContextHungrySolver:
        name = "picky"

        def available(self, context=None):
            return Availability.yes() if context["backend"] == "x" else Availability.no("nope")

    def _field(solver):
        return FieldProblem(name="phi", unknown="phi",
                            equation=(-laplacian("phi") == "charge_density"), solver=solver)

    # A non-layout "no" on a Uniform route is not refused by the layout check (it is never run).
    (pops.Case(layout=Uniform(CartesianMesh())).block("ne", physics=_StubModel())
     .field(_field(_UnresolvableSolver()))).validate()
    # A solver whose available() reads a missing key does not crash validate -- Uniform (not run)
    # nor AMR (the call is guarded; a raise is "not a known incompatibility").
    (pops.Case(layout=Uniform(CartesianMesh())).block("ne", physics=_StubModel())
     .field(_field(_ContextHungrySolver()))).validate()
    (pops.Case(layout=AMR(CartesianMesh())).block("ne", physics=_StubModel())
     .field(_field(_ContextHungrySolver()))).validate()
    print("ok test_layout_solver_check_scoped_to_amr_no_false_positive")


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


def test_compile_multi_block_uniform_lowers(monkeypatch=None):
    # C3: a multi-block Uniform Case lowers -- compile resolves EACH block's physics and carries the
    # {block: model} table on the handle (_block_models); the per-block models flow to install via
    # bind()'s _assemble_instances. compile_problem is monkeypatched so no real .so is built.
    def _fake_compile_problem(*, time, model, backend, target, **kw):
        return _StubCompiled(target=target)

    _patch(monkeypatch, "pops.codegen.compile_drivers.compile_problem", _fake_compile_problem)
    try:
        prob = (pops.Case().block("ne", physics=_StubModel())
                .block("ni", physics=_StubModel()))
        compiled = orchestration.compile(prob, time=object())
        _check(compiled._target == "system", "multi-block Uniform routes to target='system'")
        _check(set(compiled._block_models) == {"ne", "ni"},
               "compile carries a model per block (_block_models)")
    finally:
        _unpatch(monkeypatch)
    print("ok test_compile_multi_block_uniform_lowers")


def test_compile_amr_module_without_compile_raises():
    # ADC-503 honest boundary: an AMR block whose resolved model has no .compile(...) producing a
    # target='amr_system' CompiledModel (e.g. a raw pops.model.Module) raises a clear error, not a
    # cryptic AttributeError. _resolve_problem_model returns the physics as-is when it has no .dsl.
    class _NoCompilePhysics:
        name = "raw"  # no .dsl, no .compile

    prob = pops.Case(layout=AMR(CartesianMesh())).block("ne", physics=_NoCompilePhysics())
    try:
        orchestration.compile(prob)
        raise AssertionError("an AMR block without .compile must raise")
    except NotImplementedError as exc:
        _check("CompiledModel" in str(exc), "the message names the missing AMR loader")
    print("ok test_compile_amr_module_without_compile_raises")


# --- bind(): System vs AmrSystem dispatch via a monkeypatched runtime -------
class _RecordingSim:
    """A System / AmrSystem stand-in that records the _install_compiled(...) seam call."""

    last = {}

    def _install_compiled(self, compiled=None, *, instances=None, params=None, aux=None,
                          solvers=None, cadence=None, outputs=None):
        _RecordingSim.last = {"compiled": compiled, "instances": instances, "params": params,
                              "aux": aux, "solvers": solvers, "cadence": cadence,
                              "outputs": outputs}


def _bind_with_stub_runtime(target, layout=None, blocks=("ne",), initial=None):
    """Run bind() with System/AmrSystem replaced by recording stubs; return the chosen ENGINE class.

    ADC-583: bind() now returns a ``BoundSimulation`` VIEW over the internal engine (not the raw
    engine). The runtime adapters build the engine (the monkeypatched stub) and wrap it, so the
    dispatch is asserted on ``type(sim._engine)`` (the stub the adapter chose) while ``sim`` is the
    bound-simulation view. The AmrSystem stub mirrors the real constructor (it accepts the derived
    ``AmrSystemConfig``) and records the refinement flow (set_refinement / set_phi_refinement) the
    adapter applies before install. For target='amr_system' the compiled handle carries a per-block
    CompiledModel table (``_block_compiled_models``) so the install routes the native path
    (compiled=None)."""
    import pops.runtime.system as rtsys

    class _StubSystem(_RecordingSim):
        # Mirrors the real System constructor: the Uniform adapter derives a SystemConfig from the
        # Case's mesh and passes it (or None when the handle carries no layout). Recorded so a test
        # can assert n / L / periodic reached the engine instead of the System() defaults.
        def __init__(self, config=None):
            self.config = config

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
        for name in blocks:
            prob = prob.block(name, physics=_StubModel(name))
        prob = prob.field(_poisson_problem())
        block_compiled = None
        if target == "amr_system":
            block_compiled = {name: _StubCompiledModel(name) for name in blocks}
        compiled = _StubCompiled(target=target, problem=prob, layout=layout,
                                 block_compiled=block_compiled)
        if initial is None:
            initial = {name: [1.0] for name in blocks}
        sim = orchestration.bind(compiled, initial_state=initial)
        # bind() returns a BoundSimulation view (ADC-583); the dispatch is asserted on the wrapped
        # engine (sim._engine), which is the stub the adapter built.
        engine = sim._engine
        return type(engine), _RecordingSim.last, _StubSystem, _StubAmrSystem, engine
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


def test_bind_system_config_from_uniform_layout():
    # The Uniform bind derives the System's SystemConfig (n / L / periodic) from the Case's mesh,
    # mirroring the AMR route -- so a NON-default mesh reaches the engine instead of the System()
    # defaults (n=64, L=1.0, periodic). Locks the fix: pre-fix compile set _layout=None on Uniform
    # and the adapter built a bare System(), so the Case mesh was silently ignored.
    layout = Uniform(CartesianMesh(n=16, L=2.0, periodic=False))
    sim_class, _, stub_system, _, engine = _bind_with_stub_runtime("system", layout=layout)
    _check(sim_class is stub_system, "target='system' binds a System")
    cfg = engine.config
    _check(cfg is not None, "the Uniform adapter builds the System from a derived SystemConfig")
    _check(cfg.n == 16 and cfg.L == 2.0 and cfg.periodic is False,
           "SystemConfig n/L/periodic derived from the Uniform CartesianMesh (not the defaults)")
    print("ok test_bind_system_config_from_uniform_layout")


def test_bind_returns_bound_simulation_view():
    # ADC-583: bind() returns a BoundSimulation VIEW over the internal engine, NOT the raw engine.
    # The assembly setters are hidden; the wrapped engine is the internal escape hatch (sim._engine).
    import pops.runtime.system as rtsys

    class _StubSystem(_RecordingSim):
        pass

    orig = rtsys.System
    rtsys.System = _StubSystem
    try:
        prob = pops.Case().block("ne", physics=_StubModel()).field(_poisson_problem())
        compiled = _StubCompiled(target="system", problem=prob)
        sim = orchestration.bind(compiled, initial_state={"ne": [1.0]})
        _check(type(sim).__name__ == "BoundSimulation", "bind returns a BoundSimulation view")
        _check(isinstance(sim._engine, _StubSystem), "sim._engine is the internal engine (escape hatch)")
        # A hidden assembly setter raises a clear, engine-vocabulary-free AttributeError.
        try:
            sim.add_equation
            raise AssertionError("add_equation must be hidden on the bound simulation")
        except AttributeError as exc:
            msg = str(exc)
            _check("pops.Case" in msg and "pops.compile" in msg, "the reject speaks Case/compile/bind")
            for bad in ("System.", "AmrSystem", "set_poisson", "install_program", "set_refinement"):
                _check(bad not in msg, "the reject does not recommend %r" % bad)
    finally:
        rtsys.System = orig
    print("ok test_bind_returns_bound_simulation_view")


def test_bind_flows_output_policies():
    # C4 / ADC-509: bind() must flow the Case's stored output / checkpoint policies onto the install
    # seam (outputs=) so the bound sim's run() can fire them. Uses the recording stub System.
    from pops.output import OutputPolicy, CheckpointPolicy
    from pops.time.schedule import every
    import pops.runtime.system as rtsys

    class _StubSystem(_RecordingSim):
        pass

    orig = rtsys.System
    rtsys.System = _StubSystem
    try:
        out = OutputPolicy(cadence=every(5))
        ckpt = CheckpointPolicy(cadence=every(10))
        prob = (pops.Case().block("ne", physics=_StubModel())
                .field(_poisson_problem()).output(out).output(ckpt))
        compiled = _StubCompiled(target="system", problem=prob)
        orchestration.bind(compiled, initial_state={"ne": [1.0]})
        flowed = _RecordingSim.last["outputs"]
        _check(flowed == [out, ckpt], "both policies flowed to the install seam in order")
    finally:
        rtsys.System = orig
    print("ok test_bind_flows_output_policies")


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
    # ADC-503: AMR installs via the NATIVE path (compiled=None) with the per-block CompiledModel.
    _check(last["compiled"] is None,
           "AMR install passes compiled=None (the native add_native_block path, NOT install_program)")
    _check(set(last["instances"]) == {"ne"}, "the AMR block becomes one install instance")
    inst_model = last["instances"]["ne"]["model"]
    _check(getattr(inst_model, "target", None) == "amr_system"
           and getattr(inst_model, "adder", None) == "add_native_block",
           "the instance carries its target='amr_system' CompiledModel (add_native_block)")
    _check(last["instances"]["ne"]["initial"] == [1.0], "initial state routed by block name")
    _check("phi" in last["solvers"], "the Poisson field solver derived from the problem")
    print("ok test_bind_amr_dispatch")


def test_bind_multi_block_amr_native_install():
    # ADC-503: a 2-block AMR Case binds via the native path -- compiled=None and TWO instances, each
    # carrying its OWN target='amr_system' CompiledModel (so each routes add_native_block).
    from pops.mesh.amr import RegridEvery
    layout = AMR(CartesianMesh(n=32), regrid=RegridEvery(2))
    _, last, _, stub_amr, sim = _bind_with_stub_runtime(
        "amr_system", layout=layout, blocks=("ne", "ni"),
        initial={"ne": [1.0], "ni": [2.0]})
    _check(type(sim) is stub_amr, "a multi-block AMR Case binds an AmrSystem")
    _check(last["compiled"] is None, "multi-block AMR install passes compiled=None (native path)")
    _check(set(last["instances"]) == {"ne", "ni"}, "both blocks become install instances")
    for name in ("ne", "ni"):
        cm = last["instances"][name]["model"]
        _check(getattr(cm, "target", None) == "amr_system"
               and getattr(cm, "adder", None) == "add_native_block",
               "instance %r carries its own target='amr_system' CompiledModel" % name)
        _check(cm.name == name, "instance %r carries ITS block's CompiledModel" % name)
    _check(last["instances"]["ni"]["initial"] == [2.0], "per-block initial state routed by name")
    print("ok test_bind_multi_block_amr_native_install")


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


def test_bind_multi_block_amr_variable_refinement():
    # ADC-503: in MULTI-BLOCK a non-density refine subject forwards to set_refinement(variable=...)
    # (the union-of-tags engine resolves it per block); in single-block the same subject is refused.
    from pops.mesh.amr import RegridEvery, Refine
    layout = AMR(CartesianMesh(n=32), regrid=RegridEvery(2))
    layout.refine = Refine.on("ni").above(3.0)
    _, _, _, _, sim = _bind_with_stub_runtime("amr_system", layout=layout, blocks=("ne", "ni"),
                                              initial={"ne": [1.0], "ni": [2.0]})
    _check(sim.refinement == (3.0, "ni", ""),
           "a non-density subject forwards to set_refinement(threshold, variable='ni') in multi-block")
    print("ok test_bind_multi_block_amr_variable_refinement")


def test_bind_single_block_amr_variable_refinement_rejected():
    # ADC-503 boundary: a non-density subject in SINGLE-block AMR is still refused (the AmrCouplerMP
    # path refines on component 0 only); the multi-block selector needs >= 2 blocks.
    from pops.mesh.amr import RegridEvery, Refine
    layout = AMR(CartesianMesh(n=32), regrid=RegridEvery(2))
    layout.refine = Refine.on("ne_velocity").above(3.0)
    try:
        _bind_with_stub_runtime("amr_system", layout=layout, blocks=("ne",))
        raise AssertionError("a non-density single-block AMR refine subject must raise")
    except NotImplementedError as exc:
        _check("multi-block" in str(exc), "the reject names the multi-block requirement")
    print("ok test_bind_single_block_amr_variable_refinement_rejected")


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


# --- ADC-592: compile-time snapshot authority + Case-mutation drift check ---
def test_compile_freezes_snapshot_authority(monkeypatch=None):
    # ADC-592: compile() freezes WHAT it saw on the handle -- _block_specs (model + spatial per block),
    # _field_solvers, _outputs -- so bind() lowers from the compile-time truth, not a live Case re-read.
    def _fake_compile_problem(*, time, model, backend, target, **kw):
        return _StubCompiled(target=target)

    _patch(monkeypatch, "pops.codegen.compile_drivers.compile_problem", _fake_compile_problem)
    try:
        prob = (pops.Case().block("ne", physics=_StubModel("ne"))
                .block("ni", physics=_StubModel("ni")).field(_poisson_problem()))
        compiled = orchestration.compile(prob, time=object())
        _check(set(compiled._block_specs) == {"ne", "ni"},
               "compile freezes a per-block spec (model + spatial) on the handle")
        _check(all("model" in s and "spatial" in s for s in compiled._block_specs.values()),
               "each frozen block spec carries model + spatial")
        _check("phi" in compiled._field_solvers,
               "compile freezes the field solvers (not a live re-read)")
        _check(compiled._outputs == [], "compile freezes the (empty) output policies")
    finally:
        _unpatch(monkeypatch)
    print("ok test_compile_freezes_snapshot_authority")


def test_bind_rejects_case_mutated_after_compile(monkeypatch=None):
    # ADC-592 (the proven vulnerability closed): mutating the Case's blocks between compile and bind is
    # a LOUD ValueError -- a compiled artifact is frozen at compile and not affected by a later mutation.
    def _fake_compile_problem(*, time, model, backend, target, **kw):
        return _StubCompiled(target=target)

    _patch(monkeypatch, "pops.codegen.compile_drivers.compile_problem", _fake_compile_problem)
    try:
        import pops.runtime.system as rtsys

        class _StubSystem(_RecordingSim):
            pass

        orig = rtsys.System
        rtsys.System = _StubSystem
        try:
            prob = pops.Case().block("ne", physics=_StubModel("ne")).field(_poisson_problem())
            compiled = orchestration.compile(prob, time=object())
            compiled._problem = prob  # the live Case bind() re-reads
            # MUTATE the Case after compile: add a block the snapshot never saw.
            prob.block("ni_late", physics=_StubModel("ni_late"))
            try:
                orchestration.bind(compiled, initial_state={"ne": [1.0]})
                raise AssertionError("a Case mutated after compile must be refused at bind")
            except ValueError as exc:
                msg = str(exc)
                _check("mutated after pops.compile" in msg, "the drift error names the mutation")
                _check("ni_late" in msg, "the drift error names the added block")
                _check("recompile" in msg, "the drift error points at a recompile")
        finally:
            rtsys.System = orig
    finally:
        _unpatch(monkeypatch)
    print("ok test_bind_rejects_case_mutated_after_compile")


def test_bind_uses_snapshot_not_live_physics(monkeypatch=None):
    # ADC-592: the Uniform install instances come from the COMPILE-TIME snapshot (_block_specs), so a
    # mutation of a block's physics AFTER compile (without changing the block set) does not leak into
    # the bound install -- the frozen model is what gets bound.
    def _fake_compile_problem(*, time, model, backend, target, **kw):
        return _StubCompiled(target=target)

    _patch(monkeypatch, "pops.codegen.compile_drivers.compile_problem", _fake_compile_problem)
    try:
        import pops.runtime.system as rtsys

        class _StubSystem(_RecordingSim):
            pass

        orig = rtsys.System
        rtsys.System = _StubSystem
        try:
            m0 = _StubModel("ne")
            prob = pops.Case().block("ne", physics=m0).field(_poisson_problem())
            compiled = orchestration.compile(prob, time=object())
            frozen_model = compiled._block_specs["ne"]["model"]
            # Swap the block's physics AFTER compile (same block name -> no drift).
            prob._blocks["ne"]["physics"] = _StubModel("ne_swapped")
            sim = orchestration.bind(compiled, initial_state={"ne": [1.0]})
            _check(_RecordingSim.last["instances"]["ne"]["model"] is frozen_model,
                   "bind installs the COMPILE-TIME frozen model, not the live-swapped physics")
            _check(sim is not None, "bind still returns a bound simulation view")
        finally:
            rtsys.System = orig
    finally:
        _unpatch(monkeypatch)
    print("ok test_bind_uses_snapshot_not_live_physics")


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
