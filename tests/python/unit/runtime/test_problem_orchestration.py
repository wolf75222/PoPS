"""ADC-492 (Spec 5 sec.5.16 / sec.11): pops.Problem assembly + pops.compile/bind + PhysicsModel.

These are PURE-PYTHON tests of the inert assembly, the alias, and the thin dispatch wiring.
The real ``.so`` compile (``compile_problem``) and the runtime install/run are Kokkos-gated
and validated on CI / ROMEO, so the dispatch tests MONKEYPATCH ``compile_problem`` /
``System`` / ``AmrSystem`` to assert routing WITHOUT a real compile. Every deferred route is
asserted to raise loudly (never fake success).

Runs both under pytest and as a plain script (``python3 test_problem_orchestration.py``); the
CI runner executes it as a script (the ``__main__`` guard below).
"""
from pathlib import Path
import sys
import tempfile

from pops.params import ConstParam

try:
    import pops
    from pops.mesh.cartesian import CartesianMesh
    from pops.mesh.layouts import AMR, Uniform
    from pops.fields import FieldProblem
    from pops.math import laplacian
    from pops.codegen import orchestration
    from pops.codegen.loader import CompiledModel
    from pops.codegen._compiled_model_identity import model_compile_identity
    from pops.codegen._plans import InstallBlock, InstallPlan
    from pops.model import Handle, Module, OwnerPath
    from pops.model.bind_schema import BindSchema
    from pops.solvers.elliptic import GeometricMG
    from tests.python.support.assertions import _check
except Exception as exc:  # noqa: BLE001
    print("skip test_problem_orchestration (pops unavailable: %s)" % exc)
    sys.exit(0)


def _ref(name, kind="state"):
    return Handle(name, kind=kind, owner=OwnerPath.shared("problem-orchestration"))


_STUB_BINARY_DIR = Path(tempfile.mkdtemp(prefix="pops-problem-orchestration-"))


def _stub_binary(name):
    """Return a real immutable payload for identity hashing; no dynamic loading occurs here."""
    path = _STUB_BINARY_DIR / name
    if not path.exists():
        path.write_bytes(("PoPS test artifact: %s\n" % name).encode("utf-8"))
    return str(path)


# --- tiny stand-ins (no compiler / no runtime) -----------------------------
class _StubCompiledModel(CompiledModel):
    """A target='amr_system' CompiledModel stand-in: ``.so_path`` + the adder/target metadata
    AmrSystem.add_equation dispatches on (no real .so)."""

    def __init__(self, source, so_path=None, target="amr_system"):
        so_path = so_path or _stub_binary("stub_amr.so")
        super().__init__(
            so_path, "production", "add_native_block", (), (), (), 0, None, 0,
            {}, {"cpu": True, "amr": target == "amr_system"}, "abi", "model-hash",
            "c++", "c++20", target=target,
            definition_identity=model_compile_identity(source))
        self.name = source.name

    @property
    def sealed(self):
        return getattr(self, "_sealed", False)


_COMPILE_CALLS = {}


class _StubModel(Module):
    """A genuine operator-first Module with a compiler spy at the final protocol boundary.

    The global call table is deliberate: ``Problem.freeze`` deeply freezes the Module before
    compilation, so recording on the model itself would be an invalid post-freeze mutation.  The
    compiler still receives a real Module and the returned loader is an exact ``CompiledModel``.
    """

    def __init__(self, name="stub"):
        super().__init__(name)
        self.state_space("U", ("rho",))
        _COMPILE_CALLS[id(self)] = []

    @property
    def compiled(self):
        return list(_COMPILE_CALLS[id(self)])

    def compile(self, *, backend, target, **kw):
        _COMPILE_CALLS[id(self)].append((backend, target))
        return _StubCompiledModel(
            self, so_path=_stub_binary("%s_%s.so" % (self.name, target)), target=target)


class _StubCompiled:
    """A compiled-handle stand-in carrying only the immutable InstallPlan bind authority."""

    def __init__(self, target="system", problem=None, layout=None, block_compiled=None):
        self.so_path = _stub_binary("stub.so")
        self.abi_key = "stub-abi"
        self.cxx = "stub-c++"
        self.std = "c++20"
        self.model = None
        self.install_plan = None
        self._problem_snapshot = None
        if problem is not None:
            from pops.problem._snapshot import AuthoringSnapshot
            from pops.problem._detached import detached_frozen

            schema = BindSchema.from_problem(problem)
            problem.freeze()
            models = block_compiled or {
                name: _StubCompiledModel(_StubModel(name), target=target)
                for name, _spec in problem._blocks.items()
            }
            for model in models.values():
                model._seal()
            self._problem_snapshot = AuthoringSnapshot({
                "kind": "direct-bind-test-artifact",
                "target": target,
                "blocks": tuple(models),
            })
            blocks = tuple(
                InstallBlock(name, models[name], detached_frozen(spec["spatial"]))
                for name, spec in problem._blocks.items()
            )
            field_solvers = {
                name: detached_frozen(field.solver)
                for name, field in problem._field_registry.resolved_items(problem.resolve)
                if field.solver is not None
            }
            self.install_plan = InstallPlan(
                snapshot_hash=self._problem_snapshot.hash,
                target=target,
                layout=detached_frozen(layout),
                blocks=blocks,
                bind_schema=schema,
                field_solvers=field_solvers,
                outputs=tuple(detached_frozen(value) for value in (problem._outputs or ())),
                diagnostics=tuple(
                    detached_frozen(value) for value in (problem._diagnostics or ())),
                has_program=(target == "system"),
            )
            self.bind_schema = schema

    @property
    def authoring_snapshot(self):
        return self._problem_snapshot


def _StubTime():
    """Return the exact final Program type; opaque subclasses are not freeze-trusted."""
    return pops.Program("stub-time")


def _poisson_problem():
    """A minimal valid Poisson FieldProblem named 'phi' (the default-served field)."""
    return FieldProblem(name="phi", unknown="phi",
                        equation=(-laplacian("phi") == "charge_density"),
                        solver=GeometricMG())


# --- assembly + chaining + inspect -----------------------------------------
def test_assembly_chaining_and_inspect():
    model = _StubModel("ne")
    prob = (pops.Problem(name="plasma")
            .block("ne", physics=model, spatial=None)
            .aux("B_z", value=None))
    alpha = prob.param(ConstParam("alpha", 1.0))
    _check(prob is prob.block.__self__, "setters operate on the same problem")
    _check(alpha.param_kind == "const", "param returns a typed handle")
    _check(prob.layout is None, "ADC-526: a layout-free Problem carries no layout (supplied at compile)")
    info = prob.inspect().to_dict()  # ADC-564: Problem.inspect() is a typed report; to_dict() bridges
    _check(info["name"] == "plasma", "name carried")
    _check(set(info["blocks"]) == {"ne"}, "block recorded")
    _check(info["params"]["alpha"]["kind"] == "const", "param recorded")
    _check("B_z" in info["aux"], "aux recorded")
    _check(prob.options()["n_blocks"] == 1, "options report n_blocks")
    print("ok test_assembly_chaining_and_inspect")


def test_block_requires_physics_and_no_duplicate():
    prob = pops.Problem()
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
    prob = pops.Problem()
    try:
        prob.field("not a FieldProblem")
        raise AssertionError("field must reject a non-FieldProblem")
    except TypeError:
        pass
    prob.field(_poisson_problem())
    _check("phi" in prob._fields, "field registered by name")
    print("ok test_field_type_checked")


def test_amr_property():
    from pops.mesh.amr import RegridEvery
    from pops.mesh.layouts import Uniform
    # ADC-526: a layout-free Problem exposes .amr (criteria applied at compile). A CONSTRUCTOR
    # Uniform layout still refuses .amr (no level to refine onto).
    try:
        _ = pops.Problem(layout=Uniform(CartesianMesh())).amr
        raise AssertionError("amr on a Uniform constructor layout must raise")
    except ValueError:
        pass
    # A layout-free Problem records the criteria on the constraint registry and chains.
    prob = pops.Problem()
    returned = prob.amr.refine(regrid=RegridEvery(20))
    _check(returned is prob, "amr.refine chains back to the problem")
    _check(prob._constraints.refinement.get("regrid") is not None,
           "refine recorded the regrid policy on the constraint registry")
    # A constructor AMR layout is not a second authority: the registry records the policy and
    # compile-time resolution materialises it on a detached layout.
    prob2 = pops.Problem(layout=AMR(CartesianMesh()))
    prob2.amr.refine(regrid=RegridEvery(20))
    _check(prob2.layout.regrid is None, "problem.amr never mutates the constructor layout")
    resolved = orchestration._resolve_layout(prob2, None)
    _check(resolved is not prob2.layout and resolved.regrid.steps == 20,
           "layout resolution creates the detached merged AMR layout")
    print("ok test_amr_property")


# --- validate(): structural pass + each deferred case raising ---------------
def test_validate_structural_pass():
    prob = pops.Problem().block("ne", physics=_StubModel()).field(_poisson_problem())
    _check(prob.validate() is True, "a single-block + Poisson-field problem validates")
    _check(bool(prob.available()) is True, "available() is yes for a valid problem")
    print("ok test_validate_structural_pass")


def test_validate_requires_a_block():
    try:
        pops.Problem().validate()
        raise AssertionError("a problem with no block must not validate")
    except ValueError:
        pass
    print("ok test_validate_requires_a_block")


def test_validate_multi_block_uniform_lowers():
    # C3: a multi-block assembly on a Uniform layout now VALIDATES (the >1-block reject is removed);
    # each block lowers as its own instance at bind.
    prob = (pops.Problem().block("ne", physics=_StubModel("ne"))
            .block("ni", physics=_StubModel("ni")))
    _check(prob.validate() is True, "a multi-block Uniform Problem validates (C3)")
    _check(prob.options()["n_blocks"] == 2, "options report two blocks")
    print("ok test_validate_multi_block_uniform_lowers")


def test_validate_named_field_lowers():
    # C1-System: a named non-Poisson field now VALIDATES (the _POISSON_FIELD_NAMES whitelist reject
    # is removed); an undeclared field name is caught downstream at install (_install_solver).
    field = FieldProblem(name="temperature", unknown="T",
                         equation=(-laplacian("T") == "src"), solver=GeometricMG())
    prob = pops.Problem().block("ne", physics=_StubModel()).field(field)
    _check(prob.validate() is True, "a Problem with a named non-Poisson field validates (C1-System)")
    print("ok test_validate_named_field_lowers")


def test_validate_outputs_lower():
    # C4 / ADC-509: a valid OutputPolicy / CheckpointPolicy now VALIDATES (the NotImplementedError
    # deferral is removed); a non-policy object is rejected loud (it is a typo, not a deferral).
    from pops.output import OutputPolicy, CheckpointPolicy, HDF5
    from pops.time.schedule import every
    prob = (pops.Problem().block("ne", physics=_StubModel())
            .output(OutputPolicy(format=HDF5(), cadence=every(20)))
            .output(CheckpointPolicy(cadence=every(100), restartable=True)))
    _check(prob.validate() is True, "a Problem with valid output/checkpoint policies validates (C4)")

    class _NotAPolicy:
        name = "nope"
    bad = pops.Problem().block("ne", physics=_StubModel()).output(_NotAPolicy())
    try:
        bad.validate()
        raise AssertionError("a non-policy output object must raise")
    except ValueError as exc:
        # ADC-553/ADC-527: Problem.validate() aggregates the per-family reports and raises a single
        # ValueError via ProblemValidationReport.raise_if_error(); the message still names the type.
        _check("OutputPolicy" in str(exc), "non-policy reject names the expected type")
    print("ok test_validate_outputs_lower")


def test_validate_cross_family_homonym_is_typed():
    field = _poisson_problem()
    prob = pops.Problem().block("phi", physics=_StubModel()).field(field)
    _check(prob.validate() is True,
           "block and field display-name homonyms validate because their handle kinds differ")
    _check(prob.blocks()["phi"] != prob.fields()["phi"],
           "block and field homonyms retain distinct typed identities")
    print("ok test_validate_cross_family_homonym_is_typed")


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
        # ADC-545: the typed Production() flows here; the real compile_problem lowers it. Record the
        # lowered token so the assertion stays the byte-identical "production".
        from pops.codegen.backends import lower_backend
        captured.update(time=time, model=model, backend=lower_backend(backend), target=target)
        return _StubCompiled(target=target)

    _patch(monkeypatch, "pops.codegen.compile_drivers.compile_problem", _fake_compile_problem)
    try:
        prob = pops.Problem(name="u").block("ne", physics=_StubModel())
        compiled = orchestration.compile(prob, layout=Uniform(CartesianMesh()), time=_StubTime())
        _check(captured["target"] == "system", "Uniform layout routes to target='system'")
        _check(captured["backend"] == "production", "default backend forwarded (typed Production() lowered)")
        _check(compiled.install_plan.target == "system", "target carried by the InstallPlan")
        _check(not hasattr(compiled, "_problem"), "compiled artifact retains no live Problem")
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
        prob = pops.Problem(layout=layout).block("ne", physics=model)
        compiled = orchestration.compile(prob)  # no time= : the AMR route does not need one
        _check(tripwire["hit"] is False,
               "the AMR route does NOT call compile_problem (no whole-system time Program)")
        _check(model.compiled == [("production", "amr_system")],
               "the block was compiled once for target='amr_system'")
        plan = compiled.install_plan
        _check(plan.target == "amr_system", "amr_system target carried by the InstallPlan")
        _check(plan.layout is not layout and plan.layout.base is not layout.base
               and plan.layout.base.n == layout.base.n,
               "a deeply detached AMR layout is carried on the InstallPlan")
        _check(set(plan.block_models) == {"ne"},
               "compile carries one CompiledModel per InstallBlock")
        _check(getattr(compiled, "so_path", None) is not None,
               "the handle carries a .so_path so bind's so_path guard passes")
    finally:
        _restore_tripwire()
    print("ok test_compile_amr_routes_to_amr_system")


def test_compile_multi_block_amr_routes_natively():
    # ADC-503: a 2-block AMR Problem lowers -- compile() compiles EACH block to target='amr_system'
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
        prob = (pops.Problem(layout=AMR(CartesianMesh()))
                .block("ne", physics=m_ne)
                .block("ni", physics=m_ni))
        compiled = orchestration.compile(prob)
        _check(tripwire["hit"] is False, "multi-block AMR does NOT call compile_problem")
        _check(m_ne.compiled == [("production", "amr_system")],
               "block 'ne' compiled once for target='amr_system'")
        _check(m_ni.compiled == [("production", "amr_system")],
               "block 'ni' compiled once for target='amr_system'")
        _check(set(compiled.install_plan.block_models) == {"ne", "ni"},
               "compile carries a CompiledModel per block")
        _check(all(cm.target == "amr_system"
                   for cm in compiled.install_plan.block_models.values()),
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
    prob = (pops.Problem(layout=AMR(CartesianMesh(), max_levels=NATIVE_MAX_LEVELS + 1))
            .block("ne", physics=_StubModel()))
    try:
        orchestration.compile(prob, time=_StubTime())
        raise AssertionError("AMR(max_levels beyond native) must be refused")
    except ValueError as exc:
        _check("max_levels" in str(exc), "max-levels message is explicit")
    print("ok test_compile_amr_max_levels_beyond_native_raises")


def test_fft_field_solver_on_amr_layout_rejected_at_validate():
    # Spec 6 sec.8/9 (ADC-516): a field whose solver cannot serve the chosen mesh LAYOUT is
    # refused at Problem.validate (so pops.compile refuses it before any .so build), with the
    # solver's PRECISE message -- here the spectral FFT on an AMR hierarchy.
    from pops.solvers.elliptic import FFT, GeometricMG

    def _field(solver):
        return FieldProblem(name="phi", unknown="phi",
                            equation=(-laplacian("phi") == "charge_density"), solver=solver)

    # FFT on an AMR layout -> rejected with the precise sec.8 message.
    amr_fft = (pops.Problem(layout=AMR(CartesianMesh())).block("ne", physics=_StubModel())
               .field(_field(FFT())))
    try:
        amr_fft.validate()
        raise AssertionError("FFT on layout=AMR must be refused at validate")
    except ValueError as exc:
        _check("FFT requires Uniform(periodic=True), got AMR. Use GeometricMG()." in str(exc),
               "FFT-on-AMR rejection must carry the precise sec.8 message; got: %s" % exc)

    # GeometricMG on the SAME AMR layout validates (it is AMR-capable).
    (pops.Problem(layout=AMR(CartesianMesh())).block("ne", physics=_StubModel())
     .field(_field(GeometricMG()))).validate()

    # FFT on a Uniform layout still validates (its route 'partial' is not a hard 'no').
    (pops.Problem(layout=Uniform(CartesianMesh())).block("ne", physics=_StubModel())
     .field(_field(FFT()))).validate()
    print("ok test_fft_field_solver_on_amr_layout_rejected_at_validate")


def test_layout_solver_check_scoped_to_amr_no_false_positive():
    # The layout-solver check is SCOPED to an AMR route (Spec 6 no-false-positive): a solver whose
    # available() returns a hard "no" for a NON-layout reason (an unresolved external brick) on a
    # Uniform layout is NOT refused, and a solver whose available() reads a context key Problem does
    # not supply never crashes Problem.validate (the call is guarded).
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
    (pops.Problem(layout=Uniform(CartesianMesh())).block("ne", physics=_StubModel())
     .field(_field(_UnresolvableSolver()))).validate()
    # A solver whose available() reads a missing key does not crash validate -- Uniform (not run)
    # nor AMR (the call is guarded; a raise is "not a known incompatibility").
    (pops.Problem(layout=Uniform(CartesianMesh())).block("ne", physics=_StubModel())
     .field(_field(_ContextHungrySolver()))).validate()
    (pops.Problem(layout=AMR(CartesianMesh())).block("ne", physics=_StubModel())
     .field(_field(_ContextHungrySolver()))).validate()
    print("ok test_layout_solver_check_scoped_to_amr_no_false_positive")


def test_compile_missing_time_raises():
    prob = pops.Problem().block("ne", physics=_StubModel())
    try:
        orchestration.compile(prob, layout=Uniform(CartesianMesh()))  # no time= and no problem.time(...)
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
        sentinel = _StubTime()
        prob = pops.Problem().block("ne", physics=_StubModel()).time(sentinel)
        orchestration.compile(prob, layout=Uniform(CartesianMesh()))  # time taken from problem._time
        _check(captured["time"] is sentinel, "problem.time(...) is honored when time= omitted")
    finally:
        _unpatch(monkeypatch)
    print("ok test_compile_problem_time_setter_honored")


def test_compile_multi_block_uniform_lowers(monkeypatch=None):
    # C3: a multi-block Uniform Problem lowers -- compile resolves EACH block's physics into the
    # immutable InstallPlan consumed by bind. compile_problem is monkeypatched so no real .so is built.
    def _fake_compile_problem(*, time, model, backend, target, **kw):
        return _StubCompiled(target=target)

    _patch(monkeypatch, "pops.codegen.compile_drivers.compile_problem", _fake_compile_problem)
    try:
        prob = (pops.Problem().block("ne", physics=_StubModel("ne"))
                .block("ni", physics=_StubModel("ni")))
        compiled = orchestration.compile(prob, layout=Uniform(CartesianMesh()), time=_StubTime())
        _check(compiled.install_plan.target == "system",
               "multi-block Uniform routes to target='system'")
        _check(set(compiled.install_plan.block_models) == {"ne", "ni"},
               "compile carries a model per InstallBlock")
    finally:
        _unpatch(monkeypatch)
    print("ok test_compile_multi_block_uniform_lowers")


def test_compile_amr_module_without_compile_raises():
    # ADC-655 honest boundary: the install-model protocol itself refuses an opaque value. Testing
    # it directly keeps the invalid object outside Problem's stricter freeze/snapshot boundary.
    from pops.codegen._orchestration_compile import compile_install_model

    try:
        compile_install_model("ne", object(), "production", "amr_system", {})
        raise AssertionError("an AMR block without .compile must raise")
    except NotImplementedError as exc:
        _check("install-model protocol" in str(exc),
               "the message names the missing install-model protocol")
    print("ok test_compile_amr_module_without_compile_raises")


def test_program_param_routing_requires_captured_artifact_metadata():
    from pops.runtime._install_param_routing import route_program_params

    class _ProgramArtifact:
        program = _StubTime()
        program_param_routes = None

    try:
        route_program_params(_ProgramArtifact(), BindSchema(), {})
        raise AssertionError("missing program_param_routes metadata must be refused")
    except ValueError as exc:
        _check("no immutable program_param_routes metadata" in str(exc),
               "missing metadata refusal names the immutable route table")

    _ProgramArtifact.program_param_routes = ()
    _check(route_program_params(_ProgramArtifact(), BindSchema(), {}) == {},
           "a captured empty route table is distinct from missing metadata")
    print("ok test_program_param_routing_requires_captured_artifact_metadata")


def test_install_metadata_rejects_live_or_incomplete_models():
    from pops.runtime._amr_system_install import _AmrSystemInstall
    from pops.runtime._system_unified_install import _SystemUnifiedInstall

    compiled = _StubCompiledModel(_StubModel("stub"), target="system")
    compiled.elliptic_field_names = ["theta"]
    instances = {"ne": {"model": compiled}}
    _check(
        _SystemUnifiedInstall._declared_elliptic_fields(object(), instances) == {"theta"},
        "System reads named fields from detached CompiledModel metadata",
    )
    _check(
        _AmrSystemInstall._declared_elliptic_fields(instances) == {"theta"},
        "AMR reads named fields from detached CompiledModel metadata",
    )
    raw = _StubModel("live-authoring")
    for reader, args in (
        (_SystemUnifiedInstall._declared_elliptic_fields,
         (object(), {"ne": {"model": raw}})),
        (_AmrSystemInstall._declared_elliptic_fields, ({"ne": {"model": raw}},)),
    ):
        try:
            reader(*args)
            raise AssertionError("live authoring model must not reach install metadata")
        except TypeError as exc:
            _check("detached CompiledModel" in str(exc),
                   "install metadata refusal names the detached CompiledModel contract")
    try:
        _SystemUnifiedInstall._resolve_instance_model(object(), raw)
        raise AssertionError("a live Module must not resolve during install")
    except TypeError as exc:
        _check("detached CompiledModel" in str(exc),
               "instance-model refusal names the detached CompiledModel contract")
    print("ok test_install_metadata_rejects_live_or_incomplete_models")


# --- bind(): System vs AmrSystem dispatch via a monkeypatched runtime -------
class _RecordingSim:
    """A System / AmrSystem stand-in that records the _install_compiled(...) seam call."""

    last = {}

    def _install_compiled(self, compiled=None, *, instances=None, params=None, aux=None,
                          solvers=None, cadence=None, outputs=None, diagnostics=None,
                          bind_schema=None):
        _RecordingSim.last = {"compiled": compiled, "instances": instances, "params": params,
                              "aux": aux, "solvers": solvers, "cadence": cadence,
                              "outputs": outputs, "diagnostics": diagnostics,
                              "bind_schema": bind_schema}


def _bind_with_stub_runtime(target, layout=None, blocks=("ne",), initial=None):
    """Run bind() with System/AmrSystem replaced by recording stubs; return the chosen ENGINE class.

    ADC-583: bind() now returns a ``BoundSimulation`` VIEW over the internal engine (not the raw
    engine). The runtime adapters build the engine (the monkeypatched stub) and wrap it, so the
    dispatch is asserted on ``type(sim._engine)`` (the stub the adapter chose) while ``sim`` is the
    bound-simulation view. The AmrSystem stub mirrors the real constructor (it accepts the derived
    ``AmrSystemConfig``) and records the refinement flow (set_refinement / set_phi_refinement) the
    adapter applies before install. For target='amr_system' the compiled handle carries a per-block
    CompiledModel table in the ``InstallPlan`` so the install routes the native path
    (compiled=None)."""
    import pops.runtime.system as rtsys

    class _StubSystem(_RecordingSim):
        # Mirrors the real System constructor: the Uniform adapter derives a SystemConfig from the
        # Problem's mesh and passes it (or None when the handle carries no layout). Recorded so a test
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
        prob = pops.Problem(layout=layout) if layout is not None else pops.Problem()
        for name in blocks:
            prob = prob.block(name, physics=_StubModel(name))
        prob = prob.field(_poisson_problem())
        block_compiled = None
        if target == "amr_system":
            block_compiled = {
                name: _StubCompiledModel(_StubModel(name)) for name in blocks
            }
        effective_layout = (orchestration._resolve_layout(prob, layout)
                            if layout is not None else None)
        compiled = _StubCompiled(target=target, problem=prob, layout=effective_layout,
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
    _check(last["compiled"].so_path == _stub_binary("stub.so"),
           "compiled handle passed to install")
    _check(set(last["instances"]) == {"ne"}, "the block becomes one install instance")
    _check(last["instances"]["ne"]["initial"] == [1.0], "initial state routed by block name")
    _check("phi" in last["solvers"], "the Poisson field solver derived from the problem")
    print("ok test_bind_system_dispatch")


def test_bind_system_config_from_uniform_layout():
    # The Uniform bind derives the System's SystemConfig (n / L / periodic) from the Problem's mesh,
    # mirroring the AMR route -- so a NON-default mesh reaches the engine instead of the System()
    # defaults (n=64, L=1.0, periodic). Locks the fix: pre-fix compile set _layout=None on Uniform
    # and the adapter built a bare System(), so the Problem mesh was silently ignored.
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
        prob = pops.Problem().block("ne", physics=_StubModel()).field(_poisson_problem())
        compiled = _StubCompiled(target="system", problem=prob)
        sim = orchestration.bind(compiled, initial_state={"ne": [1.0]})
        _check(type(sim).__name__ == "BoundSimulation", "bind returns a BoundSimulation view")
        _check(isinstance(sim._engine, _StubSystem), "sim._engine is the internal engine (escape hatch)")
        # A hidden assembly setter raises a clear, engine-vocabulary-free AttributeError.
        try:
            _ = sim.add_equation
            raise AssertionError("add_equation must be hidden on the bound simulation")
        except AttributeError as exc:
            msg = str(exc)
            _check("pops.Problem" in msg and "pops.compile" in msg, "the reject speaks Problem/compile/bind")
            for bad in ("System.", "AmrSystem", "set_poisson", "install_program", "set_refinement"):
                _check(bad not in msg, "the reject does not recommend %r" % bad)
    finally:
        rtsys.System = orig
    print("ok test_bind_returns_bound_simulation_view")


def test_bind_flows_output_policies():
    # C4 / ADC-509: bind() must flow the Problem's stored output / checkpoint policies onto the install
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
        prob = (pops.Problem().block("ne", physics=_StubModel())
                .field(_poisson_problem()).output(out).output(ckpt))
        compiled = _StubCompiled(target="system", problem=prob)
        orchestration.bind(compiled, initial_state={"ne": [1.0]})
        flowed = _RecordingSim.last["outputs"]
        _check([type(value) for value in flowed] == [OutputPolicy, CheckpointPolicy],
               "both typed policies flowed to the install seam in order")
        _check([value.options() for value in flowed] == [out.options(), ckpt.options()],
               "detached policies preserve their exact options")
        _check(flowed[0] is not out and flowed[1] is not ckpt,
               "InstallPlan owns detached output policies, not authoring objects")
    finally:
        rtsys.System = orig
    print("ok test_bind_flows_output_policies")


def test_bind_amr_dispatch():
    from pops.mesh.amr import RegridEvery, PatchLayout, Refine
    layout = AMR(CartesianMesh(n=48, L=2.0, periodic=False), max_levels=2, ratio=2,
                 regrid=RegridEvery(4), patches=PatchLayout(distribute_coarse=True,
                                                            coarse_max_grid=16))
    layout.refine = Refine.on(_ref("density")).above(1.5)
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
    # ADC-503: a 2-block AMR Problem binds via the native path -- compiled=None and TWO instances, each
    # carrying its OWN target='amr_system' CompiledModel (so each routes add_native_block).
    from pops.mesh.amr import RegridEvery
    layout = AMR(CartesianMesh(n=32), regrid=RegridEvery(2))
    _, last, _, stub_amr, sim = _bind_with_stub_runtime(
        "amr_system", layout=layout, blocks=("ne", "ni"),
        initial={"ne": [1.0], "ni": [2.0]})
    _check(type(sim) is stub_amr, "a multi-block AMR Problem binds an AmrSystem")
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
    layout.refine = TagUnion(Refine.on(_ref("Density", kind="role")).above(2.0),
                             Refine.on(_ref("phi", kind="field")).gradient_above(0.5))
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
    layout.refine = Refine.on(_ref("ni")).above(3.0)
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
    layout.refine = Refine.on(_ref("ne_velocity")).above(3.0)
    try:
        _bind_with_stub_runtime("amr_system", layout=layout, blocks=("ne",))
        raise AssertionError("a non-density single-block AMR refine subject must raise")
    except NotImplementedError as exc:
        _check("multi-block" in str(exc), "the reject names the multi-block requirement")
    print("ok test_bind_single_block_amr_variable_refinement_rejected")


def test_bind_refuses_unavailable_expression_indicator_without_flattening_it():
    from pops.ir import ValueExpr
    from pops.ir.ops import dx, dy, sqrt
    from pops.mesh.amr import Refine

    rho = _ref("rho")
    indicator = sqrt(dx(ValueExpr(rho)) ** 2 + dy(ValueExpr(rho)) ** 2)
    layout = AMR(CartesianMesh(n=32), refine=Refine.on(indicator).above(0.2))
    try:
        _bind_with_stub_runtime("amr_system", layout=layout)
        raise AssertionError("an unavailable expression-indicator backend must raise")
    except NotImplementedError as exc:
        message = str(exc)
        _check("amr:expression_indicator unavailable" in message,
               "the refusal names the missing expression-indicator capability")
        _check("never flattened" in message,
               "the refusal guarantees that the expression was not converted to a name")
    print("ok test_bind_refuses_unavailable_expression_indicator_without_flattening_it")


def test_runtime_rejects_an_unauthenticated_canonical_looking_refine_handle():
    from pops.mesh.amr import Refine
    from pops.runtime import _bind_adapters

    class _MustNotReceiveSelector:
        def set_refinement(self, *args, **kwargs):
            raise AssertionError("unauthenticated selector reached the native runtime")

        def set_phi_refinement(self, *args, **kwargs):
            raise AssertionError("unauthenticated selector reached the native runtime")

    raw = Refine.on(_ref("rho")).above(0.1)
    try:
        _bind_adapters._apply_refine_criterion(_MustNotReceiveSelector(), raw)
        raise AssertionError("an unauthenticated Refine must be refused")
    except ValueError as exc:
        _check("not authenticated by Problem.resolve" in str(exc),
               "the runtime refusal names the missing Problem authentication boundary")
    print("ok test_runtime_rejects_an_unauthenticated_canonical_looking_refine_handle")


def test_bind_rejects_non_compiled():
    try:
        orchestration.bind(object())
        raise AssertionError("bind must reject a handle without .so_path")
    except TypeError:
        pass
    print("ok test_bind_rejects_non_compiled")


def test_bind_unknown_initial_state_raises():
    prob = pops.Problem().block("ne", physics=_StubModel())
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


# --- ADC-592: compile-time snapshot authority + Problem-mutation drift check ---
def test_compile_freezes_snapshot_authority(monkeypatch=None):
    # ADC-655: compile() retains one immutable InstallPlan, not private authoring mirrors, so bind()
    # lowers from compile-time truth without a live Problem re-read.
    def _fake_compile_problem(*, time, model, backend, target, **kw):
        return _StubCompiled(target=target)

    _patch(monkeypatch, "pops.codegen.compile_drivers.compile_problem", _fake_compile_problem)
    try:
        prob = (pops.Problem().block("ne", physics=_StubModel("ne"))
                .block("ni", physics=_StubModel("ni")).field(_poisson_problem()))
        compiled = orchestration.compile(
            prob, layout=Uniform(CartesianMesh()), time=_StubTime())
        plan = compiled.install_plan
        _check({block.name for block in plan.blocks} == {"ne", "ni"},
               "compile freezes one typed InstallBlock per declared block")
        _check(all(block.model is not None for block in plan.blocks),
               "each InstallBlock carries its compiled model")
        _check("phi" in plan.field_solvers,
               "compile freezes the field solvers (not a live re-read)")
        _check(plan.outputs == (), "compile freezes output policies as a tuple")
        for forbidden in ("_problem", "_block_specs", "_field_solvers", "_outputs"):
            _check(not hasattr(compiled, forbidden),
                   "compiled artifact does not retain private authoring mirror %s" % forbidden)
    finally:
        _unpatch(monkeypatch)
    print("ok test_compile_freezes_snapshot_authority")


def test_bind_rejects_case_mutated_after_compile(monkeypatch=None):
    # ADC-563 (the ADC-592 vulnerability closed even MORE strongly): pops.compile FREEZES the Problem,
    # so mutating it after compile is refused AT THE MUTATION -- a RuntimeError naming the frozen
    # Problem -- rather than only detected later at bind. A post-compile mutation cannot change a bound
    # artifact because it cannot happen at all.
    def _fake_compile_problem(*, time, model, backend, target, **kw):
        return _StubCompiled(target=target)

    _patch(monkeypatch, "pops.codegen.compile_drivers.compile_problem", _fake_compile_problem)
    try:
        prob = pops.Problem().block("ne", physics=_StubModel("ne")).field(_poisson_problem())
        orchestration.compile(prob, layout=Uniform(CartesianMesh()), time=_StubTime())
        _check(prob.frozen, "pops.compile froze the Problem")
        # MUTATE the Problem after compile: adding a block the snapshot never saw is refused here.
        try:
            prob.block("ni_late", physics=_StubModel("ni_late"))
            raise AssertionError("a Problem mutated after compile must be refused (frozen)")
        except RuntimeError as exc:
            msg = str(exc)
            _check("frozen" in msg, "the freeze error says the Problem is frozen")
            _check("pops.compile" in msg, "the freeze error points at pops.compile")
            _check("recompile" in msg, "the freeze error points at a recompile")
    finally:
        _unpatch(monkeypatch)
    print("ok test_bind_rejects_case_mutated_after_compile")


def test_bind_uses_snapshot_not_live_physics(monkeypatch=None):
    # ADC-655: Uniform install instances come from the COMPILE-TIME InstallPlan, so a
    # mutation of a block's physics AFTER compile (without changing the block set) does not leak into
    # the bound install -- the frozen model is what gets bound.
    def _fake_compile_problem(*, time, model, backend, target, **kw):
        return _StubCompiled(target=target)

    _patch(monkeypatch, "pops.codegen.compile_drivers.compile_problem", _fake_compile_problem)
    try:
        import pops.runtime.system as rtsys

        class _StubSystem(_RecordingSim):
            # ADC-583/#427: the Uniform adapter derives a SystemConfig from the Problem mesh and
            # passes it to the engine constructor; mirror the real System signature.
            def __init__(self, config=None):
                self.config = config

        orig = rtsys.System
        rtsys.System = _StubSystem
        try:
            m0 = _StubModel("ne")
            prob = pops.Problem().block("ne", physics=m0).field(_poisson_problem())
            stale_spec = prob._blocks.spec("ne")
            compiled = orchestration.compile(prob, layout=Uniform(CartesianMesh()), time=_StubTime())
            frozen_model = compiled.install_plan.block_models["ne"]
            # A reference obtained before compile is deliberately detached by freeze. Mutating that
            # stale authoring dictionary must not alter the frozen registry or compiled snapshot.
            swapped_model = _StubModel("ne_swapped")
            stale_spec["model"] = swapped_model
            _check(prob._blocks.spec("ne")["model"] is not swapped_model,
                   "freeze detaches stale registry dictionaries")
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
