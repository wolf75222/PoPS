"""Spec 5 problem route: compile_problem(Module, Program, layout) + sim.install.

These pure-Python host tests exercise the public Spec 5 route directly:
build a real ``pops.model.Module`` and ``pops.time.Program``, call ``pops.compile_problem(...)``
with a typed layout, then install the resulting compiled handle through ``sim.install(...)``.

No real C++ compiler is invoked here. The compiler runner is monkeypatched after source emission so
the returned object is still a real ``CompiledProblem`` carrying the Module and Program metadata.
"""
import os
import sys
import tempfile

try:
    import pops
    from pops.mesh.cartesian import CartesianMesh
    from pops.mesh.layouts import AMR, Uniform
    from pops.model import Module
except Exception as exc:  # noqa: BLE001
    msg = "skip test_problem_orchestration (pops unavailable: %s)" % exc
    if "pytest" in sys.modules:
        import pytest
        pytest.skip(msg, allow_module_level=True)
    print(msg)
    sys.exit(0)


INCLUDE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "include"))
_SAVED = []


def _check(cond, msg):
    if not cond:
        raise AssertionError(msg)


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


def _module(name="spec5_problem"):
    module = Module(name)
    module.state_space("U", ("rho",), roles={"rho": "Density"})
    return module


def _program(module, name="spec5_step", blocks=("plasma",)):
    program = pops.time.Program(name)
    state = module.state_spaces()["U"]
    for block in blocks:
        u = program.state("U", block=block, space=state).n
        program.commit(block, program.linear_combine("%s_next" % block, u))
    return program


def _compile_inert(monkeypatch, *, layout, blocks=("plasma",), name="spec5_step"):
    calls = {"targets": [], "commands": []}

    def _fake_emit(self, model=None, target="system"):
        calls["targets"].append((target, model))
        return "extern \"C\" int pops_test_problem_route() { return 0; }\n"

    def _fake_run_compile(cmd, label):
        calls["commands"].append((list(cmd), label))
        out_path = cmd[cmd.index("-o") + 1]
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "wb") as handle:
            handle.write(b"")

    module = _module("module_%s" % name)
    program = _program(module, name=name, blocks=blocks)
    so_path = os.path.join(tempfile.mkdtemp(), "%s.so" % name)

    _patch(monkeypatch, "pops.time.program.Program._emit_cpp_program_for_target", _fake_emit)
    _patch(monkeypatch, "pops.codegen.compile_drivers._run_compile", _fake_run_compile)
    _patch(monkeypatch, "pops.codegen.compile_drivers.pops_loader_build_flags",
           lambda cxx=None: ("c++", [], []))
    _patch(monkeypatch, "pops.codegen.compile_drivers._probe_cxx_std",
           lambda cc, std: "c++23")
    _patch(monkeypatch, "pops.codegen.compile_drivers.pops_header_signature",
           lambda include: "test-header-sig")
    _patch(monkeypatch, "pops.codegen.compile_drivers._dsl_optflags", lambda: [])
    try:
        compiled = pops.compile_problem(
            so_path,
            model=module,
            time=program,
            layout=layout,
            include=INCLUDE,
            force=True,
        )
    finally:
        _unpatch(monkeypatch)
    return compiled, module, program, calls


class _Solver:
    scheme = "geometric_mg"


class _RecordingSystem(pops.System):
    """Local fake System: public install entry point, recorded native side effects."""

    def __init__(self):
        self.calls = []
        self._aux_field_index = {}
        self._program_cadence_cfl = None
        self._output_policies = []

    def block_names(self):
        return []

    def _install_solver(self, field, solver, declared_fields=frozenset()):
        self.calls.append(("solver", field, getattr(solver, "scheme", None),
                           tuple(sorted(declared_fields))))

    def _resolve_instance_model(self, model):
        self.calls.append(("resolve_model", getattr(model, "name", None)))
        return model

    def _lower_spatial(self, spatial):
        self.calls.append(("lower_spatial", spatial))
        return spatial

    def _validate_riemann_capability(self, model, spatial):
        self.calls.append(("validate_riemann", getattr(model, "name", None), spatial))

    def _add_equation(self, name, model, spatial=None, time=None):
        self.calls.append(("add_equation", name, getattr(model, "name", None), spatial, time))

    def _set_state(self, name, state):
        self.calls.append(("set_state", name, state))

    def _install_aux(self, field_name, field):
        self.calls.append(("aux", field_name, field))

    def _install_params(self, resolved_models, params, reject_unknown=True):
        self.calls.append(("params", tuple(sorted(resolved_models)), dict(params), reject_unknown))
        return set()

    def _install_problem_so(self, so_path):
        self.calls.append(("install_program", so_path))

    def _install_problem_params(self, compiled, params):
        self.calls.append(("program_params", dict(params)))

    def _install_cadence(self, cadence):
        self.calls.append(("cadence", cadence))


def test_compile_problem_uniform_returns_compiled_problem(monkeypatch=None):
    layout = Uniform(CartesianMesh(n=16))
    compiled, module, program, calls = _compile_inert(monkeypatch, layout=layout)
    _check(calls["targets"] == [("system", module)],
           "layout=Uniform selects the system program ABI")
    _check(len(calls["commands"]) == 1, "compile_problem reaches the compiler runner")
    _check(compiled.model is module, "CompiledProblem carries the Module")
    _check(compiled.program is program, "CompiledProblem carries the Program")
    _check(compiled.so_path.endswith(".so"), "CompiledProblem carries the .so path")
    _check(os.path.isfile(compiled.so_path), "the fake compiler produced the artifact path")
    print("ok test_compile_problem_uniform_returns_compiled_problem")


def test_compile_problem_amr_returns_amr_target(monkeypatch=None):
    layout = AMR(CartesianMesh(n=16), max_levels=2, ratio=2)
    compiled, module, _, calls = _compile_inert(monkeypatch, layout=layout, name="amr_step")
    _check(calls["targets"] == [("amr_system", module)],
           "layout=AMR selects the AMR program ABI")
    _check(compiled.model is module, "AMR CompiledProblem still carries the Module")
    print("ok test_compile_problem_amr_returns_amr_target")


def test_system_install_uses_compiled_problem_handle(monkeypatch=None):
    compiled, module, _, _ = _compile_inert(
        monkeypatch,
        layout=Uniform(CartesianMesh(n=16)),
        name="install_uniform",
    )
    sim = _RecordingSystem()
    sim.install(
        compiled,
        instances={"plasma": {"initial": [1.0]}},
        solvers={"phi": _Solver()},
    )

    solver_calls = [call for call in sim.calls if call[0] == "solver"]
    added = [call for call in sim.calls if call[0] == "add_equation"]
    states = [call for call in sim.calls if call[0] == "set_state"]
    programs = [call for call in sim.calls if call[0] == "install_program"]
    _check(solver_calls == [("solver", "phi", "geometric_mg", ())],
           "sim.install routes typed solver descriptors")
    _check(added == [("add_equation", "plasma", module.name, None, None)],
           "install binds the named instance to the Module")
    _check(states == [("set_state", "plasma", [1.0])],
           "install routes initial state through the named instance")
    _check(programs == [("install_program", compiled.so_path)],
           "install installs the compiled Program .so")
    print("ok test_system_install_uses_compiled_problem_handle")


def test_install_missing_required_instance_raises(monkeypatch=None):
    compiled, _, _, _ = _compile_inert(
        monkeypatch,
        layout=Uniform(CartesianMesh(n=16)),
        name="missing_instance",
    )
    try:
        _RecordingSystem().install(compiled, instances={})
        raise AssertionError("missing required Program instance must raise")
    except ValueError as exc:
        _check("instance 'plasma'" in str(exc), "missing instance is named clearly")
    print("ok test_install_missing_required_instance_raises")


def test_compile_problem_rejects_non_module_model():
    module = _module("reject_model")
    program = _program(module, name="reject_model_step")

    try:
        pops.compile_problem(
            model=object(),
            time=program,
            layout=Uniform(CartesianMesh(n=8)),
            include=INCLUDE,
        )
        raise AssertionError("non-Module model must be rejected")
    except TypeError as exc:
        _check("pops.model.Module" in str(exc),
               "non-Module model rejection names the modern Module route")
    print("ok test_compile_problem_rejects_non_module_model")


def test_compile_problem_requires_time_program():
    module = _module("missing_time")
    try:
        pops.compile_problem(model=module, layout=Uniform(CartesianMesh(n=8)), include=INCLUDE)
        raise AssertionError("compile_problem without a Program must raise")
    except ValueError as exc:
        _check("time must be" in str(exc), "missing Program is rejected clearly")
    print("ok test_compile_problem_requires_time_program")


def test_install_reports_missing_required_bind_inputs():
    class _Args:
        instances = {"plasma": {"required": True}, "optional": {"required": False}}
        params = {"nu": {"required": True}}
        aux = {"B_z": {"required": True}}
        solvers = {"phi": {"required": True}}

    class _Compiled:
        so_path = "/fake/problem.so"
        model = _module("bind_requirements")

        def arguments(self):
            return _Args()

    sim = _RecordingSystem()
    try:
        sim.install(_Compiled(), instances={"plasma": {"initial": [1.0]}}, aux={"B_z": []})
        raise AssertionError("missing required bind inputs must raise")
    except ValueError as exc:
        text = str(exc)
        _check("runtime param 'nu'" in text, "missing runtime param is reported")
        _check("solver for field 'phi'" in text, "missing solver is reported")
        _check("optional" not in text, "optional instances are not reported")
    _check(not sim.calls, "missing bind inputs reject before mutating the runtime")
    print("ok test_install_reports_missing_required_bind_inputs")


def _run_all():
    funcs = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    for fn in funcs:
        fn()
    print("\nall %d test(s) passed" % len(funcs))


if __name__ == "__main__":
    _run_all()
