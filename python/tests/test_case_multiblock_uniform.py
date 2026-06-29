"""Spec 5: multi-block Uniform programs install through compile_problem + sim.install."""
import os
import sys
import tempfile

try:
    import pops
    from pops.mesh.cartesian import CartesianMesh
    from pops.mesh.layouts import Uniform
    from pops.model import Module
except Exception as exc:  # noqa: BLE001
    msg = "skip test_case_multiblock_uniform (pops unavailable: %s)" % exc
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


def _module():
    module = Module("uniform_multiblock")
    module.state_space("U", ("rho",), roles={"rho": "Density"})
    return module


def _program(module, blocks):
    program = pops.time.Program("uniform_multiblock_step")
    state = module.state_spaces()["U"]
    for block in blocks:
        u = program.state("U", block=block, space=state).n
        program.commit(block, program.linear_combine("%s_next" % block, u))
    return program


def _compile_uniform(monkeypatch, blocks=("ne", "ni")):
    calls = {"targets": []}

    def _fake_emit(self, model=None, target="system"):
        calls["targets"].append((target, model))
        return "extern \"C\" int pops_test_uniform_multiblock() { return 0; }\n"

    def _fake_run_compile(cmd, label):
        out_path = cmd[cmd.index("-o") + 1]
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "wb") as handle:
            handle.write(b"")

    module = _module()
    program = _program(module, blocks)
    so_path = os.path.join(tempfile.mkdtemp(), "uniform_multiblock.so")
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
            layout=Uniform(CartesianMesh(n=32)),
            include=INCLUDE,
            force=True,
        )
    finally:
        _unpatch(monkeypatch)
    return compiled, module, program, calls


class _RecordingSystem(pops.System):
    """Local fake System: public install entry point, recorded native side effects."""

    def __init__(self):
        self.calls = []
        self._aux_field_index = {}
        self._program_cadence_cfl = None

    def block_names(self):
        return []

    def _install_solver(self, field, solver, declared_fields=frozenset()):
        self.calls.append(("solver", field))

    def _resolve_instance_model(self, model):
        self.calls.append(("resolve", getattr(model, "name", None)))
        return model

    def _lower_spatial(self, spatial):
        return spatial

    def _validate_riemann_capability(self, model, spatial):
        return None

    def _add_equation(self, name, model, spatial=None, time=None):
        self.calls.append(("add", name, getattr(model, "name", None)))

    def _set_state(self, name, state):
        self.calls.append(("initial", name, state))

    def _install_aux(self, field_name, field):
        self.calls.append(("aux", field_name))

    def _install_params(self, resolved_models, params, reject_unknown=True):
        return set()

    def _install_problem_so(self, so_path):
        self.calls.append(("program", so_path))

    def _install_problem_params(self, compiled, params):
        self.calls.append(("program_params", dict(params)))

    def _install_cadence(self, cadence):
        self.calls.append(("cadence", cadence))


def test_multiblock_uniform_compile_problem_targets_system(monkeypatch=None):
    compiled, module, program, calls = _compile_uniform(monkeypatch)
    _check(calls["targets"] == [("system", module)],
           "Uniform layout compiles the system Program ABI")
    args = compiled.arguments()
    _check(set(args.instances) == {"ne", "ni"},
           "CompiledProblem arguments list both committed blocks")
    _check(compiled.program is program, "compiled handle carries the multi-block Program")
    print("ok test_multiblock_uniform_compile_problem_targets_system")


def test_multiblock_uniform_install_adds_each_instance(monkeypatch=None):
    compiled, module, _, _ = _compile_uniform(monkeypatch)
    sim = _RecordingSystem()
    sim.install(
        compiled,
        instances={"ne": {"initial": [1.0]}, "ni": {"initial": [2.0]}},
    )
    added = [call for call in sim.calls if call[0] == "add"]
    initials = [call for call in sim.calls if call[0] == "initial"]
    added_by_name = {call[1]: call[2] for call in added}
    initial_by_name = {call[1]: call[2] for call in initials}
    _check(added_by_name == {"ne": module.name, "ni": module.name},
           "sim.install adds one runtime instance per Program block")
    _check(initial_by_name == {"ne": [1.0], "ni": [2.0]},
           "initial state is routed by block name")
    _check([call for call in sim.calls if call == ("program", compiled.so_path)],
           "compiled Program .so is installed")
    print("ok test_multiblock_uniform_install_adds_each_instance")


def test_multiblock_uniform_install_requires_all_committed_blocks(monkeypatch=None):
    compiled, _, _, _ = _compile_uniform(monkeypatch)
    try:
        _RecordingSystem().install(compiled, instances={"ne": {"initial": [1.0]}})
        raise AssertionError("install must reject a missing committed block")
    except ValueError as exc:
        _check("instance 'ni'" in str(exc), "missing block is named in the install error")
    print("ok test_multiblock_uniform_install_requires_all_committed_blocks")


def _run_all():
    funcs = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    for fn in funcs:
        fn()
    print("\nall %d test(s) passed" % len(funcs))


if __name__ == "__main__":
    _run_all()
