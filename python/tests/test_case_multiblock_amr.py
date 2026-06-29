"""Spec 5: multi-block AMR programs use compile_problem(layout=AMR) + AmrSystem.install."""
import os
import sys
import tempfile

try:
    import pops
    from pops.mesh.cartesian import CartesianMesh
    from pops.mesh.layouts import AMR
    from pops.model import Module
    from pops.runtime.amr_layout import amr_config_from_layout, flow_amr_layout
    from pops.mesh.amr import PatchLayout, Refine, RegridEvery, TagUnion
except Exception as exc:  # noqa: BLE001
    msg = "skip test_case_multiblock_amr (pops unavailable: %s)" % exc
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
    module = Module("amr_multiblock")
    module.state_space("U", ("rho",), roles={"rho": "Density"})
    return module


def _program(module, blocks):
    program = pops.time.Program("amr_multiblock_step")
    state = module.state_spaces()["U"]
    for block in blocks:
        u = program.state("U", block=block, space=state).n
        program.commit(block, program.linear_combine("%s_next" % block, u))
    return program


def _layout():
    layout = AMR(
        CartesianMesh(n=48, L=2.0, periodic=False),
        max_levels=2,
        ratio=2,
        regrid=RegridEvery(4),
        patches=PatchLayout(distribute_coarse=True, coarse_max_grid=16),
    )
    layout.refine = TagUnion(Refine.on("Density").above(1.5),
                             Refine.on("phi").gradient_above(0.25))
    return layout


def _compile_amr(monkeypatch, blocks=("ne", "ni")):
    calls = {"targets": []}

    def _fake_emit(self, model=None, target="system"):
        calls["targets"].append((target, model))
        return "extern \"C\" int pops_test_amr_multiblock() { return 0; }\n"

    def _fake_run_compile(cmd, label):
        out_path = cmd[cmd.index("-o") + 1]
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "wb") as handle:
            handle.write(b"")

    module = _module()
    program = _program(module, blocks)
    layout = _layout()
    so_path = os.path.join(tempfile.mkdtemp(), "amr_multiblock.so")
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
    return compiled, module, program, layout, calls


class _RecordingAmrSystem(pops.AmrSystem):
    """Local fake AMR runtime: public install entry point, recorded native side effects."""

    def __init__(self, config=None):
        self.config = config
        self.calls = []
        self._aux_field_index = {}
        self._program_cadence_cfl = None
        self._output_policies = []
        self.refinement = []
        self.phi_refinement = []

    def block_names(self):
        return []

    def set_refinement(self, threshold, variable="", role=""):
        self.refinement.append((threshold, variable, role))

    def set_phi_refinement(self, threshold):
        self.phi_refinement.append(threshold)

    def _install_solver(self, field, solver, declared_fields=frozenset()):
        self.calls.append(("solver", field, getattr(solver, "scheme", None)))

    def _add_equation(self, name, model, spatial=None, time=None):
        self.calls.append(("add", name, getattr(model, "name", None), spatial, time))

    def _install_aux(self, field_name, field):
        self.calls.append(("aux", field_name, field))

    def set_density(self, name, initial):
        self.calls.append(("initial", name, initial))

    def _finish_problem_install(self, compiled, so_path, params, cadence):
        self.calls.append(("program", so_path, dict(params), cadence, compiled))


class _Solver:
    scheme = "geometric_mg"


def test_multiblock_amr_compile_problem_targets_amr(monkeypatch=None):
    compiled, module, program, layout, calls = _compile_amr(monkeypatch)
    _check(calls["targets"] == [("amr_system", module)],
           "layout=AMR compiles the AMR Program ABI")
    _check(compiled.program is program, "compiled handle carries the multi-block Program")
    _check(compiled.model is module, "compiled handle carries the Module")
    _check(set(compiled.arguments().instances) == {"ne", "ni"},
           "CompiledProblem arguments list both AMR blocks")
    cfg = amr_config_from_layout(layout)
    _check(cfg.n == 48 and cfg.L == 2.0 and cfg.periodic is False,
           "AMR runtime config derives from the typed layout")
    _check(cfg.regrid_every == 4 and cfg.distribute_coarse is True,
           "AMR regrid and patch settings derive from the typed layout")
    print("ok test_multiblock_amr_compile_problem_targets_amr")


def test_multiblock_amr_install_adds_each_instance(monkeypatch=None):
    compiled, module, _, layout, _ = _compile_amr(monkeypatch)
    sim = _RecordingAmrSystem(config=amr_config_from_layout(layout))
    flow_amr_layout(sim, layout)
    sim.install(
        compiled,
        instances={"ne": {"initial": [1.0]}, "ni": {"initial": [2.0]}},
        solvers={"phi": _Solver()},
    )
    added = [call for call in sim.calls if call[0] == "add"]
    initials = [call for call in sim.calls if call[0] == "initial"]
    added_by_name = {call[1]: call[2:] for call in added}
    initial_by_name = {call[1]: call[2] for call in initials}
    program_calls = [call for call in sim.calls if call[0] == "program"]
    _check(sim.refinement == [(1.5, "", "")], "density refinement flowed before install")
    _check(sim.phi_refinement == [0.25], "phi gradient refinement flowed before install")
    _check(added_by_name == {"ne": (module.name, None, None),
                             "ni": (module.name, None, None)},
           "AmrSystem.install adds one runtime instance per Program block")
    _check(initial_by_name == {"ne": [1.0], "ni": [2.0]},
           "AMR initial state is routed by block name")
    _check(len(program_calls) == 1 and program_calls[0][1] == compiled.so_path,
           "AMR install installs the compiled Program .so")
    print("ok test_multiblock_amr_install_adds_each_instance")


def test_amr_compile_problem_requires_time_program():
    module = _module()
    try:
        pops.compile_problem(model=module, layout=_layout(), include=INCLUDE)
        raise AssertionError("AMR compile without a Program must raise")
    except ValueError as exc:
        _check("time must be" in str(exc), "missing Program is rejected clearly")
    print("ok test_amr_compile_problem_requires_time_program")


def test_amr_install_requires_all_committed_blocks(monkeypatch=None):
    compiled, _, _, layout, _ = _compile_amr(monkeypatch)
    sim = _RecordingAmrSystem(config=amr_config_from_layout(layout))
    try:
        sim.install(compiled, instances={"ne": {"initial": [1.0]}})
        raise AssertionError("AMR install must reject a missing committed block")
    except ValueError as exc:
        _check("instance 'ni'" in str(exc), "missing AMR block is named in the install error")
    print("ok test_amr_install_requires_all_committed_blocks")


def _run_all():
    funcs = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    for fn in funcs:
        fn()
    print("\nall %d test(s) passed" % len(funcs))


if __name__ == "__main__":
    _run_all()
