"""ADC-598: architecture gates against duplicate public runtime routes.

The target authoring surface must lower to the existing Program IR / typed layout
manifests.  It must not grow a second front-door around ``System`` / ``AmrSystem``
or resurrect legacy runtime setters as public test targets.

Most checks are source-only so they run without a built native extension.  The two
runtime imports below are lowering proofs and skip cleanly when ``pops`` is not
importable.
"""
import ast
import pathlib
import re

import pytest


REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
POPS = REPO_ROOT / "python" / "pops"
BINDINGS = REPO_ROOT / "python" / "bindings"
ARCH_TESTS = REPO_ROOT / "tests" / "architecture"


TARGET_SURFACE_ROOTS = (
    "case.py",
    "diagnostics",
    "external",
    "fields",
    "ir",
    "lib",
    "linalg",
    "mesh",
    "model",
    "moments",
    "numerics",
    "output",
    "params",
    "physics",
    "solvers",
    "time",
)

TEXT_SELECTOR_SURFACE_ROOTS = (
    "case.py",
    "diagnostics",
    "external",
    "fields",
    "mesh",
    "moments",
    "numerics",
    "output",
    "params",
    "solvers",
    "time",
)

FRONT_DOORS = {"System", "AmrSystem"}
RUNTIME_MODULE_ROOTS = (
    "pops",
    "pops._pops",
    "pops.runtime",
    "pops.runtime.system",
    "pops.runtime.amr_system",
)

LEGACY_RUNTIME_SETTERS = {
    "_install_compiled",
    "add_block",
    "add_compiled_block",
    "add_equation",
    "add_native_block",
    "install_program",
    "set_density",
    "set_phi_refinement",
    "set_poisson",
    "set_program_cadence",
    "set_program_params",
    "set_refinement",
    "set_state",
}

# ADC-545: the two compliance tests that used to construct pops.System / pops.AmrSystem now assert
# those names are GONE from the top-level surface and reach the engines through the advanced
# pops.runtime.system import, so they no longer need an allowlist entry here. The list is empty.
ARCH_FRONT_DOOR_ALLOWLIST = {}

TEXT_SELECTOR_ARGS = {
    "backend",
    "format",
    "geometry",
    "kind",
    "layout",
    "limiter",
    "method",
    "mode",
    "policy",
    "recon",
    "reconstruction",
    "riemann",
    "route",
    "scheme",
    "solver",
    "target",
}

TEXT_SELECTOR_ALLOWLIST = {
    ("python/pops/time/program.py", "Program.emit_cpp_program", "target"):
        "internal codegen target seam; public compile/bind do not expose target=",
    ("python/pops/time/program_local.py", "_ProgramLocal.solve_local_nonlinear", "method"):
        "typed local nonlinear op currently has a single explicit Newton backend",
    ("python/pops/time/schedule.py", "Schedule.__init__", "policy"):
        "typed schedule descriptor limits the existing recompute/hold/accumulate policy set",
    ("python/pops/mesh/boundaries/__init__.py", "Physical.__init__", "kind"):
        "typed boundary descriptor limits the legacy wall/outlet topology selector",
}

PYBIND_LEGACY_NAME_ALLOWLIST = {
    "python/bindings/core/init/init_system.cpp":
        "internal System adapter binding; target surface reaches it through compile/bind",
    "python/bindings/core/init/init_amr.cpp":
        "internal AmrSystem adapter binding; AMR target reaches it through typed layout",
}

PYBIND_LEGACY_NAMES = {
    "System",
    "AmrSystem",
    "add_compiled_block",
    "add_native_block",
    "install_program",
    "set_phi_refinement",
    "set_poisson",
    "set_program_cadence",
    "set_program_params",
    "set_refinement",
    "solve_fields",
}

# Frozen budgets on files that have historically been used as dumping grounds for
# fallback/runtime routes.  They sit slightly above the current line counts; adding
# a duplicate subsystem should require an intentional split or a reviewed budget.
LARGE_RUNTIME_FILE_BUDGETS = {
    # ADC-632: system.cpp is now the CORE facade TU only (ctor/dtor, abi_key, step forwards,
    # mark_bound / lifecycle_state); the bulk moved into the sibling TUs below. Budget lowered
    # from 3200 to a small ceiling so any regrowth requires an intentional split.
    "python/bindings/system/base/system.cpp": 350,
    # ADC-632: the install/composition seam (structural setters + install_program + native_loader
    # instantiation) is the one sibling TU that legitimately stays over 1000 lines; the rest
    # (fields / io / profiling / program) sit well under the default. Budgeted with headroom.
    "python/bindings/system/base/system_install.cpp": 1250,
    # ADC-542: the AMR composite_reduce + rebuild_hierarchy (v3 restart) + level_owner_ranks facade
    # seams (native diagnostics / restartable-under-regridding) grew this by ~100 lines.
    # ADC-514: the native per-block runtime-param carrier residue (Impl members, make_build_params /
    # builder wiring, set_compiled_block signature) that stays inline after hoisting the two facade
    # methods and the loader guard to amr_system_params.hpp / amr_native_param_guard.hpp adds ~25 lines.
    # ADC-612: the effective-options audit (FAC + Berger-Rigoutsos knobs in effective_options_report)
    # adds ~60 lines; the report reads the TU-local private Impl, so it cannot move to a sibling TU.
    # ADC-631: the multistep history-ring facade seams (program_last_dt + uses_runtime_engine +
    # history accessors mirroring the System names so _system_io_history.py is reused verbatim)
    # add ~95 thin engine forwards.
    # ADC-635: the replay-through-regrids facade adds ~19 lines (the (dt, cursor) closure driving
    # the facade macro_step_ per re-step, its save/restore, and the last_replay_regrid_steps
    # accessor the v3 reader asserts); it reads the TU-local private Impl, so it cannot move.
    "python/bindings/amr/amr_system.cpp": 2518,
    "include/pops/runtime/amr/amr_runtime.hpp": 1850,
    "include/pops/runtime/system/system_field_solver.hpp": 1100,
    "python/pops/runtime/_system_unified_install.py": 550,
    # ADC-542: the run-loop diagnostics firing (run / _fire_diagnostics) added to System.run.
    "python/pops/runtime/system.py": 285,
    "python/pops/runtime/amr_system.py": 560,
}


def _rel(path):
    return path.relative_to(REPO_ROOT).as_posix()


def _read(path):
    return path.read_text(encoding="utf-8")


def _parse(path):
    return ast.parse(_read(path), filename=str(path))


def _py_files(root):
    return sorted(p for p in root.rglob("*.py") if "__pycache__" not in p.parts)


def _target_surface_files():
    for entry in TARGET_SURFACE_ROOTS:
        path = POPS / entry
        if path.is_file():
            yield path
        elif path.is_dir():
            yield from _py_files(path)


def _text_selector_surface_files():
    for entry in TEXT_SELECTOR_SURFACE_ROOTS:
        path = POPS / entry
        if path.is_file():
            yield path
        elif path.is_dir():
            yield from _py_files(path)


def _dotted_name(node):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _dotted_name(node.value)
        return ("%s.%s" % (base, node.attr)) if base else node.attr
    return None


def _is_runtime_front_door_name(name):
    if not name:
        return False
    if name in FRONT_DOORS:
        return True
    parts = name.split(".")
    if parts[-1] not in FRONT_DOORS:
        return False
    module = ".".join(parts[:-1])
    return any(module == root or module.startswith(root + ".") for root in RUNTIME_MODULE_ROOTS)


def _function_qualifiers(tree):
    class Visitor(ast.NodeVisitor):
        def __init__(self):
            self.stack = []
            self.functions = []

        def visit_ClassDef(self, node):
            self.stack.append(node.name)
            self.generic_visit(node)
            self.stack.pop()

        def visit_FunctionDef(self, node):
            qual = ".".join(self.stack + [node.name])
            self.functions.append((qual, node))
            self.stack.append(node.name)
            self.generic_visit(node)
            self.stack.pop()

        visit_AsyncFunctionDef = visit_FunctionDef

    visitor = Visitor()
    visitor.visit(tree)
    return visitor.functions


def _string_defaults(function):
    args = list(function.args.args)
    defaults = list(function.args.defaults)
    offset = len(args) - len(defaults)
    for index, default in enumerate(defaults):
        arg = args[offset + index]
        if isinstance(default, ast.Constant) and isinstance(default.value, str):
            yield arg.arg, default.value, default.lineno
    for arg, default in zip(function.args.kwonlyargs, function.args.kw_defaults):
        if isinstance(default, ast.Constant) and isinstance(default.value, str):
            yield arg.arg, default.value, default.lineno


def _line_count(path):
    with path.open("rb") as handle:
        return sum(1 for _ in handle)


def _has_meaningful_reason(reason):
    text = reason.lower()
    return len(reason) >= 24 and any(
        token in text for token in ("adapter", "internal", "typed", "existing", "target")
    )


def test_allowlists_are_short_named_and_justified():
    assert len(ARCH_FRONT_DOOR_ALLOWLIST) <= 2
    assert len(TEXT_SELECTOR_ALLOWLIST) <= 4
    assert len(PYBIND_LEGACY_NAME_ALLOWLIST) <= 2
    assert len(LARGE_RUNTIME_FILE_BUDGETS) <= 8

    allowlists = (
        ARCH_FRONT_DOOR_ALLOWLIST,
        TEXT_SELECTOR_ALLOWLIST,
        PYBIND_LEGACY_NAME_ALLOWLIST,
        LARGE_RUNTIME_FILE_BUDGETS,
    )
    for allowlist in allowlists:
        for key in allowlist:
            key_text = "/".join(key) if isinstance(key, tuple) else str(key)
            assert "*" not in key_text and "..." not in key_text, (
                "architecture allowlists must name exact files/functions, got %r" % (key,))

    for reason in ARCH_FRONT_DOOR_ALLOWLIST.values():
        assert _has_meaningful_reason(reason)
    for reason in TEXT_SELECTOR_ALLOWLIST.values():
        assert _has_meaningful_reason(reason)
    for reason in PYBIND_LEGACY_NAME_ALLOWLIST.values():
        assert _has_meaningful_reason(reason)


def test_target_surface_does_not_construct_system_front_doors():
    violations = []
    for path in _target_surface_files():
        tree = _parse(path)
        rel = _rel(path)

        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                module = node.module
                if any(module == root or module.startswith(root + ".")
                       for root in RUNTIME_MODULE_ROOTS):
                    for alias in node.names:
                        if alias.name in FRONT_DOORS:
                            violations.append(
                                "%s:%d imports %s from %s" % (rel, node.lineno, alias.name, module))

            if isinstance(node, ast.Call):
                name = _dotted_name(node.func)
                if _is_runtime_front_door_name(name):
                    violations.append("%s:%d calls legacy runtime front-door %s()"
                                      % (rel, node.lineno, name))
            elif isinstance(node, ast.Attribute):
                name = _dotted_name(node)
                if _is_runtime_front_door_name(name):
                    violations.append("%s:%d references legacy runtime front-door %s"
                                      % (rel, node.lineno, name))

    assert not violations, (
        "target authoring layers must lower through Problem/Program/layout descriptors, not construct "
        "System/AmrSystem directly:\n  " + "\n  ".join(violations)
    )


def test_architecture_surface_tests_do_not_use_runtime_legacy_setters():
    violations = []
    for path in _py_files(ARCH_TESTS):
        rel = _rel(path)
        tree = _parse(path)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                name = _dotted_name(node.func)
                if _is_runtime_front_door_name(name) and rel not in ARCH_FRONT_DOOR_ALLOWLIST:
                    violations.append("%s:%d constructs %s" % (rel, node.lineno, name))
                attr = node.func.attr if isinstance(node.func, ast.Attribute) else None
                if attr in LEGACY_RUNTIME_SETTERS:
                    violations.append("%s:%d calls runtime setter %s()" % (rel, node.lineno, attr))
            elif isinstance(node, ast.Attribute):
                name = _dotted_name(node)
                if _is_runtime_front_door_name(name) and rel not in ARCH_FRONT_DOOR_ALLOWLIST:
                    violations.append("%s:%d references %s" % (rel, node.lineno, name))

    assert not violations, (
        "architecture tests for the target surface must not reintroduce legacy runtime setters or "
        "new System/AmrSystem front-door probes:\n  " + "\n  ".join(violations)
    )


def test_public_program_surface_does_not_call_runtime_directly():
    violations = []
    for path in _py_files(POPS / "time"):
        tree = _parse(path)
        rel = _rel(path)
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                if node.module.startswith(("pops.runtime", "pops._pops")):
                    violations.append("%s:%d imports %s" % (rel, node.lineno, node.module))
            elif isinstance(node, ast.Call):
                name = _dotted_name(node.func)
                attr = node.func.attr if isinstance(node.func, ast.Attribute) else None
                if _is_runtime_front_door_name(name):
                    violations.append("%s:%d constructs %s" % (rel, node.lineno, name))
                if attr in LEGACY_RUNTIME_SETTERS:
                    violations.append("%s:%d calls runtime setter %s()" % (rel, node.lineno, attr))

    assert not violations, (
        "pops.time must build Program IR only; runtime installation belongs behind compile/bind:\n  "
        + "\n  ".join(violations)
    )


def test_field_solve_facade_lowers_to_program_ir_and_context():
    try:
        from pops.time import Program
    except Exception as exc:  # pragma: no cover - bare source tree without importable pops.
        pytest.skip("pops import unavailable: %s" % exc)

    program = Program("arch_field_gate")
    state = program.state("gas")
    fields = program.solve_fields(state)

    assert fields.op == "solve_fields"
    assert fields.vtype == "fields"
    assert any(value.op == "solve_fields" for value in program._values)

    rhs = program._rhs_legacy(state=state, fields=fields, flux=True, sources=["default"])
    program.commit("gas", program.linear_combine("U_next", state + program.dt * rhs))
    source = program.emit_cpp_program(model=None)

    assert "ProgramContext" in source
    assert "ctx.solve_fields_from_state" in source
    assert "System(" not in source and "AmrSystem(" not in source


def test_amr_route_lowers_through_typed_layout_policy_manifest():
    try:
        from pops.mesh import CartesianMesh
        from pops.mesh.amr import (
            AMROutput,
            AllLevels,
            CheckpointPolicy,
            PatchLayout,
            ProperNesting,
            Refine,
            RegridEvery,
            TagUnion,
        )
        from pops.mesh.layouts import AMR
    except Exception as exc:  # pragma: no cover - bare source tree without importable pops.
        pytest.skip("pops import unavailable: %s" % exc)

    layout = AMR(
        base=CartesianMesh(n=16, L=1.0),
        max_levels=2,
        ratio=2,
        regrid=RegridEvery(4),
        patches=PatchLayout(distribute_coarse=True, coarse_max_grid=16),
        refine=TagUnion(Refine.on("rho").above(0.1), Refine.on("phi").gradient_above(0.2)),
        nesting=ProperNesting(buffer=1),
        checkpoint=CheckpointPolicy(restartable=True),
        output=AMROutput(fields=("rho",), levels=AllLevels(), include_patch_boxes=True),
    )

    assert layout.validate() is True
    manifest = layout.inspect()
    assert manifest["capabilities"]["layout"] == "amr"
    assert manifest["available"]["ok"] is True

    amr_report = manifest["amr_report"]
    assert amr_report["layout"] == "amr"
    assert amr_report["ratio"] == 2
    slots = {row["slot"] for row in amr_report["policies"]}
    assert {"refine", "regrid", "patches", "nesting", "checkpoint", "output"} <= slots

    # ADC-589 criterion #34: sim.amr.inspect() is the unified hierarchy/patch/regrid/limitations
    # view. Building a live AmrSystem needs the native extension; skip cleanly without it (the
    # manifest assertions above already run source-only). ADC-545: the engine left the top-level
    # surface, so reach it through the advanced pops.runtime.system seam.
    try:
        from pops.runtime.system import AmrSystem  # ADC-545 advanced runtime seam
        sim = AmrSystem(n=16, L=1.0, periodic=True, regrid_every=4)
    except Exception as exc:  # pragma: no cover - native extension unavailable in this build.
        pytest.skip("AmrSystem construction unavailable: %s" % exc)

    report = sim.amr.inspect()
    payload = report.to_dict()
    assert set(payload) == {"hierarchy", "patches", "regrid", "limitations"}
    assert payload["hierarchy"]["max_levels"] == 2
    assert payload["patches"]["built"] is False
    assert payload["regrid"]["regrid_every"] == 4
    assert isinstance(payload["limitations"], list)


class _FakeAmrRefineModel:
    """A minimal model advertising its declared subjects (mirrors HyperbolicModel's surface)."""

    cons_names = ["rho"]
    cons_roles = None


def test_uniform_plus_amr_tags_refused_by_default():
    try:
        import pops
        from pops.mesh import CartesianMesh
        from pops.mesh.amr import Refine
        from pops.mesh.layouts import Uniform
    except Exception as exc:  # pragma: no cover - bare source tree without importable pops.
        pytest.skip("pops import unavailable: %s" % exc)

    layout = Uniform(CartesianMesh(n=16), refine=Refine.on("rho").above(0.1))
    case = pops.Problem(layout=layout).block("ne", physics=_FakeAmrRefineModel())
    with pytest.raises(ValueError, match="carries active AMR criteria"):
        case.validate()


def test_ignore_amr_criteria_escape():
    try:
        import pops
        from pops.mesh import CartesianMesh
        from pops.mesh.amr import IgnoreAMRCriteria, Refine
        from pops.mesh.layouts import Uniform
    except Exception as exc:  # pragma: no cover - bare source tree without importable pops.
        pytest.skip("pops import unavailable: %s" % exc)

    layout = Uniform(
        CartesianMesh(n=16),
        refine=Refine.on("rho").above(0.1),
        ignore_amr=IgnoreAMRCriteria())
    case = pops.Problem(layout=layout).block("ne", physics=_FakeAmrRefineModel())
    # The explicit escape is honoured: no refusal.
    case.validate()


def test_text_behavior_selectors_stay_on_explicit_allowlist():
    violations = []
    seen_allowlist = set()
    for path in _text_selector_surface_files():
        rel = _rel(path)
        tree = _parse(path)
        for qual, function in _function_qualifiers(tree):
            for arg_name, default, lineno in _string_defaults(function):
                if arg_name not in TEXT_SELECTOR_ARGS:
                    continue
                key = (rel, qual, arg_name)
                if key in TEXT_SELECTOR_ALLOWLIST:
                    seen_allowlist.add(key)
                    continue
                violations.append(
                    "%s:%d %s(%s=%r) is a new behavior string selector"
                    % (rel, lineno, qual, arg_name, default))

    missing = set(TEXT_SELECTOR_ALLOWLIST) - seen_allowlist
    assert not missing, "stale text-selector allowlist entries: %s" % sorted(missing)
    assert not violations, (
        "new target-surface behavior selectors must be typed descriptors/handles, not string "
        "defaults:\n  " + "\n  ".join(violations)
    )


def test_pybind_legacy_names_remain_internal_to_adapter_bindings():
    def_re = re.compile(r"\.def\s*\(\s*\"([A-Za-z_][A-Za-z0-9_]*)\"")
    class_re = re.compile(r"py::class_<[^>]+>\s*\([^,]+,\s*\"(System|AmrSystem)\"")
    violations = []
    seen_allowlist = set()

    for path in sorted(BINDINGS.rglob("*.cpp")):
        rel = _rel(path)
        text = _read(path)
        names = {m.group(1) for m in def_re.finditer(text)}
        names.update(m.group(1) for m in class_re.finditer(text))
        legacy = sorted(names & PYBIND_LEGACY_NAMES)
        if not legacy:
            continue
        if rel in PYBIND_LEGACY_NAME_ALLOWLIST:
            seen_allowlist.add(rel)
            continue
        violations.append("%s exports legacy pybind names %s" % (rel, legacy))

    missing = set(PYBIND_LEGACY_NAME_ALLOWLIST) - seen_allowlist
    assert not missing, "stale pybind legacy allowlist entries: %s" % sorted(missing)
    assert not violations, (
        "legacy runtime pybind names may only live in the two internal adapter init files:\n  "
        + "\n  ".join(violations)
    )


def _is_specialized_binding_unit(path):
    rel = _rel(path)
    name = path.name
    if rel == "python/bindings/system/base/system.cpp":
        return False
    if rel.startswith("python/bindings/system/") and name.startswith("system_"):
        return True
    if rel.startswith("python/bindings/amr/block/") and name.startswith("amr_block_"):
        return True
    if rel.startswith("python/bindings/amr/compiled/") and name.startswith("amr_compiled_"):
        return True
    return False


def test_specialized_numeric_binding_units_have_internal_justification():
    violations = []
    for path in sorted(BINDINGS.rglob("*.cpp")):
        if not _is_specialized_binding_unit(path):
            continue
        header = "\n".join(_read(path).splitlines()[:8]).lower()
        has_ticket = "adc-" in header
        has_reason = any(
            token in header
            for token in ("seam", "subdivision", "instantiates", "dispatcher", "compiles")
        )
        if not has_ticket or not has_reason:
            violations.append("%s needs an ADC ticket and internal seam/subdivision reason"
                              % _rel(path))

    assert not violations, (
        "specialized numeric-combination binding units must justify why they exist instead of "
        "becoming a duplicate runtime route:\n  " + "\n  ".join(violations)
    )


def test_large_runtime_and_binding_files_stay_on_frozen_budget():
    violations = []
    for rel, limit in LARGE_RUNTIME_FILE_BUDGETS.items():
        path = REPO_ROOT / rel
        assert path.exists(), "budgeted architecture file disappeared: %s" % rel
        lines = _line_count(path)
        if lines > limit:
            violations.append("%s: %d lines (budget %d)" % (rel, lines, limit))

    assert not violations, (
        "large runtime/binding files exceeded the ADC-598 frozen budget; split the route or update "
        "this allowlist with a focused justification:\n  " + "\n  ".join(violations)
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
