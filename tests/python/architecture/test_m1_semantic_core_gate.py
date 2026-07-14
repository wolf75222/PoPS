"""Source-only M1 gate: exact proofs plus fences against semantic escape hatches."""
from __future__ import annotations

import ast
import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
POPS = ROOT / "python/pops"
MANIFEST = ROOT / "tests/gates/m1_semantic_core.toml"
RUNNER = ROOT / "scripts/run_m1_gate.py"


def _load_runner():
    spec = importlib.util.spec_from_file_location("pops_run_m1_gate", RUNNER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _parse(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _python_files(root: Path):
    return sorted(path for path in root.rglob("*.py") if "__pycache__" not in path.parts)


def _call_name(node: ast.Call) -> str:
    value = node.func
    parts = []
    while isinstance(value, ast.Attribute):
        parts.append(value.attr)
        value = value.value
    if isinstance(value, ast.Name):
        parts.append(value.id)
    return ".".join(reversed(parts))


def _relative(path: Path, node: ast.AST) -> str:
    return "%s:%d" % (path.relative_to(ROOT).as_posix(), getattr(node, "lineno", 0))


def test_m1_manifest_is_complete_and_references_only_real_mandatory_tests():
    _, errors = _load_runner().validate_manifest(MANIFEST)
    assert not errors, "M1 gate matrix is incomplete:\n  " + "\n  ".join(errors)


def test_public_handle_constructors_never_offer_an_ownerless_mode():
    """Handle classes may inherit ownership, but cannot make an explicit owner optional."""
    roots = (POPS / "model", POPS / "physics", POPS / "problem", POPS / "time")
    violations = []
    base_handle_seen = False
    for root in roots:
        for path in _python_files(root):
            for node in ast.walk(_parse(path)):
                if not isinstance(node, ast.ClassDef) or not node.name.endswith("Handle"):
                    continue
                init = next((child for child in node.body
                             if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
                             and child.name == "__init__"), None)
                if init is None:
                    continue
                positional = list(init.args.posonlyargs) + list(init.args.args)
                arguments = positional + list(init.args.kwonlyargs)
                owner_index = next((index for index, arg in enumerate(arguments)
                                    if arg.arg == "owner"), None)
                if owner_index is None:
                    continue
                if node.name == "Handle" and path == POPS / "model/handles.py":
                    base_handle_seen = True
                # Positional defaults align to the tail; keyword-only defaults align exactly.
                optional = False
                positional_names = [arg.arg for arg in positional]
                if "owner" in positional_names:
                    first_default = len(positional_names) - len(init.args.defaults)
                    owner_pos = positional_names.index("owner")
                    if owner_pos >= first_default:
                        default = init.args.defaults[owner_pos - first_default]
                        optional = isinstance(default, ast.Constant) and default.value is None
                else:
                    kw_pos = [arg.arg for arg in init.args.kwonlyargs].index("owner")
                    default = init.args.kw_defaults[kw_pos]
                    optional = isinstance(default, ast.Constant) and default.value is None
                if optional:
                    violations.append("%s %s.__init__(owner=None)" % (
                        _relative(path, init), node.name))
    assert base_handle_seen, "canonical model.Handle constructor was not inspected"
    assert not violations, (
        "ownerless Handle construction is forbidden:\n  " + "\n  ".join(violations))


def test_no_handle_call_explicitly_erases_its_owner():
    violations = []
    for root in (POPS / "model", POPS / "physics", POPS / "problem", POPS / "time"):
        for path in _python_files(root):
            for node in ast.walk(_parse(path)):
                if not isinstance(node, ast.Call) \
                        or not _call_name(node).split(".")[-1].endswith("Handle"):
                    continue
                for keyword in node.keywords:
                    if keyword.arg == "owner" and isinstance(keyword.value, ast.Constant) \
                            and keyword.value.value is None:
                        violations.append("%s calls %s(owner=None)" % (_relative(path, node),
                                                                      _call_name(node)))
    assert not violations, "explicit owner erasure is forbidden:\n  " + "\n  ".join(violations)


def test_phase_calls_never_enable_strict_false():
    phases = {"validate", "resolve", "compile", "bind", "install"}
    violations = []
    for path in _python_files(POPS):
        for node in ast.walk(_parse(path)):
            if not isinstance(node, ast.Call) or _call_name(node).split(".")[-1] not in phases:
                continue
            for keyword in node.keywords:
                if keyword.arg == "strict" and isinstance(keyword.value, ast.Constant) \
                        and keyword.value.value is False:
                    violations.append("%s calls %s(strict=False)" % (_relative(path, node),
                                                                     _call_name(node)))
    assert not violations, "canonical phases have no permissive mode:\n  " + "\n  ".join(violations)


def test_compiled_records_do_not_store_live_authoring_references():
    forbidden = {
        "_problem", "problem", "authoring", "authoring_problem", "source_problem",
        "source_program", "builder", "dsl", "operator_registry", "declaration_index",
    }
    files = (POPS / "codegen/_plans.py", POPS / "codegen/_compiled_artifact.py")
    record_names = {
        "ResolvedSimulationPlan", "CompiledBlockArtifact", "CompiledSimulationArtifact",
        "BindInputs", "InstallPlan",
    }
    violations = []
    seen = set()
    for path in files:
        for node in _parse(path).body:
            if not isinstance(node, ast.ClassDef) or node.name not in record_names:
                continue
            seen.add(node.name)
            for child in ast.walk(node):
                name = None
                if isinstance(child, ast.AnnAssign) and isinstance(child.target, ast.Name):
                    name = child.target.id
                elif isinstance(child, ast.Attribute) and isinstance(child.value, ast.Name) \
                        and child.value.id == "self":
                    name = child.attr
                if name in forbidden:
                    violations.append("%s %s stores %s" % (_relative(path, child), node.name, name))
    assert seen == record_names, (
        "compiled-boundary scan missed records %s" % sorted(record_names - seen))
    assert not violations, (
        "compiled values retain live authoring references:\n  " + "\n  ".join(violations))


def test_public_phase_functions_do_not_probe_or_swallow_unknown_shapes():
    path = POPS / "codegen/_phases.py"
    phases = {"validate", "resolve", "compile", "bind", "install"}
    violations = []
    seen = set()
    for node in _parse(path).body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) or node.name not in phases:
            continue
        seen.add(node.name)
        for child in ast.walk(node):
            if isinstance(child, ast.Call) and _call_name(child) in {"getattr", "hasattr"}:
                violations.append("%s %s uses permissive %s" % (
                    _relative(path, child), node.name, _call_name(child)))
            if isinstance(child, ast.ExceptHandler):
                caught = child.type
                if caught is None or (isinstance(caught, ast.Name)
                                      and caught.id in {"Exception", "BaseException"}):
                    violations.append("%s %s swallows a broad exception" % (
                        _relative(path, child), node.name))
    assert seen == phases, "phase probing scan missed %s" % sorted(phases - seen)
    assert not violations, "canonical phases must fail closed:\n  " + "\n  ".join(violations)


def test_runtime_does_not_import_superseded_whole_problem_loaders():
    forbidden_modules = {"pops.codegen._compile_drivers", "pops.codegen._loader_dump"}
    forbidden_names = {"CompiledProblem", "compile_problem", "CompiledProblemDumpMixin"}
    violations = []
    for path in _python_files(POPS / "runtime"):
        for node in ast.walk(_parse(path)):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name in forbidden_modules:
                        violations.append("%s imports %s" % (_relative(path, node), alias.name))
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                names = {alias.name for alias in node.names}
                if module in forbidden_modules or names & forbidden_names:
                    violations.append("%s imports %s from %s" % (
                        _relative(path, node), sorted(names), module))
    assert not violations, (
        "runtime imports superseded whole-problem loaders:\n  " + "\n  ".join(violations))

    compile_facade = _parse(POPS / "codegen/_compile.py")
    retired_exports = []
    for node in ast.walk(compile_facade):
        if isinstance(node, ast.ImportFrom):
            names = {alias.name for alias in node.names}
            retired_exports.extend(sorted(names & {"compile_problem", "_module_to_model"}))
    assert not retired_exports, (
        "pops.codegen._compile re-exports superseded whole-problem compiler paths: %s"
        % sorted(set(retired_exports)))

    legacy_tokens = {
        "runtime.native_loader.legacy_metadata",
        "report_and_compat",
        "kNativeLoaderLegacyMetadata",
        "allow_legacy_abi",
    }
    roots = (POPS / "runtime", ROOT / "include/pops/runtime")
    hits = []
    for root in roots:
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.suffix not in {".py", ".hpp", ".cpp"}:
                continue
            text = path.read_text(encoding="utf-8")
            for token in legacy_tokens:
                if token in text:
                    hits.append("%s contains %s" % (path.relative_to(ROOT), token))
    assert not hits, "runtime retains legacy loader compatibility:\n  " + "\n  ".join(hits)
