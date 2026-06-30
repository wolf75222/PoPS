"""Spec-corrective architecture gates.

These tests are intentionally narrow but hard: they guard the final public route
and the final examples against reintroducing transitional APIs or false
operator-first lowering. They do not scan the historical test suite, which still
contains explicit legacy regression coverage.
"""

import ast
import importlib
from pathlib import Path

import pytest

import pops
from pops import model
from pops import physics
from examples.spec_final import lib_time_predictor_corrector_poisson_lorentz as lib_time
from examples.spec_final import manual_board_predictor_corrector_poisson_lorentz as manual


REPO_ROOT = Path(__file__).resolve().parents[2]


LEGACY_TOP_LEVEL = (
    "compile",
    "bind",
    "Problem",
    "Case",
    "CompiledTime",
    "Explicit",
    "IMEX",
    "Strang",
    "CondensedSchur",
    "integrate",
    "dsl",
)

LEGACY_MODULES = (
    "pops.dsl",
    "pops.integrate",
    "pops.library",
    "pops.std",
    "pops.lib.std",
)

FORBIDDEN_EXAMPLE_TOKENS = (
    "try:",
    "except ",
    "skip",
    "pops.compile(",
    "pops.bind",
    "pops.Problem",
    "pops.Case",
    "add_equation",
    "install_program",
    "_get_state",
    "_set_state",
    "_eval_rhs",
    "P.rhs",
    "P.solve_fields",
    "P.linear_source",
)

FORBIDDEN_PUBLIC_DOC_TOKENS = (
    "pops.compile(",
    "pops.bind(",
    "pops.Problem",
    "pops.Case",
    "CompiledTime",
    "add_equation",
    "install_program",
    "P.rhs",
    "P.solve_fields",
    "P.linear_source",
    "sim.add_block",
    "set_poisson(",
    "pops.dsl",
    "pops.integrate",
    "pops.library",
    "pops.std",
    "pops.lib.std",
)

FALSE_LOWERING_OPS = {"rhs", "solve_fields", "linear_source"}

MODULE_NATIVE_SURFACE_FILES = tuple(
    sorted((REPO_ROOT / "python" / "pops" / "model").glob("*.py"))
) + (
    REPO_ROOT / "python" / "pops" / "physics" / "__init__.py",
    REPO_ROOT / "python" / "pops" / "physics" / "board.py",
    REPO_ROOT / "python" / "pops" / "physics" / "_board_internals.py",
    REPO_ROOT / "python" / "pops" / "physics" / "_board_multispecies.py",
    REPO_ROOT / "python" / "pops" / "physics" / "board_handles.py",
)

FORBIDDEN_MODULE_NATIVE_DOC_TOKENS = (
    "pops.dsl",
    "dsl.Model",
    "Module.to_dsl",
    "_module_to_model",
    "pops.physics.facade.Model",
    "add_native_block",
    "Program .so",
    "DEFERRED",
    "deferred",
    "later phase",
    "source_term",
    "linear_source",
)


def _example_modules():
    return (manual, lib_time)


def test_legacy_imports_fail():
    for name in LEGACY_MODULES:
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module(name)


def test_top_level_has_no_legacy_symbols():
    assert hasattr(pops, "compile_problem")
    for name in LEGACY_TOP_LEVEL:
        assert not hasattr(pops, name), "pops.%s is a forbidden legacy public symbol" % name


def test_no_public_runtime_fragmented_api():
    sim = pops.System(n=4)
    for name in ("add_equation", "install_program", "add_block", "set_poisson"):
        assert not hasattr(sim, name), "System.%s must not be public" % name

    program = pops.time.Program("api_gate")
    for name in ("rhs", "solve_fields", "linear_source", "source"):
        assert not hasattr(program, name), "Program.%s must not be public" % name
    assert hasattr(program, "call")


def test_module_and_physics_model_are_module_native_public_surfaces():
    mod = model.Module("module_gate")
    board = physics.Model("board_gate")

    for obj in (mod, model.Module, board, physics.Model):
        for name in ("to_dsl", "dsl", "_dsl", "_m", "_pde_model", "compile", "run"):
            assert not hasattr(obj, name), "%r exposes forbidden legacy surface %s" % (obj, name)

    lowered = board.to_module()
    assert isinstance(lowered, model.Module)
    assert lowered is board.module


def test_model_and_board_facade_import_graph_has_no_runtime_or_codegen():
    forbidden = {"pops.runtime", "pops.codegen", "_pops"}
    offenders = []
    for path in MODULE_NATIVE_SURFACE_FILES:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    name = alias.name
                    if name in forbidden or any(name.startswith(prefix + ".") for prefix in forbidden):
                        offenders.append("%s imports %s" % (path.relative_to(REPO_ROOT), name))
            elif isinstance(node, ast.ImportFrom):
                module_name = node.module or ""
                if node.level == 0:
                    imported = [module_name]
                    if module_name == "pops":
                        imported.extend("pops.%s" % alias.name for alias in node.names)
                    for name in imported:
                        if name in forbidden or any(name.startswith(prefix + ".") for prefix in forbidden):
                            offenders.append("%s imports %s" % (path.relative_to(REPO_ROOT), name))
    assert offenders == []


def test_module_native_docstrings_do_not_advertise_legacy_routes():
    offenders = {}
    for path in MODULE_NATIVE_SURFACE_FILES:
        text = path.read_text(encoding="utf-8")
        hits = [tok for tok in FORBIDDEN_MODULE_NATIVE_DOC_TOKENS if tok in text]
        if hits:
            offenders[str(path.relative_to(REPO_ROOT))] = hits
    assert offenders == {}


def test_examples_no_skip_or_legacy_route_tokens():
    for module in _example_modules():
        text = Path(module.__file__).read_text(encoding="utf-8")
        offenders = [tok for tok in FORBIDDEN_EXAMPLE_TOKENS if tok in text]
        assert offenders == [], "%s contains forbidden public-route token(s): %s" % (
            module.__file__, offenders)


def test_active_docs_and_public_examples_do_not_advertise_legacy_routes():
    roots = (REPO_ROOT / "README.md", REPO_ROOT / "docs", REPO_ROOT / "examples")
    ignored_parts = {
        ("docs", "archive"),
        ("docs", "_build"),
        ("docs", "validation"),
    }
    ignored_files = {
        ("docs", "SPEC_CORRECTIVE_TASKS.md"),
    }
    offenders = {}
    for root in roots:
        paths = [root] if root.is_file() else list(root.rglob("*"))
        for path in paths:
            if not path.is_file() or path.suffix not in (".md", ".py", ".rst"):
                continue
            rel = path.relative_to(REPO_ROOT)
            parts = rel.parts
            if any(parts[:len(prefix)] == prefix for prefix in ignored_parts):
                continue
            if parts in ignored_files:
                continue
            text = path.read_text(encoding="utf-8")
            hits = [tok for tok in FORBIDDEN_PUBLIC_DOC_TOKENS if tok in text]
            if hits:
                offenders[str(rel)] = hits
    assert offenders == {}


def test_pcall_creates_call_nodes_not_false_lowering_nodes():
    for module in _example_modules():
        model = module.build_model()
        program = module.build_program(model)
        ops = [v.op for v in program._values]
        assert "call" in ops
        offenders = [v for v in program._values if v.op in FALSE_LOWERING_OPS]
        assert offenders == [], "final example lowered P.call through legacy node(s): %r" % offenders


def test_generated_cpp_calls_generated_module():
    model = manual.build_model()
    program = manual.build_program(model)
    cpp = program._emit_cpp_program_for_target(model=model)

    for op in ("fields_from_state", "explicit_rate", "implicit_operator"):
        assert "GeneratedModule::Operators::%s" % op in cpp

    body_start = cpp.index("auto generated_program_body")
    body_end = cpp.index("ctx.install", body_start)
    program_body = cpp[body_start:body_end]
    assert "ctx.rhs_into" not in program_body
    assert "ctx.solve_fields_from_state" not in program_body
    assert "ctx.source_default_into" not in program_body
    assert "ctx.neg_div_flux_default_into" not in program_body


def test_no_false_lowering_tokens_in_call_dispatch_source():
    source = Path(pops.codegen.program_emit_ops.__file__).read_text(encoding="utf-8")
    call_start = source.index('elif v.op == "call":')
    call_end = source.index('elif v.op == "solve_fields":', call_start)
    call_branch = source[call_start:call_end]

    assert "GeneratedModule::Operators" in call_branch
    for forbidden in ("ctx.rhs_into", "ctx.solve_fields", "ctx.solve_fields_from_state",
                      "ctx.source_default_into", "ctx.neg_div_flux_default_into"):
        assert forbidden not in call_branch


def test_pcall_generated_module_path_has_no_notimplemented_errors():
    """The final operator-first path must validate clearly, not surface NotImplementedError."""
    source = (REPO_ROOT / "python" / "pops" / "time" / "program_core.py").read_text(
        encoding="utf-8")
    lower_start = source.index("    def _lower_call(")
    lower_end = source.index("    def _lower_coupled_rate(", lower_start)
    assert "NotImplementedError" not in source[lower_start:lower_end]

    source = (REPO_ROOT / "python" / "pops" / "codegen" / "program_emit_ops.py").read_text(
        encoding="utf-8")
    call_start = source.index('elif v.op == "call":')
    call_end = source.index('elif v.op == "solve_fields":', call_start)
    assert "NotImplementedError" not in source[call_start:call_end]

    source = (REPO_ROOT / "python" / "pops" / "codegen" / "program_emit_module_ops.py").read_text(
        encoding="utf-8")
    assert "NotImplementedError" not in source

    source = (REPO_ROOT / "python" / "pops" / "codegen" / "program_codegen.py").read_text(
        encoding="utf-8")
    call_start = source.index('    if v.op == "call":')
    call_end = source.index("    if v.op in _MODEL_OPS:", call_start)
    assert "NotImplementedError" not in source[call_start:call_end]


def test_model_kernel_missing_declarations_are_validation_errors():
    """Missing model declarations are invalid authoring, not deferred implementation routes."""
    from pops.codegen.program_emit_kernels import _model_impl
    from pops.codegen.program_emit_model_kernels import (
        _emit_flux_kernel,
        _emit_source_kernel,
        _linear_source_rows,
    )

    mdl = manual.build_model()
    impl = _model_impl(mdl)

    for emit in (
        lambda: _emit_source_kernel(mdl, "missing", "u", "r"),
        lambda: _emit_flux_kernel(mdl, ["missing"], "u", "fx", "fy"),
        lambda: _linear_source_rows(impl, "missing"),
    ):
        with pytest.raises(ValueError):
            emit()
