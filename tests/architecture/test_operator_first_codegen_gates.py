"""TASK-070 source gates for operator-first Program codegen.

These tests are intentionally source-only so the fast architecture CI can catch a
regression before any C++ toolchain is available.
"""

import ast
import pathlib


REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
POPS = REPO_ROOT / "python" / "pops"


def _source(path):
    return path.read_text(encoding="utf-8")


def _function_source(path, name):
    text = _source(path)
    tree = ast.parse(text, filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return ast.get_source_segment(text, node)
    raise AssertionError("function %s not found in %s" % (name, path.relative_to(REPO_ROOT)))


def _call_branch_source():
    path = POPS / "codegen" / "program_emit_ops.py"
    text = _function_source(path, "_emit_op")
    start = text.index('elif v.op == "call":')
    end = text.index('elif v.op == "solve_fields":', start)
    return text[start:end]


def test_pcall_creates_call_node():
    """TASK-070: P.call must record first-class call IR nodes, not legacy rhs/fields nodes."""
    node_src = _function_source(POPS / "time" / "program_core.py", "_call_node")
    assert '"operator_id"' in node_src
    assert '"operator_handle"' in node_src
    assert '"output_type"' in node_src
    assert '"output_vtype"' in node_src
    assert 'self._new(storage_vtype, "call"' in node_src
    serialize_src = _function_source(POPS / "time" / "program_passes.py", "_serialize_node")
    assert "missing output_vtype" in serialize_src
    assert 'setdefault("output_vtype"' not in serialize_src

    src = _function_source(POPS / "time" / "program_core.py", "_lower_call")
    assert src.count("self._call_node(") == 4
    for forbidden in (
        'self._new("fields", "call"',
        'self._new("rhs", "call"',
        'self._new("operator", "call"',
        'self._new("state", "call"',
        "_fields_from_state",
        "_rate_from_transport",
        "_legacy_rhs",
        "_legacy_solve_fields",
        '"kind"',
    ):
        assert forbidden not in src, "_lower_call must not reintroduce %s" % forbidden


def test_generated_cpp_calls_generated_module():
    """TASK-070: Program call lowering must route through GeneratedModule::Operators."""
    template = _source(POPS / "codegen" / "program_emit_kernels.py")
    assert "namespace GeneratedProgram" in template
    assert "GeneratedProgram::step(ctx, dt, generated_program_body)" in template
    assert "auto generated_program_body = [=](auto& ctx, double dt)" in template

    src = _call_branch_source()
    assert "GeneratedModule::Operators::%s" in src
    assert "operator_function_name" in src
    assert "output_vtype" in src
    assert "missing output_vtype" in src
    for forbidden in (
        'v.vtype == "fields"',
        'v.vtype == "rhs"',
        'v.vtype == "operator"',
        'v.vtype == "state"',
        "ctx.solve_fields_from_state",
        "ctx.solve_fields_from_blocks",
        "ctx.rhs_into",
        "ctx.neg_div_flux_default_into",
        "_emit_flux_kernel",
        "_emit_source_kernel",
        "_emit_apply_kernel",
        "/* local_linear_operator",
    ):
        assert forbidden not in src, (
            "v.op == 'call' must not dispatch directly through %s; GeneratedModule owns it"
            % forbidden)


def test_problem_so_is_primary_native_abi():
    """TASK-024/TASK-027: generated artifacts expose the problem.so ABI."""
    template = _source(POPS / "codegen" / "program_emit_kernels.py")
    amr = _source(POPS / "codegen" / "program_emit_amr.py")
    system_loader = _source(REPO_ROOT / "python" / "bindings" / "system" / "base" / "system.cpp")
    amr_loader = _source(REPO_ROOT / "python" / "bindings" / "amr" / "amr_system.cpp")

    for token in (
        "pops_problem_abi_key",
        "pops_problem_name",
        "pops_problem_hash",
        "pops_problem_install",
        "pops_problem_has_dt_bound",
        "pops_problem_dt_bound",
    ):
        assert token in template, "problem.so ABI missing %s" % token
    assert "pops_problem_install_amr" in amr

    for token in (
        '"pops_problem_abi_key"',
        '"pops_problem_install"',
        '"pops_problem_hash"',
        '"pops_problem_block_count"',
        '"pops_problem_param_count"',
        '"pops_problem_has_dt_bound"',
        '"pops_problem_dt_bound"',
    ):
        assert token in system_loader, "System loader must resolve %s" % token
    for token in (
        '"pops_problem_abi_key"',
        '"pops_problem_install_amr"',
        '"pops_problem_hash"',
        '"pops_problem_block_count"',
        '"pops_problem_param_count"',
    ):
        assert token in amr_loader, "AMR loader must resolve %s" % token

    joined_codegen = "\n".join(
        _source(path) for path in (POPS / "codegen").rglob("*.py")
    )
    assert "Program .so" not in joined_codegen
    assert "program.so" not in joined_codegen
    assert "pops_install_program" not in joined_codegen
    assert "pops_program_abi_key" not in joined_codegen
