"""Exact public Block keys, collision-free scratch families; actual Host TUs."""
from pathlib import Path
import re
import subprocess
import sys
import pytest
from pops.codegen.cpp_strings import cpp_string_literal
from pops.codegen.cpp_symbols import block_scratch_identifiers, variable_scope
from pops.codegen.program_codegen import emit_cpp_program
from pops.codegen.program_models import ProgramModelGraph
from tests.python.support.public_coupled_block_name_case import build

ROOT = Path(__file__).resolve().parents[4]


def test_token_table_is_injective_order_independent_and_keeps_safe_legacy():
    labels = ("a-b", "a_b", "a", "aA", "κ", "λ", "for", "pops_block_612d62")
    table = block_scratch_identifiers("cr2_", labels)
    assert table == block_scratch_identifiers("cr2_", reversed(labels))
    assert table["a_b"] == "cr2_a_b" and table["for"] == "cr2_for"
    tokens = [token + suffix for token in table.values() for suffix in ("", "A")]
    assert len(tokens) == len(set(tokens))
    assert all(re.fullmatch(r"[a-zA-Z_][a-zA-Z_0-9]*", token) for token in tokens)
    assert table["a-b"] != table["pops_block_612d62"]
    with variable_scope((("cons", "cr2_safeA"),)):
        assert block_scratch_identifiers("cr2_", ("safe",))["safe"] != "cr2_safe"
    assert block_scratch_identifiers("cr2_", ("safe",))["safe"] == "cr2_safe"


def compile_program(source, tmp_path):
    cpp = tmp_path / "program.cpp"
    cpp.write_text(source)
    command = ["/usr/bin/clang++", "-Xpreprocessor", "-fopenmp", "-std=c++20",
        "-DPOPS_NATIVE_DIM=2", "-DPOPS_HAS_KOKKOS=1", "-DPOPS_RUNTIME_SHARED_EXCEPTION_ABI=1",
        "-I" + str(ROOT / "include"), "-I" + str(Path(sys.prefix) / "include"),
        "-fsyntax-only", str(cpp)]
    result = subprocess.run(command, capture_output=True)
    (tmp_path / "command.json").write_text(__import__("json").dumps(command))
    (tmp_path / "compile.stdout").write_bytes(result.stdout)
    (tmp_path / "compile.stderr").write_bytes(result.stderr)
    assert result.returncode == 0, result.stderr.decode()


@pytest.mark.parametrize("labels,reverse", [
    (("z-first", "a-second"), False), (("a-b", "a_b"), False),
    (("a-b", "a_b"), True), (("κ", "λ"), True),
    (("a", "aA"), False), (('quoted"block', "back\\slash"), True),
])
def test_complete_public_coupled_program(labels, reverse, tmp_path):
    resolved = build(labels, reverse=reverse)
    graph = ProgramModelGraph.from_resolved_blocks(resolved.blocks)
    before = resolved.time._ir_hash()
    source = emit_cpp_program(resolved.time, model_graph=graph)
    assert source == emit_cpp_program(resolved.time, model_graph=graph)
    assert resolved.time._ir_hash() == before
    assert {block.name for block in resolved.blocks} == set(labels)
    for label in labels:
        assert "return " + cpp_string_literal(label) + ";" in source
    assert "pops_input_0_component_0_" in source and "pops_input_1_component_0_" in source
    compile_program(source, tmp_path)


@pytest.mark.parametrize("reverse", (False, True))
def test_public_original_implicit_product_uses_same_block_token_table(reverse, tmp_path):
    import pops
    from tests.python.support.local_residual_product_case import make_case
    case, layout, _ = make_case(reverse=reverse, block_names=("a-b", "a_b"))
    resolved = pops.resolve(pops.validate(case), layout=layout)
    source = emit_cpp_program(resolved.time, model_graph=ProgramModelGraph.from_resolved_blocks(resolved.blocks))
    assert "prepare_local_nonlinear_problem<5>" in source
    assert "solve_prepared_local_nonlinear(prepared_, G_)" in source
    compile_program(source, tmp_path)
