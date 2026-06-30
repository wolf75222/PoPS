"""Integration: final matrix-free Krylov route."""

from pathlib import Path

import pytest

from examples.spec_final import matrix_free_krylov_bicgstab as example


FORBIDDEN_PUBLIC_ROUTE_TOKENS = (
    "tr" + "y:",
    "ex" + "cept ",
    "sk" + "ip",
    "pops." + "compile(",
    "pops." + "bind",
    "pops." + "Problem",
    "pops." + "Case",
    "add_" + "equation",
    "install_" + "program",
    "_get_" + "state",
    "_set_" + "state",
    "_eval_" + "rhs",
    "P." + "rhs",
    "P." + "solve_fields",
    "P." + "linear_source",
)


def test_matrix_free_example_has_no_legacy_public_route_tokens():
    text = Path(example.__file__).read_text(encoding="utf-8")
    offenders = [token for token in FORBIDDEN_PUBLIC_ROUTE_TOKENS if token in text]
    assert offenders == []


def test_matrix_free_bicgstab_codegen_calls_cpp_krylov():
    module = example.build_model()
    program = example.build_program()
    cpp = program._emit_cpp_program_for_target(model=module)

    assert "pops::ApplyFn apply_A" in cpp
    assert "ctx.laplacian" in cpp
    assert "pops::bicgstab_solve" in cpp
    assert "krylov_solves" in cpp
    assert "krylov_iters" in cpp
    assert "did not converge" in cpp
    assert "(void)kr" not in cpp
    assert "GeneratedProgram" in cpp
    assert ("install_" + "program") not in cpp


def test_matrix_free_codegen_has_no_notimplemented_placeholders():
    import pops.codegen.program_emit_solve as emit_solve

    text = Path(emit_solve.__file__).read_text(encoding="utf-8")
    assert "NotImplementedError" not in text


def test_matrix_free_program_ir_is_krylov_not_python_callback():
    program = example.build_program()
    ops = [v.op for v in program._values]

    assert "matrix_free_operator" in ops
    assert "solve_linear" in ops
    assert "commit" not in ops
    assert not any(op in {"rhs", "solve_fields", "linear_source"} for op in ops)


@pytest.mark.requires_toolchain
def test_matrix_free_bicgstab_runs_one_cfl_step():
    result = example.run_once(n=8, cfl=0.4)
    assert result["state"].shape == (1, 8, 8)
    assert result["compiled"].problem_hash
