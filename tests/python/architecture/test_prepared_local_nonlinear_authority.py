"""ADC-750 fences for the sole prepared local nonlinear solver authority."""

from __future__ import annotations

import ast
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
PROVIDER = ROOT / "include/pops/numerics/nonlinear/prepared_local_nonlinear.hpp"
COLLECTIVE = ROOT / "include/pops/numerics/nonlinear/local_nonlinear_collective.hpp"
IMPLICIT_STEPPER = ROOT / "include/pops/numerics/time/integrators/implicit_stepper.hpp"
MODEL_KERNELS = ROOT / "python/pops/codegen/program_emit_model_kernels.py"
PROGRAM_OPS = ROOT / "python/pops/codegen/program_emit_ops.py"


def _without_cpp_comments(source: str) -> str:
    return re.sub(r"//.*?$|/\*.*?\*/", "", source, flags=re.MULTILINE | re.DOTALL)


def _cpp_body(source: str, signature: str) -> str:
    start = source.index(signature)
    opening = source.index("{", start + len(signature))
    depth = 0
    for index in range(opening, len(source)):
        if source[index] == "{":
            depth += 1
        elif source[index] == "}":
            depth -= 1
            if depth == 0:
                return source[opening + 1 : index]
    raise AssertionError(f"unterminated C++ body for {signature!r}")


def _python_function_source(path: Path, name: str) -> str:
    source = path.read_text(encoding="utf-8")
    module = ast.parse(source)
    functions = [
        node
        for node in module.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name
    ]
    assert len(functions) == 1
    function = functions[0]
    assert function.end_lineno is not None
    return "\n".join(source.splitlines()[function.lineno - 1 : function.end_lineno])


def test_prepared_provider_is_the_only_local_nonlinear_algorithm_definition():
    definition = re.compile(
        r"LocalNonlinearCellResult<N>\s+solve_prepared_local_nonlinear\s*\("
    )
    definitions = [
        path.relative_to(ROOT).as_posix()
        for path in sorted((ROOT / "include").rglob("*.hpp"))
        if definition.search(_without_cpp_comments(path.read_text(encoding="utf-8")))
    ]
    assert definitions == ["include/pops/numerics/nonlinear/prepared_local_nonlinear.hpp"]

    provider = _without_cpp_comments(PROVIDER.read_text(encoding="utf-8"))
    solve = _cpp_body(provider, "solve_prepared_local_nonlinear(")
    assert "pivoted_dense_solve" in solve
    assert "build_local_jacobian" in solve
    assert "LocalNonlinearCellResult<N> result;" in solve
    for forbidden in (
        "mat_inverse",
        "std::function",
        "std::vector",
        "std::unique_ptr",
        "std::shared_ptr",
        "malloc(",
        "calloc(",
        "realloc(",
        "throw ",
    ):
        assert forbidden not in solve


def test_generated_program_routes_delegate_instead_of_emitting_newton():
    for function_name in (
        "_emit_solve_coupled_implicit_kernel",
        "_emit_solve_local_nonlinear_kernel",
    ):
        source = _python_function_source(MODEL_KERNELS, function_name)
        assert source.count("solve_prepared_local_nonlinear") == 1
        for forbidden in (
            "mat_inverse",
            "pivoted_dense_solve",
            "build_local_jacobian",
            "for (int newton",
            "for (int iteration",
        ):
            assert forbidden not in source


def test_implicit_source_device_kernel_is_a_stack_only_provider_adapter():
    source = _without_cpp_comments(IMPLICIT_STEPPER.read_text(encoding="utf-8"))
    adapter = source.split("struct PreparedImplicitSourceKernel", 1)[1].split(
        "struct LocalStatValue", 1
    )[0]
    kernel = _cpp_body(
        adapter, "POPS_HD void operator()(const Index<Dim>& index) const"
    )
    assert kernel.count("solve_prepared_local_nonlinear") == 1
    assert "LocalNonlinearCellResult<N> solved;" in kernel
    for forbidden in (
        "mat_inverse",
        "pivoted_dense_solve",
        "build_local_jacobian",
        "std::function",
        "std::vector",
        "std::unique_ptr",
        "std::shared_ptr",
        "malloc(",
        "calloc(",
        "realloc(",
        "throw ",
    ):
        assert forbidden not in kernel


def test_implicit_source_publication_consumes_one_collective_outcome():
    source = _without_cpp_comments(IMPLICIT_STEPPER.read_text(encoding="utf-8"))
    publication = _cpp_body(
        source, "const MultiFab<Dim, MemorySpace>* active_cells = nullptr)"
    )
    assert publication.count("PreparedImplicitSourceKernel<Dim, Model>") == 1
    assert publication.count("SolveOutcome::collective_world") == 1
    assert "ImplicitSourcePublication" in publication
    assert "solved_value_available()" not in publication


def test_failure_location_uses_staged_integer_collectives_without_float_packing():
    provider = PROVIDER.read_text(encoding="utf-8")
    implicit = IMPLICIT_STEPPER.read_text(encoding="utf-8")
    generated = MODEL_KERNELS.read_text(encoding="utf-8")
    program_ops = PROGRAM_OPS.read_text(encoding="utf-8")
    collective = COLLECTIVE.read_text(encoding="utf-8")

    for source in (provider, implicit, generated, program_ops):
        assert "encode_local_nonlinear_failure" not in source
        assert "encode_ranked_local_nonlinear_failure" not in source
    assert "Kokkos::atomic_min" in collective
    assert "all_reduce_min(static_cast<long>" in collective
    assert "LocalNonlinearFailureAxisMin" in collective
    assert "LocalNonlinearFailureComponentMin" in collective
    assert "for (int axis = Dim - 1; axis >= 0; --axis)" in collective
    assert "collective_first_local_nonlinear_failure" in implicit
    assert "Kokkos::Min<int>" not in implicit
    assert "Kokkos::Rank<" not in implicit


def test_implicit_provider_and_failure_collective_share_one_ranked_authority():
    implicit = _without_cpp_comments(IMPLICIT_STEPPER.read_text(encoding="utf-8"))
    collective = _without_cpp_comments(COLLECTIVE.read_text(encoding="utf-8"))

    for source in (implicit, collective):
        for forbidden in (
            "Box2D",
            "Fab2D",
            "Array4",
            "ConstArray4",
            ".array(",
            ".const_array(",
        ):
            assert forbidden not in source
    assert "FieldView<const Real, Dim>" in implicit
    assert "MultiFab<Dim, MemorySpace>" in implicit
    assert "LocalNonlinearFailureLocation<Dim>" in collective
    assert "Index<Dim> selected" in collective
    assert not re.search(r"if constexpr\s*\(\s*Dim\b", implicit)
    assert not re.search(r"if constexpr\s*\(\s*Dim\b", collective)
