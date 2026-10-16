"""The prepared solver's private solution is delivered through its declared owner."""
from __future__ import annotations

import pytest

from pops.codegen.program_codegen import emit_cpp_program
from pops.solvers import CompositeTensorFAC
from test_hierarchy_scoped_solve_emit import _build


def test_hierarchy_krylov_delivers_only_after_success_and_keeps_provider_selection_lazy():
    program, source = _build(CompositeTensorFAC())
    solve = next(value for value in program._values if value.op == "solve_linear")
    owner = program._block_indices()[solve.block]
    assert owner == 1  # an equally shaped block zero must not become the owner
    delivery = "auto& delivered = ctx.scalar_scratch(%d, 0, ctx.state(1), 1, 1);" % solve.id
    assert source.count(delivery) == 1
    begin = source.index("pops::SolveOutcome kr%d = ctx.solve_prepared_linear(" % solve.id)
    end = source.index(delivery)
    assert begin < end
    between = source[begin:end]
    assert "solved_value_available()" in between
    assert "throw" in between
    assert "ctx.prepare_spatial_collectively([&]() {" in between
    assert "pops::PureFieldAlgebra::copy_allocated(delivered, *sf_sol%d);" % solve.id in source
    use = "ctx.linear_solution(%d, 1, 1, 1)" % solve.id
    assert source.count(use) >= 4  # flat consumers and separate hierarchy publication
    assert "ctx.linear_solution(*sf_sol%d)" % solve.id not in source
    assert source.index("ctx.store_history(", end) > end
    assert "ctx.solve_hierarchy_tensor(1, 1," in source


def test_hierarchy_krylov_rejects_a_changed_declared_owner_before_emission():
    program, _, model = _build(CompositeTensorFAC(), _return_model=True)
    solve = next(value for value in program._values if value.op == "solve_linear")
    object.__setattr__(solve, "attrs", {**solve.attrs, "hierarchy_block_index": 0})
    with pytest.raises(ValueError, match="owner|binding|block"):
        emit_cpp_program(program, model=model, target="amr_system")
