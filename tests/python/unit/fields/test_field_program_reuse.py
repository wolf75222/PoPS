"""Reuse exact frozen field evaluations, with conservative effect barriers."""
from __future__ import annotations

import pytest

import pops
from pops.codegen import Production
from pops.codegen.program_codegen import emit_cpp_program
from pops.codegen.program_models import ProgramModelGraph
from pops.layouts import Uniform
from pops.mesh import CartesianGrid, PeriodicAxes
from pops.solvers import CG
from pops.time import FailRun, FixedDt
from tests.python.unit.fields.test_program_field_problem import field_case


def _emit_variant(variant):
    component_transform = variant in ("momentum_update", "density_update", "coefficient_update")
    case, field, _problem, program, values, point = field_case(component_transforms=component_transform)
    program.solve(field, values=values, at=point).consume(action=FailRun())
    solver = None
    if variant == "fresh_stage":
        point = program.stage("same time", c=0)
    elif variant == "changed_state":
        key = next(iter(values))
        values[key] = program.value("updated input", 2 * values[key], at=point)
    elif variant == "effect_barrier":
        program.fill_boundary(next(iter(values.values())))
    elif variant == "solver":
        solver = CG(max_iter=5000, rel_tol=1e-12, abs_tol=1e-13)
    elif component_transform:
        key = next(iter(values))
        block = case._block_registry.handles()["first"]
        model = case._block_registry.spec("first")["model"]
        values[key] = program.transform(values[key], transform=block[model.operators[variant]])
    second = program.solve(field, values=values, at=point, solver=solver).consume(action=FailRun())
    assert second.problem_identity is not None
    for handle in values:
        current = program.state(handle)
        program.commit(current.next, program.value("unchanged_" + current.n.name, 1 * current.n, at=current.next.point))
    program.step_strategy(FixedDt(0.1))
    case.program(program)
    frame = case._block_registry.spec("first")["model"]._frame
    grid = CartesianGrid(frame=frame, cells=(16, 16), periodic=PeriodicAxes(frame.axes))
    plan = pops.resolve(pops.validate(case), layout=Uniform(grid), backend=Production())
    code = emit_cpp_program(program, model_graph=ProgramModelGraph.from_resolved_blocks(plan.blocks))
    return code


def test_exact_repeated_stage_reuses_only_one_successful_native_solve():
    code = _emit_variant("same")
    assert code.count("ctx.solve_prepared_linear(") == 1
    assert code.count("++field_solve_count_0;") == 1
    assert code.count("++field_reuse_count_0;") == 1
    assert "field.solves/" in code and "field.reuses/" in code
    # Counters live in the invocation, so attempts/restarts cannot retain old candidates.
    assert "pops::Real field_solve_count_0 = pops::Real(0)" in code


@pytest.mark.parametrize("variant", ("fresh_stage", "changed_state", "effect_barrier", "solver",
                                    "density_update", "coefficient_update"))
def test_new_context_state_effect_or_solver_invalidates_field_reuse(variant):
    code = _emit_variant(variant)
    assert code.count("ctx.solve_prepared_linear(") == 2
    assert code.count("++field_solve_count_0;") == 2
    assert "++field_reuse_count_0;" not in code


def test_proven_momentum_only_transform_reuses_density_and_coefficient_inputs():
    code = _emit_variant("momentum_update")
    assert "transformed_2_" in code
    assert code.count("ctx.solve_prepared_linear(") == 1
    assert code.count("++field_reuse_count_0;") == 1
