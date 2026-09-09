"""Scalar solver vectors publish through exact destination storage contracts."""
from __future__ import annotations

import re

import pytest

from pops.codegen.module_lowering import lower_and_validate
from pops.codegen.program_emit_control import _emit_body
from pops.linalg import LinearOperatorProperties, LinearProblem
from pops.physics._facade import Model
from pops.solvers import krylov
from pops.time import FailRun, Program
from tests.python.support.typed_program import program_states


def _program(*, post_sync=False):
    model = Model("scalar_publication")
    (q,) = model.conservative_vars("q")
    model.flux(x=[0 * q], y=[0 * q])
    program = Program("scalar_publication")
    _, states = program_states(program, model, ("left", "right"))
    solutions = {}
    for name, temporal in states.items():
        operator = program.matrix_free_operator("identity_" + name)
        program.set_apply(operator, lambda _scope, _out, value: 2 * value)
        solution = program.solve(
            LinearProblem(operator, temporal.n, at=temporal.next.point,
                          properties=LinearOperatorProperties.general(), nullspace=None),
            solver=krylov.GMRES(max_iter=2, restart=2, rel_tol=1e-12),
        ).consume(action=FailRun())
        assert solution.vtype == "scalar_field"
        solutions[name] = solution
        program.commit(temporal.next, solution)
    if post_sync:
        def publish_again(body):
            for name, temporal in states.items():
                candidate = body.value("synchronized_" + name, solutions[name],
                                       at=temporal.next.point)
                body.commit(temporal.next, candidate)
        program.after_synchronization(publish_again)
    lowered, _ = lower_and_validate(model)
    return program, lowered


@pytest.mark.parametrize("target", ["system", "amr_system"])
def test_scalar_commit_keeps_solver_footprint_and_all_targets_provisional(target):
    program, model = _program()
    prelude, body, post_sync, _ = _emit_body(program, model, target=target)
    assert not post_sync
    # The mathematical iterate remains a zero-halo scalar, independently of each
    # physical destination. Both solves must be consumed before ANY publication.
    assert "KrylovFootprint" in prelude + body
    assert "alloc_scalar_field(1, 0)" in prelude + body or "scalar_scratch(" in body
    solves = [v for v in program._values if v.op == "solve_linear"]
    assert len(solves) == 2
    assert all(v.attrs["krylov_footprint"]["input_ghosts"] == 0 for v in solves)
    assert "solved_state" not in prelude + body
    assert body.count("ctx.commit_many(") == 1
    first_adapter = body.index("auto* commit_source_")
    assert body.rfind("ctx.solve_prepared_linear(") < first_adapter
    accepted = ".consume(pops::SolveConsumption::kAccept)"
    assert body.count(accepted) == 2
    assert body.rfind(accepted) < first_adapter
    assert body.rfind('throw std::runtime_error(std::string("solve_linear failed: ")') < first_adapter
    bases = [v for v in program._values if v.op == "state"]
    assert len(bases) == 2
    producer_ids = {v.id for v in program._values if v.op != "state"}
    for base in bases:
        assert base.id not in producer_ids
        assert "ctx.scratch_state(%d, 0, u%d)" % (base.id, base.id) in body
        assert "->ghosts() != u%d.ghosts()" % base.id in body
        assert "{&u%d, &(*commit_source_%d_0)}" % (base.id, base.id) in body
        # Only private publication storage is the copy destination.
        assert "PureFieldAlgebra::copy(u%d," % base.id not in body
    assert body.count("PureFieldAlgebra::copy(publication,") == 2


def test_post_sync_reacquires_current_destinations_with_separate_cache_keys():
    program, model = _program(post_sync=True)
    _, body, post_sync, _ = _emit_body(program, model, target="amr_system")
    ordinary_keys = set(re.findall(r"ctx.scratch_state\((\d+), 0, u\d+\)", body))
    post_keys = set(re.findall(r"ctx.scratch_state\((\d+), 1, u\d+\)", post_sync))
    assert len(ordinary_keys) == 2
    assert ordinary_keys == post_keys
    assert post_sync.count("ctx.state(") == 2
    assert post_sync.count("ctx.commit_many(") == 1
    assert post_sync.index("ctx.state(") < post_sync.index("auto* commit_source_")
    assert "ctx.rotate_histories" not in post_sync
