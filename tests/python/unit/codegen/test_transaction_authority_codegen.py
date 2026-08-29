"""Finite Program transaction authorities are declared by the generated candidate prelude."""
from __future__ import annotations

import re

import pytest

from pops.codegen.program_codegen import emit_cpp_program
from pops.codegen.program_emit_solve import (
    _SOLVE_STATUS_CPP,
    _validate_native_solve_outcome_emission,
)
from pops.codegen.program_emit_ops import _append_transaction_authority_declaration
from pops.diagnostics import BalanceLedger
from pops.identity import make_identity
from pops.linalg import LinearProblem
from pops.output._balance_due_contract import (
    BalanceDueConsumer,
    BalanceDueContract,
    BalanceDueRoute,
)
from pops.solvers.krylov import Richardson
from pops.time import BlockProjection, FailRun, Program, RejectAttempt, SOLVE_STATUSES, every
from pops.time.stencil import StencilAccess
from typed_program_support import typed_state


def _balance_contract(route, clock):
    return BalanceDueContract(
        make_identity("consumer-graph", {"test": "transaction-authority-codegen"}),
        (
            BalanceDueRoute(
                route,
                (
                    BalanceDueConsumer(
                        make_identity("consumer-manifest", {"test": "balance"}),
                        every(2, clock=clock),
                    ),
                ),
            ),
        ),
    )


def test_candidate_prelude_declares_exact_reachable_authorities_once_before_step():
    program = Program("transaction_authorities")
    temporal = typed_state(program, "fluid", state_name="U")
    total = program.sum(temporal.n)
    program.record_scalar("zeta", total)
    program.record_scalar("alpha", total)
    program.record_scalar("zeta", total)

    ledger = BalanceLedger("mass")
    program.record_balance(
        ledger,
        storage_change=total,
        outward_boundary_flux=total,
        sources=total,
        reflux=total,
        projection=total,
    )
    route = ledger.route_identity(temporal.n.block)
    candidate = program.value("candidate", temporal.n, at=temporal.next.point)
    projected = program._project_for_step_guard(
        "guard_projection",
        candidate,
        BlockProjection(),
        "embedded_error",
    )
    program.commit(temporal.next, projected)

    source = emit_cpp_program(
        program,
        balance_due_contract=_balance_contract(route, program.clock),
    )
    declarations = [
        line.strip()
        for line in source.splitlines()
        if line.strip().startswith("ctx.declare_")
    ]
    assert declarations == [
        'ctx.declare_diagnostic("zeta");',
        'ctx.declare_diagnostic("alpha");',
        'ctx.declare_balance_route("%s");' % route.token,
        'ctx.declare_step_projection("embedded_error");',
    ]
    begin_step = source.index("ctx.begin_step(dt)")
    for declaration in declarations:
        assert source.index(declaration) < begin_step
    assert source.index('ctx.declare_diagnostic("zeta")') < source.index(
        'ctx.record_scalar("zeta",'
    )
    assert source.index('ctx.declare_balance_route("%s")' % route.token) < source.index(
        "ctx.record_balance_term("
    )
    assert source.index('ctx.declare_step_projection("embedded_error")') < source.index(
        'ctx.note_step_projection("embedded_error")'
    )


def test_dead_matrix_free_body_emits_no_transaction_authority_declaration():
    program = Program("dead_transaction_authority")
    temporal = typed_state(program, "fluid", state_name="U")
    operator = program.matrix_free_operator("unused")

    def apply(builder, _out, value):
        # Matrix-free apply blocks accept only nodes with an explicit stencil contract.  Build the
        # dead diagnostic with that exact pointwise evidence so this test reaches codegen liveness
        # instead of being rejected earlier by authoring validation.
        pointwise = StencilAccess.pointwise()
        reduced = builder._new(
            "scalar",
            "reduce",
            (value,),
            {"kind": "norm2", "comp": 0, "stencil_access": pointwise},
            "dead_apply_reduction",
            value.block,
        )
        builder._new(
            "scalar",
            "record_scalar",
            (reduced,),
            {"diagnostic": "dead_apply_diagnostic", "stencil_access": pointwise},
            "dead_apply_diagnostic",
            value.block,
        )
        return value

    program.set_apply(operator, apply)
    program.record_scalar("live_diagnostic", program.norm2(temporal.n))
    program.commit(
        temporal.next,
        program.value("next", temporal.n, at=temporal.next.point),
    )

    source = emit_cpp_program(program)
    assert source.count('ctx.declare_diagnostic("live_diagnostic")') == 1
    assert "dead_apply_diagnostic" not in source


def test_declaration_ledger_is_shared_across_regions_and_has_no_step_fallback():
    prelude = []
    outer = {("program_transaction_authority_declarations",): set()}
    left = dict(outer)
    right = dict(outer)
    _append_transaction_authority_declaration(
        prelude, left, kind="diagnostic", identity="shared"
    )
    _append_transaction_authority_declaration(
        prelude, right, kind="diagnostic", identity="shared"
    )
    assert prelude == ['ctx.declare_diagnostic("shared");']

    with pytest.raises(NotImplementedError, match="install-time prelude"):
        _append_transaction_authority_declaration(
            None, {}, kind="diagnostic", identity="late"
        )


def _linear_solve_program(action=FailRun()):
    """Build the smallest publishable prepared solve with an explicit outcome policy."""
    program = Program("solve_outcome_authority")
    temporal = typed_state(program, "fluid", state_name="U")
    operator = program.matrix_free_operator("identity")
    program.set_apply(operator, lambda _builder, _out, value: value)
    rhs = program.value("rhs", temporal.n, at=temporal.next.point)
    outcome = program.solve(
        LinearProblem(operator, rhs, nullspace=None),
        solver=Richardson(max_iter=2, rel_tol=1.0e-8),
    )
    solved = rhs if action is None else outcome.consume(action=action)
    program.commit(temporal.next, solved)
    return program


@pytest.mark.parametrize("status", SOLVE_STATUSES)
def test_generated_solve_maps_every_failure_category_through_one_explicit_policy(status):
    source = emit_cpp_program(_linear_solve_program(RejectAttempt(statuses=(status,))))
    expected = _SOLVE_STATUS_CPP[status]

    assert "pops::SolveOutcome kr" in source
    assert expected in source
    assert "pops::SolveConsumption::kRejectAttempt" in source
    assert "pops::SolveConsumption::kFailRun" in source
    assert "pops::SolveConsumption::kAccept" in source


def test_generated_solve_consumes_before_the_solution_can_reach_commit():
    source = emit_cpp_program(
        _linear_solve_program(RejectAttempt(statuses=("iteration_limit",)))
    )
    match = re.search(r"pops::SolveOutcome (kr\d+) =", source)
    assert match is not None
    outcome_name = match.group(1)
    outcome = match.start()
    failure_guard = source.index("if (!%s" % outcome_name, outcome)
    accept = source.index("(void)%s" % outcome_name, failure_guard)
    commit = source.index("ctx.commit", accept)

    assert outcome < failure_guard < accept < commit
    assert source.count("%s.consume(" % outcome_name) == 2  # mutually-exclusive failure/accept


def test_codegen_refuses_unconsumed_or_forged_double_solve_outcomes():
    with pytest.raises(ValueError, match="exactly one explicit outcome.consume"):
        emit_cpp_program(_linear_solve_program(action=None))

    program = _linear_solve_program()
    solve = next(value for value in program._values if value.op == "solve_linear")
    program._new(
        "state", "solve_outcome", (solve,), {"action": FailRun()}, "forged_second_outcome",
        solve.block, space=solve.space, point=solve.point,
    )
    with pytest.raises(ValueError, match="found 2"):
        emit_cpp_program(program)


def test_native_hierarchy_provider_must_only_produce_one_typed_outcome():
    _validate_native_solve_outcome_emission(
        ("pops::SolveOutcome kr17 = ctx.solve_hierarchy_tensor();",), "kr17"
    )
    with pytest.raises(ValueError, match="exactly one typed SolveOutcome"):
        _validate_native_solve_outcome_emission(("auto kr17 = ctx.solve();",), "kr17")
    with pytest.raises(ValueError, match="must not consume"):
        _validate_native_solve_outcome_emission(
            ("pops::SolveOutcome kr17 = ctx.solve();", "(void)kr17.consume();"), "kr17"
        )


def test_v5_refuses_opaque_rng_claim_before_resource_planning(monkeypatch):
    """An attribute cannot forge a transactional RNG service before artifact preparation."""
    from pops.codegen import program_codegen

    class OpaqueRngProvider:
        __pops_ir_immutable__ = True

    program = Program("opaque_rng_authority")
    temporal = typed_state(program, "fluid", state_name="U")
    program._new(
        "state", "deterministic_kernel", (temporal.n,), {"rng_provider": OpaqueRngProvider()},
        "opaque_rng", temporal.n.block, space=temporal.n.space, point=temporal.n.point,
    )
    before = tuple(program._values)
    monkeypatch.setattr(
        program_codegen,
        "get_program_resource_plan",
        lambda *_args, **_kwargs: pytest.fail("RNG refusal must precede resource planning"),
    )

    with pytest.raises(ValueError, match="claims an RNG provider.*ABI v5"):
        emit_cpp_program(program)
    assert tuple(program._values) == before
