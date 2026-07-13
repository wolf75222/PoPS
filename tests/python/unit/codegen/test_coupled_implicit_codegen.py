"""Fail-closed generic multi-block implicit Program lowering (ADC-690)."""
from __future__ import annotations

import pytest

from typed_program_support import typed_state

time = pytest.importorskip("pops.time")
from pops import model  # noqa: E402
from pops.ir.expr import Var  # noqa: E402


def _program(*, consume=True):
    module = model.Module("implicit_exchange")
    electrons = module.state_space("electron_state", ("ne", "pex", "pey"))
    ions = module.state_space("ion_state", ("ni", "pix", "piy"))
    bundle = model.RateBundle({
        "electrons": model.Rate(electrons),
        "ions": model.Rate(ions),
    })
    ne, pex, pey = (Var(name, "cons") for name in electrons.components)
    ni, pix, piy = (Var(name, "cons") for name in ions.components)
    collision = module.operator(
        name="collision",
        signature=model.Signature((electrons, ions), bundle),
        kind="coupled_rate",
        expr={
            "electrons": [ni - ne, pix - pex, piy - pey],
            "ions": [ne - ni, pex - pix, pey - piy],
        },
    )
    program = time.Program("implicit_collision")
    electron_handle = module.state_handle(electrons)
    ion_handle = module.state_handle(ions)
    electron_n = typed_state(
        program, "electrons", space=electrons, model=module, state=electron_handle)
    ion_n = typed_state(program, "ions", space=ions, model=module, state=ion_handle)
    outcome = program.solve_implicit(
        collision, (electron_n, ion_n), tol=1.0e-11, max_iter=12, fd_eps=1.0e-6,
        name="collision_step")
    electron_next = typed_state(
        program, "electrons", state_name=electrons.name, space=electrons,
        model=module, state=electron_handle).next
    ion_next = typed_state(
        program, "ions", state_name=ions.name, space=ions,
        model=module, state=ion_handle).next
    if not consume:
        program.commit_many({
            electron_next: program.value("electron_identity", electron_n, at=electron_next.point),
            ion_next: program.value("ion_identity", ion_n, at=ion_next.point),
        })
        return module, program, outcome
    solved = outcome.consume(action=time.RejectAttempt())
    program.commit_many({
        electron_next: solved[electron_n.block],
        ion_next: solved[ion_n.block],
    })
    return module, program, solved


def test_coupled_implicit_is_one_native_newton_kernel_with_explicit_action():
    _module, program, solved = _program()
    source = program.emit_cpp_program(model=None)

    assert len(solved) == 2
    assert source.count("pops::for_each_cell") == 1
    assert "pops::detail::mat_inverse<6>" in source
    assert "Ueval[0] - G_[0] - dt *" in source
    assert "pops::reduce_max(ci_status_" in source
    assert "SolveStatus::kSingular" in source
    assert "StepAttemptRejected" in source
    assert source.count("ctx.lincomb(") >= 2


def test_unconsumed_coupled_implicit_is_rejected_by_graph_validation_and_codegen():
    _module, program, outcome = _program(consume=False)
    with pytest.raises(ValueError, match="consumed exactly once"):
        program.to_graph()
    with pytest.raises(ValueError, match="must have exactly one explicit"):
        program.emit_cpp_program(model=None)
    with pytest.raises(TypeError, match="not readable"):
        _ = outcome.token


def test_failed_coupled_implicit_never_aliases_live_state_before_guard():
    _module, program, _solved = _program()
    source = program.emit_cpp_program(model=None)
    guard = source.index("if (!ci_report_")
    first_commit = source.index("ctx.lincomb(", guard)

    assert "ctx.scratch_state_like(u0)" in source
    assert "ctx.scratch_state_like(u1)" in source
    assert guard < first_commit
