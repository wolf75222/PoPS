"""Spec 3 scheduler cache CODEGEN (ADC-458): a held solve_fields lowers to the cache branch.

A `solve_fields` node carrying `Schedule(Every(...), off=Hold())` must codegen to
`if (ctx.cache_should_update(id, N)) { solve; ctx.cache_store_aux(id); } else { ctx.cache_restore_aux(id); }`
-- recompute + cache the System aux every N macro-steps, reuse it in between. This is the emit-level
check (the cache RUNTIME cadence runs in a compiled .so -- ROMEO; the CacheManager is unit-tested by
tests/cpp/integration/runtime/test_cache_manager.cpp). Other ops/policies still refuse to lower (not yet supported).
"""
import pytest

from pops import model
from pops.ir.expr import Var
from typed_program_support import typed_state

adctime = pytest.importorskip("pops.time")


def _every(clock, n, off):
    return adctime.Schedule(adctime.Every(adctime.AcceptedStep(clock), n), off=off)


def _at_start(clock, off):
    return adctime.Schedule(adctime.AtStart(adctime.AcceptedStep(clock)), off=off)


def _at_end(clock, off):
    return adctime.Schedule(adctime.AtEnd(adctime.AcceptedStep(clock)), off=off)


def _module():
    mod = model.Module("held_fields")
    u = mod.state_space("U", ("rho", "mx", "my"))
    fields = mod.field_space("fields", ("phi",))
    rho = Var("rho", "cons")
    mod.operator(name="fields_from_state", signature=(u,) >> fields, kind="field_operator", expr=rho)
    mod.operator_capabilities("fields_from_state", cacheable=True)
    return mod, u


def _held_program(schedule):
    mod, u = _module()
    P = adctime.Program("held").bind_operators(mod)
    schedule = schedule(P.clock) if callable(schedule) else schedule
    U = typed_state(P, "plasma", space=u)
    P._call("fields_from_state", U, schedule=schedule)
    endpoint = typed_state(P, "plasma", state_name="U", space=u).next
    P.commit(endpoint, P.value("U1", U, at=endpoint.point))
    return P


def test_held_solve_fields_lowers_and_emits_cache_branch():
    P = _held_program(lambda clock: _every(clock, 3, adctime.Hold()))
    P._check_schedules_lowerable()                # must NOT raise: solve_fields + hold is lowerable
    cpp = P.emit_cpp_program()
    assert "cache_should_update" in cpp, "due check emitted"
    assert "cache_store_aux" in cpp, "recompute branch stores the aux"
    assert "cache_restore_aux" in cpp, "held branch restores the cached aux"
    # the period N from every(3) reaches the due check
    assert "cache_should_update" in cpp and ", 3)" in cpp


def test_unscheduled_solve_fields_has_no_cache_branch():
    mod, u = _module()
    P = adctime.Program("plain").bind_operators(mod)
    U = typed_state(P, "plasma", space=u)
    P._call("fields_from_state", U)               # no schedule
    endpoint = typed_state(P, "plasma", state_name="U", space=u).next
    P.commit(endpoint, P.value("U1", U, at=endpoint.point))
    cpp = P.emit_cpp_program()
    assert "cache_should_update" not in cpp       # plain unconditional solve, no cache
    assert "ctx.solve_fields_from_state(" in cpp


def test_always_solve_fields_has_no_cache_branch():
    P = _held_program(lambda clock: adctime.Schedule(
        adctime.Always(adctime.AcceptedStep(clock))))
    cpp = P.emit_cpp_program()
    assert "cache_should_update" not in cpp       # always() == default cadence, no caching


def test_skip_policy_on_solve_fields_now_lowers():
    # ADC-458 scheduler codegen: skip on a field solve lowers (the op runs only when due; the aux keeps
    # its stale content off-cadence) -- it no longer refuses. See test_scheduler_codegen for the full
    # policy/kind matrix.
    P = _held_program(lambda clock: _every(clock, 5, adctime.Skip()))
    P._check_schedules_lowerable()                 # no raise
    cpp = P.emit_cpp_program()
    assert "skip: stale aux off-cadence" in cpp
    assert "cache_should_update" in cpp


def test_hold_on_non_every_kind_now_lowers():
    # ADC-458: a hold on on_start lowers to the macro_step()==0 due test; subcycle lowers to a sub-loop.
    # Only on_end still refuses (a compiled step loop has no end-of-run signal).
    cpp = _held_program(
        lambda clock: _at_start(clock, adctime.Hold())).emit_cpp_program()
    assert "(ctx.macro_step() == 0)" in cpp
    assert "cache_store_aux" in cpp
    with pytest.raises(NotImplementedError, match="ClockTick"):
        _held_program(lambda clock: adctime.Schedule(
            adctime.Always(adctime.ClockTick(clock))))._check_schedules_lowerable()
    with pytest.raises(NotImplementedError, match="AtEnd"):
        _held_program(
            lambda clock: _at_end(clock, adctime.Hold()))._check_schedules_lowerable()
