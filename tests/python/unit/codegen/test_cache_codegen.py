"""Spec 3 scheduler cache CODEGEN (ADC-458): a held solve_fields lowers to the cache branch.

A `solve_fields` node carrying `Schedule(Every(...), off=Hold())` must codegen to
`if (ctx.schedule_is_due(id, N, domain)) { solve; ctx.cache_store_aux(id); } else { ctx.cache_restore_aux(id); }`
-- recompute + cache the System aux every N macro-steps, reuse it in between. This is the emit-level
check (the cache RUNTIME cadence runs in a compiled .so -- ROMEO; the CacheManager is unit-tested by
tests/cpp/integration/runtime/test_cache_manager.cpp). Other ops/policies still refuse to lower (not yet supported).
"""
from types import SimpleNamespace

import pytest

from pops import model
from pops.codegen.program_codegen import emit_cpp_program
from pops._ir.expr import Var
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
    fields_from_state = mod.operator(
        name="fields_from_state", signature=(u,) >> fields, kind="field_operator", expr=rho
    )
    mod.operator_capabilities("fields_from_state", cacheable=True)
    return mod, u, fields_from_state


def _held_program(schedule):
    mod, u, fields_from_state = _module()
    P = adctime.Program("held")._bind_operators(mod)
    schedule = schedule(P.clock) if callable(schedule) else schedule
    state = mod.state_handle(u)
    U = typed_state(P, "plasma", space=u, model=mod, state=state)
    fields_from_state(U, schedule=schedule).consume(action=adctime.FailRun())
    endpoint = typed_state(
        P, "plasma", state_name="U", space=u, model=mod, state=state
    ).next
    P.commit(endpoint, P.value("U1", U, at=endpoint.point))
    return P, mod


def _emit(P, mod):
    """Supply the exact field-provider route that final operator-handle lowering requires."""
    fields_from_state = mod.operator_handle("fields_from_state")
    field_plans = {
        "fields": SimpleNamespace(
            rhs_providers=(fields_from_state,),
            native_options={
                "provider_slot": "fields",
                "boundary_kernel_required": False,
            },
        )
    }
    return emit_cpp_program(P, model=mod.to_dsl(), field_plans=field_plans)


def test_held_solve_fields_lowers_and_emits_cache_branch():
    P, mod = _held_program(lambda clock: _every(clock, 3, adctime.Hold()))
    P._check_schedules_lowerable()                # must NOT raise: solve_fields + hold is lowerable
    cpp = _emit(P, mod)
    assert "schedule_is_due" in cpp, "due check emitted"
    assert "cache_store_aux" in cpp, "recompute branch stores the aux"
    assert "cache_restore_aux" in cpp, "held branch restores the cached aux"
    # the period N from every(3) reaches the due check
    assert "schedule_is_due" in cpp and ", 3," in cpp


def test_unscheduled_solve_fields_has_no_cache_branch():
    mod, u, fields_from_state = _module()
    P = adctime.Program("plain")._bind_operators(mod)
    state = mod.state_handle(u)
    U = typed_state(P, "plasma", space=u, model=mod, state=state)
    fields_from_state(U).consume(action=adctime.FailRun())  # no schedule
    endpoint = typed_state(
        P, "plasma", state_name="U", space=u, model=mod, state=state
    ).next
    P.commit(endpoint, P.value("U1", U, at=endpoint.point))
    cpp = _emit(P, mod)
    assert "cache_should_update" not in cpp       # plain unconditional solve, no cache
    assert "ctx.solve_fields_from_state(" in cpp


def test_always_solve_fields_has_no_cache_branch():
    P, mod = _held_program(lambda clock: adctime.Schedule(
        adctime.Always(adctime.AcceptedStep(clock))))
    cpp = _emit(P, mod)
    assert "cache_should_update" not in cpp       # always() == default cadence, no caching


def test_skip_policy_on_solve_fields_now_lowers():
    # ADC-458 scheduler codegen: skip on a field solve lowers (the op runs only when due; the aux keeps
    # its stale content off-cadence) -- it no longer refuses. See test_scheduler_codegen for the full
    # policy/kind matrix.
    P, mod = _held_program(lambda clock: _every(clock, 5, adctime.Skip()))
    P._check_schedules_lowerable()                 # no raise
    cpp = _emit(P, mod)
    assert "skip: stale aux off-cadence" in cpp
    assert "schedule_is_due" in cpp


def test_hold_on_non_every_kind_now_lowers():
    # ADC-677: on_start and clock-tick are qualified by their explicit runtime domains.
    # Only on_end still refuses (a compiled step loop has no end-of-run signal).
    P, mod = _held_program(lambda clock: _at_start(clock, adctime.Hold()))
    cpp = _emit(P, mod)
    assert "ctx.schedule_at_start(" in cpp
    assert "cache_store_aux" in cpp
    P, mod = _held_program(lambda clock: adctime.Schedule(
        adctime.Always(adctime.ClockTick(clock))))
    cpp = _emit(P, mod)
    assert "ctx.schedule_domain_occurs(" in cpp
    assert "ScheduleDomainKind::kClockTick" in cpp
    with pytest.raises(NotImplementedError, match="AtEnd"):
        P, _ = _held_program(lambda clock: _at_end(clock, adctime.Hold()))
        P._check_schedules_lowerable()
