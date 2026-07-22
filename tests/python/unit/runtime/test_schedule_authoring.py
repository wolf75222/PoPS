"""Spec 3 unified-scheduler AUTHORING (ADC-458, epic ADC-450).

The schedule vocabulary, the policy chaining, recording a schedule on a Program node, the
cacheable-capability validation, and the honest refusal to lower a non-always schedule (the
runtime that honors caches / accumulate_dt / checkpoint is the C++ part of ADC-458). These are
pure-Python: only pops.time / pops.model are needed, no compiled step is run.
"""
from pops.codegen.program_codegen import _check_schedules_lowerable
import pytest

from pops import model
from pops.descriptors import Descriptor
from pops.fields import FieldDiscretization, FieldOperator
from pops._ir.expr import Var
from pops.math import ValueExpr, laplacian
from pops.problem import Case

adctime = pytest.importorskip("pops.time")


def _every(clock, n, off=None):
    return adctime.Schedule(adctime.Every(adctime.AcceptedStep(clock), n), off=off)


def _at_end(clock, off=None):
    return adctime.Schedule(adctime.AtEnd(adctime.AcceptedStep(clock)), off=off)


def _when(clock, condition, off=None):
    return adctime.Schedule(adctime.When(adctime.AcceptedStep(clock), condition), off=off)


def _module(cacheable=True):
    """A module with a cacheable field_operator and a (non-cacheable) flux grid_operator."""
    mod = model.Module("sched_demo")
    u = mod.state_space("U", ("rho", "mx", "my"))
    fields = mod.field_space("fields", ("phi",))
    rho = Var("rho", "cons")
    fields_from_state = mod.operator(
        name="fields_from_state", signature=(u,) >> fields,
        kind="field_operator", expr=rho)
    mod.operator(name="flux", signature=(u,) >> model.Rate(u), kind="grid_operator",
                 expr={"x": [rho, rho, rho], "y": [rho, rho, rho]})
    if cacheable:
        mod.operator_capabilities("fields_from_state", cacheable=True)
    return mod, u, fields, fields_from_state


class _FieldMethod(Descriptor):
    category = "field_method"

    def to_data(self):
        return {"type": "schedule-authoring-second-order"}


class _FieldSolver(Descriptor):
    category = "elliptic_solver"

    def to_data(self):
        return {"type": "schedule-authoring-krylov"}


def _program_state(mod, state, name="p"):
    """Return one Program state and its final Case-owned field solve authority."""
    state_handle = mod.state_handle(state)
    field_space = mod.field_spaces()["fields"]
    provider = mod.operator_handle("fields_from_state")
    case = Case(name="%s_case" % name)
    block = case.block("plasma", mod, states=(state_handle,))
    unknown = block[mod.field_handle(field_space)]
    operator = FieldOperator(
        "fields",
        unknown=unknown,
        equation=-laplacian(ValueExpr(unknown)) == ValueExpr(block[state_handle]),
        providers=provider,
    )
    field = case.field(
        operator,
        FieldDiscretization(
            method=_FieldMethod(), boundaries=(), solver=_FieldSolver()
        ),
    )
    program = adctime.Program(name)
    temporal = program.state(block[state_handle])
    return program, temporal.n, temporal, field


# --- schedule vocabulary -----------------------------------------------------
def test_always_is_default_recompute():
    s = adctime.Schedule(adctime.Always(adctime.AcceptedStep(adctime.Clock("macro"))))
    assert isinstance(s.trigger, adctime.Always) and s.off is None and s.is_always()


def test_every_carries_n_and_is_not_always():
    s = _every(adctime.Clock("macro"), 10)
    assert isinstance(s.trigger, adctime.Every) and s.trigger.n == 10 and not s.is_always()


def test_every_rejects_non_positive():
    with pytest.raises(ValueError):
        _every(adctime.Clock("macro"), 0)
    with pytest.raises(ValueError):
        _every(adctime.Clock("macro"), True)


def test_every_dt_is_an_exact_consumer_physical_time_cadence():
    clock = adctime.Clock("macro")
    schedule = adctime.every_dt(0.1, clock=clock)

    assert isinstance(schedule.trigger, adctime.EveryDt)
    assert schedule.trigger.interval == 0.1
    assert not schedule.is_always()
    assert schedule.to_data()["trigger"] == {
        "type": "every_dt",
        "interval": {"binary64": (0.1).hex()},
    }
    assert schedule.trigger.consumer_next_deadline(physical_time_hex=(0.0).hex()) == (0.1).hex()
    with pytest.raises(NotImplementedError, match="ConsumerGraph physical-time cadence"):
        schedule.native_schedule_ir(where="compiled Program test")


@pytest.mark.parametrize("interval", (0.0, -1.0, float("nan"), float("inf"), True, "0.1"))
def test_every_dt_rejects_non_positive_or_non_binary64_intervals(interval):
    with pytest.raises((TypeError, ValueError)):
        adctime.every_dt(interval, clock=adctime.Clock("macro"))


def test_every_dt_fails_when_interval_is_below_the_current_clock_resolution():
    schedule = adctime.every_dt(1.0e-308, clock=adctime.Clock("macro"))

    with pytest.raises(RuntimeError, match="below binary64 clock resolution"):
        schedule.trigger.consumer_next_deadline(physical_time_hex=(1.0e308).hex())


def test_other_kinds_exist():
    clock = adctime.Clock("macro")
    assert isinstance(_when(clock, lambda: True).trigger, adctime.When)
    assert isinstance(adctime.Schedule(
        adctime.AtStart(adctime.AcceptedStep(clock))).trigger, adctime.AtStart)
    assert isinstance(_at_end(clock).trigger, adctime.AtEnd)
    assert isinstance(adctime.ClockTick(clock), adctime.ClockTick)


# --- policy chaining ---------------------------------------------------------
def test_policy_chaining():
    clock = adctime.Clock("macro")
    assert isinstance(_every(clock, 10, adctime.Hold()).off, adctime.Hold)
    assert isinstance(_every(clock, 5, adctime.AccumulateDt()).off, adctime.AccumulateDt)
    assert isinstance(_at_end(clock, adctime.Zero()).off, adctime.Zero)
    assert isinstance(_every(clock, 2, adctime.Error()).off, adctime.Error)
    s = _every(clock, 7, adctime.Hold())
    assert s.trigger.n == 7


def test_schedule_repr_reads_like_the_api():
    clock = adctime.Clock("macro")
    assert "Every" in repr(_every(clock, 10, adctime.Hold()))
    assert "Always" in repr(adctime.Schedule(adctime.Always(adctime.AcceptedStep(clock))))


# --- operator_capabilities setter/getter -------------------------------------
def test_operator_capabilities_setter_then_getter():
    mod, _, _, _ = _module(cacheable=True)
    assert mod.operator_capabilities("fields_from_state")["cacheable"] is True
    # getter form is unchanged for an operator with no declared caps
    assert mod.operator_capabilities("flux").get("cacheable") is None


# --- recording a schedule on a node ------------------------------------------
def test_call_records_schedule_on_value():
    mod, u, _, fields_from_state = _module()
    P, U, _, fields_from_state = _program_state(mod, u)
    f = fields_from_state(U, schedule=_every(P.clock, 10, adctime.Hold()))
    assert isinstance(f._token.attrs["schedule"].off, adctime.Hold)
    assert "schedule" in P.dump_operator_ir()       # inspectable: recorded, not dropped


def test_call_without_schedule_is_unchanged():
    mod, u, _, fields_from_state = _module()
    P, U, _, fields_from_state = _program_state(mod, u)
    f = fields_from_state(U)
    assert "schedule" not in f._token.attrs


# --- cacheable validation (criterion 27) -------------------------------------
def test_hold_on_non_cacheable_operator_raises():
    mod, u, _, fields_from_state = _module(cacheable=False)
    P, U, _, fields_from_state = _program_state(mod, u)
    with pytest.raises(ValueError, match="not cacheable"):
        fields_from_state(U, schedule=_every(P.clock, 10, adctime.Hold()))


def test_accumulate_dt_on_non_cacheable_raises():
    mod, u, _, fields_from_state = _module(cacheable=False)
    P, U, _, fields_from_state = _program_state(mod, u)
    with pytest.raises(ValueError, match="not cacheable"):
        fields_from_state(U, schedule=_every(P.clock, 4, adctime.AccumulateDt()))


def test_hold_on_cacheable_operator_ok():
    mod, u, _, fields_from_state = _module(cacheable=True)
    P, U, _, fields_from_state = _program_state(mod, u)
    fields_from_state(U, schedule=_every(P.clock, 10, adctime.Hold()))


def test_skip_does_not_require_cacheable():
    # skip / recompute / zero produce nothing cached, so they do not require cacheable
    mod, u, _, fields_from_state = _module(cacheable=False)
    P, U, _, fields_from_state = _program_state(mod, u)
    fields_from_state(U, schedule=_every(P.clock, 10, adctime.Skip()))


# --- honesty gate: the two genuinely-unlowerable cases must fail loud, never silently no-op ---
# (ADC-458 codegen lowers every kind/policy EXCEPT on_end() -- no end-of-run signal in a compiled step
# loop -- and a when() over a Python callable. The full policy/kind matrix is in test_scheduler_codegen.)
def test_on_end_schedule_refuses_to_lower():
    mod, u, _, fields_from_state = _module(cacheable=True)
    P, U, _, fields_from_state = _program_state(mod, u)
    fields_from_state(U, schedule=_at_end(P.clock, adctime.Hold()))
    with pytest.raises(NotImplementedError, match="AtEnd"):
        _check_schedules_lowerable(P)


def test_when_python_callable_refuses_to_lower():
    mod, u, _, fields_from_state = _module(cacheable=True)
    P, U, _, fields_from_state = _program_state(mod, u)
    # a when() over a bare Python callable is not a Program value -> cannot lower
    fields_from_state(U, schedule=_when(P.clock, lambda: True, adctime.Hold()))
    serialized = P._serialize()
    callable_token = serialized["nodes"][-1]["attrs"]["schedule"]["trigger"]["payload"][
        "condition"]
    assert "unsupported_python_callable" in callable_token
    assert isinstance(P._ir_hash(), str)
    with pytest.raises(NotImplementedError, match="ADC-458"):
        _check_schedules_lowerable(P)


def test_held_solve_fields_now_lowers():
    # ADC-458 codegen: a held field solve lowers to the cache branch -- it must NOT raise (the runtime
    # cadence is exercised in the compiled .so / ROMEO).
    mod, u, _, fields_from_state = _module(cacheable=True)
    P, U, _, fields_from_state = _program_state(mod, u)
    fields_from_state(U, schedule=_every(P.clock, 10, adctime.Hold()))
    _check_schedules_lowerable(P)   # no raise


def test_skip_now_lowers():
    # ADC-458: skip on a field solve lowers (the op runs only when due; the aux is stale off-cadence).
    mod, u, _, fields_from_state = _module(cacheable=True)
    P, U, _, fields_from_state = _program_state(mod, u)
    fields_from_state(U, schedule=_every(P.clock, 10, adctime.Skip()))
    _check_schedules_lowerable(P)   # no raise


def test_always_schedule_lowers_fine():
    mod, u, _, fields_from_state = _module(cacheable=True)
    P, U, _, fields_from_state = _program_state(mod, u)
    fields_from_state(U, schedule=adctime.Schedule(
        adctime.Always(adctime.AcceptedStep(P.clock))))
    _check_schedules_lowerable(P)   # no raise: always() == the default cadence


def test_scheduled_node_serializes_for_codegen():
    # a Schedule object is not JSON-serializable; it must be reduced to its repr in the IR hash
    # (regression: an always()-scheduled node passed the gate then crashed _ir_hash with a TypeError).
    mod, u, _, fields_from_state = _module(cacheable=True)
    P, U, _, fields_from_state = _program_state(mod, u)
    fields_from_state(U, schedule=adctime.Schedule(
        adctime.Always(adctime.AcceptedStep(P.clock))))
    h = P._ir_hash()                 # must not raise
    assert isinstance(h, str) and h
    # the schedule is part of the IR identity: a different cadence yields a different hash
    P2, U2, _, second_field = _program_state(mod, u)
    second_field(U2, schedule=_every(P2.clock, 10, adctime.Skip()))
    assert P2._ir_hash() != h


def test_schedule_parameters_that_change_lowering_change_ir_identity():
    mod, u, _, fields_from_state = _module(cacheable=True)

    def build(domain_factory):
        program, state, temporal, field = _program_state(mod, u)
        field(state, schedule=adctime.Schedule(
            adctime.Always(domain_factory(program.clock))))
        final = program.value("final", state, at=temporal.next.point)
        program.commit(temporal.next, final)
        return program

    short = build(adctime.ClockTick)
    long = build(lambda clock: adctime.AMRLevel(clock, level=1))
    assert short._ir_hash() != long._ir_hash()

    first, first_state, _, first_field = _program_state(mod, u, "when_identity")
    first_cond = first.norm2(first_state) < 1
    _first_other = first.norm2(first_state) < 2
    first_field(first_state, schedule=_when(first.clock, first_cond))
    second, second_state, _, second_field = _program_state(mod, u, "when_identity")
    _second_other = second.norm2(second_state) < 1
    second_cond = second.norm2(second_state) < 2
    second_field(second_state, schedule=_when(second.clock, second_cond))
    assert first._ir_hash() != second._ir_hash()


def test_when_rejects_bool_value_from_another_program_even_with_colliding_ssa_id():
    mod, u, _, fields_from_state = _module(cacheable=True)
    owner, owner_state, _, owner_field = _program_state(mod, u, "owner")
    foreign, foreign_state, _, _ = _program_state(mod, u, "foreign")
    foreign_cond = foreign.norm2(foreign_state) > 0

    with pytest.raises(ValueError, match="different Program"):
        owner_field(owner_state, schedule=_when(owner.clock, foreign_cond))
