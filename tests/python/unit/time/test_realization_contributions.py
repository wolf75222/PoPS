"""Real IR/code-generation contracts, not a substitute numerical runtime."""
from fractions import Fraction as F
from functools import cache

import pytest

from pops.contributions import Contribution, Delivery, Increment, bdf2_increment, combine
from pops.frames import Cartesian2D
from pops.math import ddt
from pops.physics import Model
from pops.realization import Realize, Use, require_coverage, validate_realized_outputs
from pops.time import Program
from pops.time.expressions import ProgramExpression
from typed_program_support import state_refs


def fixture():
    model = Model("realized_equations", frame=Cartesian2D())
    state = model.state("U", components=("x", "y"))
    source = model.source("transfer", on=state,
                          value=(-state[0] * state[1], state[0] * state[1]))
    balance = model.rate("equation", equation=ddt(state) == source)
    p = Program("realized_program")
    block, _ = state_refs(p, "material", model=model, state=state)
    q = p.state(block[state])
    use = Use(balance.occurrences[0], "cell", "current_stage", "inventory")
    return model, p, q, use


def emit(model, program):
    from pops.codegen.module_lowering import lower_and_validate
    from pops.codegen.program_codegen import emit_cpp_program
    lowered, _ = lower_and_validate(model, facade=model)
    return emit_cpp_program(program, model=lowered)


@cache
def interval(p):
    return p.interval()


def contribution(p, base, name, components, *, measure="increment", kind="exchange"):
    value = p.value(name, ProgramExpression(components, base))
    return Contribution(name, value, interval(p), measure, kind)


def identity_delivery(c, base):
    return Delivery(c, base, ((1, 0), (0, 1)))


def test_realized_formula_is_used_by_update_and_emitted_as_cpp():
    model, p, q, use = fixture()
    recipe = Realize("transfer", owns=(use,), bindings={"q": q.n},
                     body=lambda p, q: {"J": (q[0] * q[1], q[0] * q[1])},
                     outputs={use: ("J",)})
    result = recipe.evaluate(p)
    shared = Contribution("single_transfer", result[use], interval(p))
    inc = Increment((shared,), (Delivery(shared, q.n, ((-1, 0), (0, 1))),))
    after = inc.apply(p, q.n, name="accepted", at=q.next.point)
    inc.require_application(after)
    p.commit(q.next, after)
    assert require_coverage((use,), (recipe,))
    assert validate_realized_outputs(p)
    assert p.validate()
    source = emit(model, p)
    assert "pops::for_each_cell" in source
    assert "ctx.scratch_state" in source
    assert any("realization_use_v02" in row.get("attrs", {}) for row in p.ir_nodes())


def test_reading_collision_does_not_own_its_volume_delivery():
    _, p, q, use = fixture()
    face = Use(use.occurrence, "face", "interval", "inventory")
    recipe = Realize("face", owns=(face,), reads=(use,), bindings={"q": q.n},
                     body=lambda p, q: (q[0], q[1]), outputs={face: ()})
    assert require_coverage((face,), (recipe,))
    with pytest.raises(ValueError, match="missing or extra"):
        require_coverage((face, use), (recipe,))
    with pytest.raises(ValueError, match="multiple owning"):
        require_coverage((face,), (recipe, recipe))


def test_output_paths_and_consumption_are_required():
    _, p, q, use = fixture()
    with pytest.raises(ValueError, match="cover"):
        Realize("missing", owns=(use,), bindings={}, body=lambda p: {}, outputs={})
    recipe = Realize("dead", owns=(use,), bindings={"q": q.n},
                     body=lambda p, q: (q[0] ** 2, q[1]), outputs={use: ()})
    recipe.evaluate(p)
    p.commit(q.next, p.value("unchanged", q.n, at=q.next.point))
    with pytest.raises(ValueError, match="not consumed"):
        validate_realized_outputs(p)


def test_hidden_temporal_capture_rolls_back_authoring():
    _, p, q, use = fixture()
    hidden = p.value("hidden", q.n * q.n)
    before = p._ir_hash()
    recipe = Realize("hidden_capture", owns=(use,), bindings={"q": q.n},
                     body=lambda p, q: (hidden[0], q[1]), outputs={use: ()})
    with pytest.raises(ValueError, match="undeclared temporal"):
        recipe.evaluate(p)
    assert p._ir_hash() == before


def test_hidden_capture_inside_native_region_is_also_refused():
    _, p, q, use = fixture()
    hidden = p.value("hidden", q.n * q.n)
    def body(builder, q):
        return builder.fold(q, 2, lambda inner, carried, index:
                            inner.value("read_hidden_in_fold", carried + hidden))
    recipe = Realize("hidden_region_capture", owns=(use,), bindings={"q": q.n},
                     body=body, outputs={use: ()})
    before = p._ir_hash()
    with pytest.raises(ValueError, match="undeclared temporal"):
        recipe.evaluate(p)
    assert p._ir_hash() == before


def test_realized_native_region_keeps_declared_bindings_and_consumption():
    model, p, q, use = fixture()
    def body(builder, q):
        return builder.fold(q, 2, lambda inner, carried, index:
                            inner.value("squared_in_fold", carried * carried))
    recipe = Realize("native_region", owns=(use,), bindings={"q": q.n},
                     body=body, outputs={use: ()})
    value = recipe.evaluate(p)[use]
    p.commit(q.next, p.value("endpoint", value, at=q.next.point))
    assert p.validate() and validate_realized_outputs(p)
    assert "for (" in emit(model, p)


def test_explicit_constant_support_has_no_false_zero_times_state():
    model, p, q, use = fixture()
    recipe = Realize("constant", owns=(use,), bindings={"q": q.n},
                     body=lambda p, q: ProgramExpression((F(2), F(3)), q),
                     outputs={use: ()})
    value = recipe.evaluate(p)[use]
    assert value.attrs["support_only_inputs"] == (0,)
    p.commit(q.next, p.value("next", q.n + value, at=q.next.point))
    assert validate_realized_outputs(p)
    source = emit(model, p)
    assert "pops::for_each_cell" in source


@pytest.mark.parametrize("consumer", ("constant_state", "constant_rate", "foreign_expression_rate"))
def test_geometric_template_does_not_consume_realized_output(consumer):
    _, p, q, use = fixture()
    value = Realize("unconsumed", owns=(use,), bindings={"q": q.n},
        body=lambda p, q: (q[0]**2, q[1]**2), outputs={use: ()}).evaluate(p)[use]
    if consumer == "constant_state":
        unrelated = p.value("constant", ProgramExpression((0., 0.), value))
    elif consumer == "constant_rate":
        unrelated = p.rate("constant", (0., 0.), of=value)
    else:
        unrelated = p.rate("other_input", (q.n[0], q.n[1]), of=value)
    template_index = next(i for i, item in enumerate(unrelated.inputs) if item is value)
    assert template_index in unrelated.attrs["support_only_inputs"]
    p.commit(q.next, p.value("accepted", q.n+p.dt*unrelated, at=q.next.point))
    with pytest.raises(ValueError, match="not consumed"):
        validate_realized_outputs(p)


def test_realize_binding_metadata_is_not_consumption():
    _, p, q, use = fixture()
    first = Realize("first", owns=(use,), bindings={"q": q.n},
        body=lambda p, q: (q[0]**2, q[1]), outputs={use: ()}).evaluate(p)[use]
    second_use = Use(use.occurrence, "cell", "second_stage", "inventory")
    second = Realize("second", owns=(second_use,), bindings={"q": q.n, "unused": first},
        body=lambda p, q, unused: (q[0], q[1]), outputs={second_use: ()}).evaluate(p)[second_use]
    p.commit(q.next, p.value("accepted", second, at=q.next.point))
    with pytest.raises(ValueError, match="not consumed"):
        validate_realized_outputs(p)


def test_replacement_uses_new_value_once_and_rejects_old_application():
    model, p, q, _ = fixture()
    old = contribution(p, q.n, "old_interval", (F(1, 2), F(1, 2)))
    inc = Increment((old,), (Delivery(old, q.n, ((-1, 0), (0, 1))),))
    stale = inc.apply(p, q.n, name="stale")
    new = contribution(p, q.n, "new_interval", (F(3, 4), F(3, 4)))
    corrected = inc.replace(old, new)
    with pytest.raises(ValueError, match="exact increment version"):
        corrected.require_application(stale)
    after = corrected.apply(p, q.n, name="corrected", at=q.next.point)
    corrected.require_application(after)
    assert all(d.contribution is new for d in corrected.deliveries)
    assert len(corrected.contributions) == 1
    p.commit(q.next, after)
    assert p.validate()
    assert "pops::for_each_cell" in emit(model, p)


def test_duplicate_delivery_and_unintegrated_rate_are_refused():
    _, p, q, _ = fixture()
    rate = contribution(p, q.n, "rate", (q.n[0], q.n[1]), measure="rate")
    delivery = identity_delivery(rate, q.n)
    with pytest.raises(ValueError, match="duplicate delivery"):
        Increment((rate,), (delivery, delivery))
    inc = Increment((rate,), (delivery,))
    with pytest.raises(ValueError, match="rate without"):
        inc.apply(p, q.n, name="wrong_units")
    with pytest.raises(ValueError, match="inserted twice"):
        combine(inc, inc)


def test_historical_contribution_cannot_fake_its_original_interval():
    _, p, q, _ = fixture()
    base = q.n
    old_value = p.history("accepted_increment", lag=1, space=base.space,
                          block=base.block, state_ref=base.state_ref)
    with pytest.raises(TypeError, match="issued runtime Interval"):
        Contribution("previous", old_value, (-1, 0))
    with pytest.raises(NotImplementedError, match="HistorySampleIdentity"):
        Contribution("previous", old_value, interval(p))


def test_interval_endpoints_and_attempt_guard_are_real_native_dependencies():
    model,p,q,_=fixture()
    window=p.interval()
    shared=Contribution("J",p.value("J",(q.n[0],q.n[1])),window)
    inc=Increment((shared,),(identity_delivery(shared,q.n),))
    result=inc.apply(p,q.n,name="delivered",at=q.next.point)
    p.commit(q.next,result)
    source=emit(model,p)
    assert window.end.inputs==(window.begin,window.duration)
    assert "ctx.resource_attempt()" in source
    assert ".same_attempt(ctx.resource_attempt())" in source
    assert "ctx.boundary_evaluation_point(0)" in source
    assert "interval_checked_" in source and "pops::all_reduce_max" in source
    assert any(v.op=="interval_check" and v.inputs==(window.begin,window.duration,window.end)
               for v in p._values)


def test_requested_interval_cannot_be_published_as_reached():
    _,p,q,_=fixture()
    window=p.interval(role="requested")
    item=Contribution("J",p.value("J",(q.n[0],q.n[1])),window)
    inc=Increment((item,),(identity_delivery(item,q.n),))
    with pytest.raises(ValueError,match="reached Interval"):
        inc.apply(p,q.n,name="wrong_frontier")


def test_interval_replace_and_scale_cannot_relabel_runtime_endpoints():
    _,p,q,_=fixture()
    first=p.interval(); second=p.interval(duration=p.requested_dt()*2)
    old=Contribution("old",p.value("old",q.n),first)
    new=Contribution("new",p.value("new",q.n),second)
    inc=Increment((old,),(identity_delivery(old,q.n),))
    with pytest.raises(ValueError,match="interval"):
        inc.replace(old,new)
    with pytest.raises(NotImplementedError,match="relabeling"):
        inc.scale(p,1,name="relabel",interval=second)


def test_interval_child_capture_and_amr_lowering_are_refused():
    model,p,q,_=fixture()
    window=p.interval()
    result=p.fold(q.n,1,lambda inner,carried,index:window.bind(inner,carried,name="bad"))
    p.commit(q.next,p.value("candidate",result,at=q.next.point))
    with pytest.raises((NotImplementedError,ValueError),match="effect|Interval|qualified|unsupported"):
        p.validate()
    model,p,q,_=fixture()
    window=p.interval()
    shared=Contribution("J",p.value("J",q.n),window)
    inc=Increment((shared,),(identity_delivery(shared,q.n),))
    p.commit(q.next,inc.apply(p,q.n,name="root",at=q.next.point))
    from pops.codegen.program_emit_intervals import emit_interval
    with pytest.raises(NotImplementedError,match="Uniform"):
        emit_interval(p,window.begin,{},[],target="amr_system")


def test_computed_interval_requires_exact_published_duration_ssa():
    model,p,q,_=fixture()
    from pops.time import ComputedDt
    duration=p.requested_dt();window=p.interval(duration=duration)
    item=Contribution("J",p.value("J",q.n),window)
    inc=Increment((item,),(identity_delivery(item,q.n),))
    p.commit(q.next,inc.apply(p,q.n,name="delivered",at=q.next.point))
    p.reached_duration(duration*.5);p.step_strategy(ComputedDt(.1))
    with pytest.raises(ValueError,match="exact ComputedDt frontier SSA"):
        emit(model,p)


def test_history_interval_weights_the_exact_retained_publication_in_branch():
    model,p,q,_=fixture()
    current=p.interval()
    p.store_history("accepted_increment",q.n,depth=1)
    def arm(b):
        retained=b.history("accepted_increment",lag=1,space=q.n.space,
                           block=q.n.block,state_ref=q.n.state_ref)
        historical=b.history_interval(retained)
        previous=Contribution("previous",retained,historical)
        old=Increment((previous,),(identity_delivery(previous,q.n),))
        with pytest.raises(ValueError,match="explicit weight"):
            old.apply(b,q.n,name="wrong_window")
        weighted=old.scale(b,F(1,3),name="weighted",interval=current)
        return weighted.apply(b,q.n,name="updated",at=q.next.point)
    value=p.branch(p.max(q.n)>0,arm,lambda b:b.value("startup",q.n,at=q.next.point),name="startup")
    p.commit(q.next,value)
    assert p.validate()
    source=emit(model,p)
    assert "HistorySampleKind::Publication" in source
    assert "interaction_history_selected" in source
    assert "history_interval_snapshot_" in source
    assert ".same_attempt(ctx.resource_attempt())" in source
    assert "std::bit_cast<double>(history_interval_sample_" in source
    assert "std::bit_cast<pops::Real>" not in source
    assert source.index("HistoryInterval publication exceeds Real range") < source.index(
        "static_cast<pops::Real>(history_interval_begin64_")
    assert "std::bit_cast<std::uint64_t>(static_cast<double>(history_interval_begin_" in source
    assert "std::bit_cast<std::uint64_t>(static_cast<double>(history_interval_duration_" in source
    assert "HistoryInterval publication is not exactly representable in Real" in source
    assert "ctx.sys_block(0)" in source
    assert "HistoryInterval read failed on another rank" in source
    assert "HistoryInterval publication differs across ranks" in source


@pytest.mark.parametrize("missing",(True,False))
def test_history_negative_programs_resolve_and_emit_real_checks(missing):
    import pops
    from tests.python.integration.runtime.test_api_v02_intervals_native import _build_history
    from pops.codegen.program_codegen import emit_cpp_program
    from pops.codegen.program_models import ProgramModelGraph
    spec=_build_history(missing=missing)
    plan=pops.resolve(pops.validate(spec.case),layout=spec.layout)
    source=emit_cpp_program(plan.time,model_graph=ProgramModelGraph.from_resolved_blocks(plan.blocks))
    assert "HistoryInterval read failed on another rank" in source
    assert "HistoryInterval publication differs across ranks" in source
    assert "history_interval_snapshot_" in source
    assert "weighted_past_origin_0_history_origin" in source


def test_history_interval_cannot_be_attached_to_another_carrier():
    _,p,q,_=fixture()
    retained=p.history("accepted_increment",lag=1,space=q.n.space,
                       block=q.n.block,state_ref=q.n.state_ref)
    historical=p.history_interval(retained)
    copied=p.value("copied",retained)
    with pytest.raises(ValueError,match="exact retained carrier"):
        Contribution("borrowed",copied,historical)
    with pytest.raises(ValueError,match="exact retained numerical carrier"):
        historical.bind(p,copied,name="borrowed")
    with pytest.raises(TypeError,match="issued full-state history"):
        p.history_interval(copied)


@pytest.mark.parametrize("step", (F(-1,2), F(0), -1., 0, True))
def test_bdf2_rejects_nonpositive_exact_step_before_authoring(step):
    _, p, _, _ = fixture()
    empty = Increment()
    before = p._ir_hash()
    with pytest.raises(ValueError, match="positive"):
        bdf2_increment(p, empty, empty, h=step, h_previous=F(1,2),
                       solve_residual=empty, interval=(0,1))
    assert p._ir_hash() == before


def delivery_case(*, second_units=None):
    import pops
    from pops.model import PhysicalDimension
    from pops.numerics import DiscretizationPlan, StateStorage
    from examples.api_v02.models_M01_M14 import _domain
    frame, layout = _domain("co_located_delivery", (4, 4))
    energy = PhysicalDimension((("energy", 1),))
    momentum = PhysicalDimension((("momentum", 1),))
    mass = PhysicalDimension((("mass", 1),))
    first, second = Model("donor", frame=frame), Model("receiver", frame=frame)
    left = first.state("Q", components=("E", "P"), units=(energy, momentum))
    right = second.state("Q", components=("stored_mass", "stored_momentum", "stored_energy"),
                         units=(mass, momentum, energy) if second_units is None else second_units)
    case = pops.Case("explicit_component_delivery")
    a, b = case.block("donor", first), case.block("receiver", second)
    for block, state in ((a, left), (b, right)):
        numerics = DiscretizationPlan()
        numerics.states.add(state, StateStorage())
        case.numerics(numerics, block=block)
    p = Program("shared_inventory_transfer")
    return case, layout, p, p.state(a[left]), p.state(b[right])


def test_delivery_between_blocks_with_different_components_uses_explicit_finite_map():
    import pops
    from pops.codegen.program_codegen import emit_cpp_program
    from pops.codegen.program_models import ProgramModelGraph
    case, layout, p, left, right = delivery_case()
    shared = contribution(p, left.n, "one_exchange", (F(1, 10), F(1, 5)))
    inc = Increment((shared,), (
        Delivery(shared, left.n, ((-1, 0), (0, -1))),
        Delivery(shared, right.n, ((0, 0), (0, 1), (1, 0))),
    ))
    a = inc.apply(p, left.n, name="donor_after", at=left.next.point)
    b = inc.apply(p, right.n, name="receiver_after", at=right.next.point)
    assert a.state_ref == left.n.state_ref and b.state_ref == right.n.state_ref
    assert a.space.components == ("E", "P")
    assert b.space.components == ("stored_mass", "stored_momentum", "stored_energy")
    assert any(value is shared.value for value in a.inputs[0].inputs)
    assert any(value is shared.value for value in b.inputs[0].inputs)
    assert a.inputs[0].attrs["finite_support_v1"][1] == a.space.components
    assert b.inputs[0].attrs["finite_support_v1"][1] == b.space.components
    inc.require_application(a); inc.require_application(b)
    p.commit_many({left.next: a, right.next: b})
    p.step_strategy(pops.time.FixedDt(.1)); case.program(p)
    resolved = pops.resolve(pops.validate(case), layout=layout)
    cpp = emit_cpp_program(resolved.time,
                          model_graph=ProgramModelGraph.from_resolved_blocks(resolved.blocks))
    assert "pops::for_each_cell" in cpp
    # The metadata check is supplemented by an actual native layout/rank check
    # before the generated cross-block cell loop; this test does not run it.
    assert "finite materialization requires co-located layouts, distributions and ranks" in cpp


@pytest.mark.parametrize("unknown", (False, True))
def test_delivery_rejects_bad_or_unresolved_component_units(unknown):
    from pops.model import PhysicalDimension
    one = PhysicalDimension()
    case, layout, p, left, right = delivery_case(second_units=(one, one, None if unknown else one))
    shared = contribution(p, left.n, "exchange", (1., 2.))
    recipient = right.n
    before = p._ir_hash()
    with pytest.raises(ValueError, match="incompatible component units"):
        Delivery(shared, recipient, ((0, 0), (0, 0), (1, 0)))
    assert p._ir_hash() == before


def test_delivery_rejects_a_different_declared_layout_before_materialization():
    import pops
    from pops.model import Module
    first, second = Module("cell_quantity"), Module("face_quantity")
    sa = first.state_space("Q", ("u",), layout="cell")
    sb = second.state_space("Q", ("v",), layout="face")
    case = pops.Case("incompatible_delivery_supports")
    a, b = case.block("cell", first), case.block("face", second)
    p = Program("refuse_implicit_interpolation")
    left = p.state(a[first.state_handle(sa)])
    right = p.state(b[second.state_handle(sb)])
    shared = contribution(p, left.n, "exchange", (1.,))
    recipient = right.n
    before = p._ir_hash()
    with pytest.raises(ValueError, match="co-location obligation: different layout"):
        Delivery(shared, recipient, ((1,),))
    assert p._ir_hash() == before
