"""G03 lot M1b (ROM-21): the rate of a total on a moving configuration, source-only checks.

Note §7.2, §7.6: ``moving_total_rate`` reads the totals, the members and the members' rate of one
stage; every other balance on a moving configuration is refused by name: an average (write
``represent={U: pops.CellTotal()}``, or lot A), a static face pass, a face other than first-order
Rusanov, and a step rule (lot T).
"""
from __future__ import annotations

import numpy as np
import pytest

import pops
from pops.codegen._cell_total_consumers import CellTotalNotYetElaborated
from pops.codegen._moving_members_consumers import MovingMembersNotYetElaborated
from pops.codegen.program_lowerability import all_ops
from pops.time import AdaptiveCFL
from pops.time._program.moving_total_rate import OPERATION, moving_total_rate

from examples.api_v02.models_G03_moving_transport import (
    CELLS, CONSTANT, build_transport, initial_density)


def test_each_stage_reads_the_totals_the_members_and_their_rate_of_that_stage():
    case, layout, values, configuration = build_transport()
    plan = pops.resolve(pops.validate(case), layout=layout)
    assert [block.name for block in plan.blocks] == ["configuration", "line"]
    rates = [value for value in all_ops(plan.time) if value.op == OPERATION]
    assert len(rates) == 3
    for value in rates:
        state, members, member_rate = value.inputs
        assert state.block.local_id == "line" and members.block.local_id == "configuration"
        assert state.point == members.point == value.point
        source = member_rate
        while source.op == "linear_combine":
            (source,) = source.inputs
        assert source.op == "source" and source.inputs[0].id == members.id
    start = configuration.initial_members()
    (totals,) = [array for key, array in values.items() if array.shape[0] == 1]
    np.testing.assert_array_equal(
        totals[0], initial_density(0.5*(start[0] + start[1]), "wave")*start[2])


def test_a_constant_density_is_bound_as_u0_times_the_volume_member():
    _, _, values, configuration = build_transport(initial="constant")
    (totals,) = [array for array in values.values() if array.shape[0] == 1]
    assert totals.shape == (1, CELLS)
    assert totals.tobytes() == (CONSTANT*configuration.initial_members()[2])[None].tobytes()


def test_a_member_rate_of_another_stage_is_refused():
    case, _, _, configuration = build_transport()
    program = case._time
    rates = [value for value in all_ops(program) if value.op == OPERATION]
    first, second = rates[0], rates[1]
    with pytest.raises(ValueError, match="same stage"):
        moving_total_rate(program, "mixed", second.inputs[0], second.inputs[1],
                          first.inputs[2], rate=first.attrs["operator_handle"])


def test_an_average_on_a_moving_configuration_is_refused_naming_cell_total_and_lot_a():
    case, layout, _, _ = build_transport(represent="average")
    with pytest.raises(MovingMembersNotYetElaborated, match=r"pops\.CellTotal\(\)") as refused:
        pops.resolve(pops.validate(case), layout=layout)
    assert refused.value.reason == "moving_balance_not_total"
    assert "G03 lot A" in str(refused.value)


def test_a_static_face_pass_on_a_moving_configuration_is_refused():
    case, layout, _, _ = build_transport(static_rate=True)
    with pytest.raises(MovingMembersNotYetElaborated, match="spacing of the base") as refused:
        pops.resolve(pops.validate(case), layout=layout)
    assert refused.value.reason == "moving_static_face"
    assert "MovingMembers.transport" in str(refused.value)


def test_a_face_other_than_first_order_rusanov_is_refused():
    case, layout, _, _ = build_transport(face="hll")
    with pytest.raises(MovingMembersNotYetElaborated, match="FirstOrder and Rusanov") as refused:
        pops.resolve(pops.validate(case), layout=layout)
    assert refused.value.reason == "moving_face"


def test_a_step_rule_on_a_transported_total_is_refused_naming_lot_t():
    case, layout, _, _ = build_transport(strategy=AdaptiveCFL(cfl=0.5, max_dt=0.01))
    with pytest.raises((CellTotalNotYetElaborated, MovingMembersNotYetElaborated),
                       match="G03 lot T"):
        pops.resolve(pops.validate(case), layout=layout)


@pytest.mark.parametrize("change", ["scaled", "negated", "retimed"])
def test_a_member_rate_alias_must_preserve_its_exact_value_and_stage(change):
    case, _, _, geometry = build_transport()
    program = case._time
    original = next(value for value in all_ops(program) if value.op == OPERATION)
    total, member, member_rate = original.inputs
    owner = geometry.declared
    assert owner is not None
    if change == "retimed":
        changed = program.value(change, member_rate, at=program.stage("later", c=1 / 3))
    else:
        coefficient = 2 if change == "scaled" else -1
        changed = program.value(change, coefficient * member_rate, at=member_rate.point)
    transport = owner.transport(original.attrs["operator_handle"])
    with pytest.raises(ValueError, match="rate of its members"):
        transport(total, member, changed)


def test_a_transport_cannot_use_another_moving_configurations_members():
    from pops.mesh.moving_configuration import MovingConfiguration

    case, _, _, geometry = build_transport()
    program = case._time
    original = next(value for value in all_ops(program) if value.op == OPERATION)
    total, _, member_rate = original.inputs
    owner = geometry.declared
    assert owner is not None and geometry.node_rate is not None
    other = MovingConfiguration(geometry.base, node_rate=2 * geometry.node_rate)
    foreign = other.members(case, name="foreign_motion")
    assert foreign is not None and foreign.block != owner.block
    member = program.state(foreign.block[foreign.state]).n
    rate = program.value("foreign_motion_rate", foreign.rate(member), at=member_rate.point)
    transport = owner.transport(original.attrs["operator_handle"])
    with pytest.raises(ValueError, match="owner of the moving members"):
        transport(total, member, rate)


def test_a_transport_refuses_a_layout_with_different_patches_from_the_declared_members():
    from pops.layouts import Uniform
    from pops.mesh import CartesianGrid

    case, _, _, configuration = build_transport()
    layout = Uniform(CartesianGrid(frame=configuration.frame, cells=(2*CELLS,)))
    with pytest.raises(MovingMembersNotYetElaborated, match="co-located layout") as refused:
        pops.resolve(pops.validate(case), layout=layout)
    assert refused.value.reason == "moving_layout"


@pytest.mark.parametrize("change, diagnostic", [
    ("stage", r"stage.*u1.*m1.*coefficient"),
    ("terminal", r"commit.*coefficient"),
    ("missing", r"commit.*configuration"),
])
def test_moving_total_and_geometry_require_the_same_update(monkeypatch, change, diagnostic):
    make_value = pops.Program.value
    commit_many = pops.Program.commit_many

    def value(program, name, expr, **kwargs):
        if change == "stage" and name.endswith("_m1"):
            m = next(state for state in program._values if state.name == "configuration")
            km = next(rate for rate in program._values if rate.name.endswith("_km0"))
            expr = m + 2*program.dt*km
        if change == "terminal" and name.endswith("_m_next"):
            m = next(state for state in program._values if state.name == "configuration")
            rates = tuple(next(rate for rate in program._values if rate.name.endswith("_km%d" % i))
                          for i in range(3))
            expr = m + program.dt*((1/6)*rates[0] + (2/3)*rates[1] + (1/6)*rates[2])
        return make_value(program, name, expr, **kwargs)

    def commit(program, mapping):
        if change == "missing":
            mapping = {key: item for key, item in mapping.items() if not item.name.endswith("_m_next")}
        return commit_many(program, mapping)

    monkeypatch.setattr(pops.Program, "value", value)
    monkeypatch.setattr(pops.Program, "commit_many", commit)
    case, layout, _, _ = build_transport(initial="constant")
    with pytest.raises(MovingMembersNotYetElaborated, match=diagnostic) as refused:
        pops.resolve(pops.validate(case), layout=layout)
    assert refused.value.reason == "moving_update"


def test_an_independent_static_layout_does_not_require_cell_totals():
    from pops.analytic import coordinate, time
    from pops.domain import CartesianDomain
    from pops.frames import Cartesian1D
    from pops.initial import InitialCondition
    from pops.layouts import Uniform
    from pops.lib.initial import BindArray
    from pops.mesh import CartesianGrid
    from pops.mesh.moving_configuration import MovingConfiguration
    from pops.mesh.layout_plan import LayoutPlanBuilder
    from pops.projection import ConservativeCellAverage
    from pops.time import FixedDt

    frame = CartesianDomain("mixed", (0.0,), (1.0,)).frame(Cartesian1D())
    case = pops.Case("mixed-layouts")
    program = pops.Program("mixed-layouts")
    x, t = coordinate(frame, frame.axes[0]), time(program.clock)
    moving_layout = Uniform(CartesianGrid(frame=frame, cells=(4,)))
    fixed_layout = Uniform(CartesianGrid(frame=frame, cells=(8,)))
    motion = MovingConfiguration(moving_layout, node_rate=x * (1 - x)).members(case, "motion")
    m = program.state(motion.block[motion.state])
    program.commit(m.next, program.value("new_geometry", m.n + program.dt*motion.rate(m.n),
                                         at=m.next.point))

    scalar = pops.Model("independent-scalar", frame=frame)
    u = scalar.state("u", components=("u",))
    scalar.source("decay", on=u, value=(-u[0],))
    b = case.block("fixed", scalar)
    case.initials.add(InitialCondition(state=b[u], value=BindArray(),
                                       projection=ConservativeCellAverage()))
    q = program.state(b[u])
    decay = scalar.module.operator_handle("decay")
    program.commit(q.next, program.value("new_scalar", q.n + program.dt*decay(q.n), at=q.next.point))
    program.step_strategy(FixedDt(1e-3))
    case.program(program)

    checked = pops.validate(case)
    subjects = checked.layout_subjects()
    layout_builder = LayoutPlanBuilder(checked.owner_path.canonical())
    layouts = {"motion": moving_layout, "fixed": fixed_layout}
    handles = {name: layout_builder.layout(name, layout) for name, layout in layouts.items()}
    for block in subjects.blocks:
        layout_builder.assign_block(block, handles[block.local_id])
    for state in subjects.states:
        layout_builder.assign_state(state, handles[state.block_ref.local_id])
    plan = layout_builder.resolve(**subjects.to_dict())
    resolved = pops.resolve(checked, layout=plan, layout_providers={
        handle: layouts[name] for name, handle in handles.items()
    })
    assert [block.name for block in resolved.blocks] == ["motion", "fixed"]
    assert tuple(resolved.layout_targets.values()) == ("system", "system")
