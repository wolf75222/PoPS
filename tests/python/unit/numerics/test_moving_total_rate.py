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
