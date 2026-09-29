"""Exact AMR storage authority cannot be borrowed from another state or transfer."""

from dataclasses import replace

import pops
import pytest

from pops.amr._resolution import ResolvedAMRStateStorage
from pops.initial import InitialCondition
from pops.lib.amr import StateTransfer
from pops.lib.initial import Constant
from pops.mesh._amr._transfer_contracts import COARSE_FINE_FILL
from pops.numerics import StateStorage
from pops.projection import ConservativeCellAverage
from tests.python.integration.runtime.test_amr_local_transform_post_sync import _case_and_layout
from tests.python.support.affine_push_forward_amr_case import affine_amr_case


def _mixed_case():
    case, layout = _case_and_layout()
    field_subject, = case._block_registry.spec("field")["states"]
    model = pops.Model("local_neighbor", frame=layout.grid.frame)
    state = model.state("L", components=("r",))
    subject = case.block("local", model)[state]
    program = case._time
    local = program.state(subject)
    program.commit(local.next, program.value(
        "local_passthrough", 1.0 * local.n, at=local.next.point))
    case.initials.add(InitialCondition(
        state=subject, value=Constant((.8,)), projection=ConservativeCellAverage()))
    layout.transfer.state(subject, StateTransfer())
    return case, layout, field_subject, subject


def _projection(plan):
    return plan.layout_plan.project(plan.layout_plan.layouts[0].handle)


def test_mixed_numerical_neighbor_does_not_replace_local_storage_authority():
    case, layout, field_subject, local_subject = _mixed_case()
    plan = pops.resolve(pops.validate(case), layout=layout)
    blocks = {block.name: block for block in plan.blocks}
    assert blocks["field"].numerics is not None
    assert blocks["local"].numerics is None
    assert type(blocks["local"].spatial) is StateStorage
    assert plan.resolved_hierarchy.plan.nesting.reflux.minimum_buffer == (1, 1)

    local = case.resolve(local_subject)
    numerical = blocks["field"].numerics
    deeper = ResolvedAMRStateStorage(
        replace(blocks["local"], spatial=StateStorage(ghost_depth=3)), local)
    transfer = layout.transfer.resolve_references(case.resolve).resolve(
        _projection(plan), (numerical,), state_storage=(deeper,))
    local_fill = transfer.for_subject(local, COARSE_FINE_FILL)
    field_fill = transfer.for_subject(case.resolve(field_subject), COARSE_FINE_FILL)
    assert all(row.accuracy.ghost_depth == (3, 3) for row in local_fill.requirements)
    assert all(row.accuracy.ghost_depth == (1, 1) for row in field_fill.requirements)
    assert transfer.nesting_requirement.minimum_buffer == (3, 3)

    with pytest.raises(ValueError, match="no exact resolved spatial or local storage"):
        layout.transfer.resolve_references(case.resolve).resolve(
            _projection(plan), (numerical,), state_storage=())


def test_local_storage_cannot_authenticate_a_numerical_neighbor():
    case, layout, field_subject, local_subject = _mixed_case()
    plan = pops.resolve(pops.validate(case), layout=layout)
    blocks = {block.name: block for block in plan.blocks}
    field_subject = case.resolve(field_subject)
    with pytest.raises(TypeError, match="flux-free"):
        ResolvedAMRStateStorage(blocks["field"], field_subject)
    with pytest.raises(ValueError, match="exact resolved block state"):
        ResolvedAMRStateStorage(blocks["local"], field_subject)
    assert ResolvedAMRStateStorage(blocks["local"], case.resolve(local_subject))


def test_halo_beyond_native_coarse_fine_routes_refuses_exactly():
    case, layout, subject = affine_amr_case(levels=2)
    plan = pops.resolve(pops.validate(case), layout=layout)
    deeper = ResolvedAMRStateStorage(
        replace(plan.blocks[0], spatial=StateStorage(ghost_depth=4)), case.resolve(subject))
    with pytest.raises(ValueError, match="incompatible AMR transfer provider"):
        layout.transfer.resolve_references(case.resolve).resolve(
            _projection(plan), (), state_storage=(deeper,))
