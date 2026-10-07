"""Resolved AMR authority follows exact pointwise numerical methods, not storage adapters."""
from dataclasses import replace

import pops
import pytest
from pops.amr import (AMRExecution, AMRHierarchy, AMRRegrid, AMRTagging, AMRTransfer,
                      Buffer, ConflictPolicy, EqualityPolicy, Hysteresis, Tag)
from pops.amr._resolution import AMRTaggingResolutionContext, ResolvedAMRStateStorage
from pops.domain import Rectangle
from pops.frames import Cartesian2D
from pops.initial import InitialCondition
from pops.layouts import AMR
from pops.lib.amr import StateTransfer
from pops.lib.initial import Constant
from pops.lib.time import ForwardEuler
from pops.math import ValueExpr, ddt
from pops.mesh import CartesianGrid, PeriodicAxes
from pops.mesh._amr._transfer_contracts import COARSE_FINE_FILL
from pops.numerics import DiscretizationPlan, StateStorage
from pops.params import RuntimeParam
from pops.projection import ConservativeCellAverage
from pops.time import FixedDt


def _case(*, name="explicit_local", levels=1, depth=1):
    frame = Rectangle("local_domain", (0., 0.), (1., 1.)).frame(Cartesian2D())
    model = pops.Model("local_source", frame=frame)
    state = model.state("U", components=("u",))
    rate = model.rate("decay", equation=ddt(state) == model.source(
        "reaction", on=state, value=(-state[0],)))
    case = pops.Case(name)
    block = case.block("material", model)
    plan = DiscretizationPlan()
    plan.rates.add(rate, StateStorage(ghost_depth=depth))
    case.numerics(plan, block=block)
    program = ForwardEuler(block[state], rate=rate)
    program.step_strategy(FixedDt(.01))
    case.program(program)
    case.initials.add(InitialCondition(state=block[state], value=Constant((1.,)),
                                      projection=ConservativeCellAverage()))
    transfer = AMRTransfer()
    transfer.state(block[state], StateTransfer())
    threshold = case.param(RuntimeParam("refine_threshold", default=.5))
    layout = AMR(grid=CartesianGrid(frame=frame, cells=(8, 8),
                                    periodic=PeriodicAxes(frame.axes)),
        hierarchy=AMRHierarchy(max_levels=levels, ratios=(2,) * (levels - 1)),
        tagging=AMRTagging(rules=(Tag(ValueExpr(block[state])["u"] > case.value(threshold)),
                                 Buffer(cells=0)),
            hysteresis=Hysteresis(0, EqualityPolicy.HOLD),
            conflict_policy=ConflictPolicy.REFINE_WINS),
        regrid=AMRRegrid.frozen(), transfer=transfer, execution=AMRExecution.synchronous())
    return case, layout, block[state]


@pytest.mark.parametrize("levels,depth", [(1, 4), (2, 3)])
def test_explicit_plan_publishes_exact_amr_storage_and_halo(levels, depth):
    case, layout, state = _case(levels=levels, depth=depth)
    resolved = pops.resolve(pops.validate(case), layout=layout)
    block, = resolved.blocks
    assert block.numerics is not None
    assert all(type(row.method) is StateStorage for row in block.numerics.rates)
    authority = ResolvedAMRStateStorage(block, case.resolve(state))
    assert authority.ghost_depth == depth
    assert resolved.resolved_hierarchy.plan.nesting.stencil.minimum_buffer == (depth, depth)
    assert resolved.resolved_hierarchy.plan.nesting.reflux.minimum_buffer == (0, 0)
    if levels == 2:
        fill = resolved.amr_transfer.for_subject(case.resolve(state), COARSE_FINE_FILL)
        assert fill.requirements
        assert all(row.accuracy.ghost_depth == (depth, depth) for row in fill.requirements)


def test_foreign_numerical_block_and_rate_cannot_authenticate_storage():
    case, layout, state = _case()
    resolved = pops.resolve(pops.validate(case), layout=layout)
    other, other_layout, _ = _case(name="foreign_local")
    foreign = pops.resolve(pops.validate(other), layout=other_layout)
    block, = resolved.blocks
    neighbor, = foreign.blocks
    subject = case.resolve(state)
    with pytest.raises(TypeError, match="flux-free"):
        ResolvedAMRStateStorage(replace(block, numerics=neighbor.numerics), subject)
    swapped = replace(block.numerics, rates=neighbor.numerics.rates)
    with pytest.raises(ValueError, match="exact resolved block state"):
        ResolvedAMRStateStorage(replace(block, numerics=swapped), subject)


def test_explicit_plan_halo_cannot_be_overridden_by_detached_storage():
    case, layout, state = _case(depth=2)
    resolved = pops.resolve(pops.validate(case), layout=layout)
    with pytest.raises(TypeError, match="flux-free"):
        ResolvedAMRStateStorage(replace(resolved.blocks[0], spatial=StateStorage(ghost_depth=3)),
                                case.resolve(state))


def test_explicit_plan_and_spatial_competition_still_refuses():
    case, layout, _ = _case()
    case._block_registry.spec("material")["spatial"] = StateStorage()
    with pytest.raises(ValueError, match="competing spatial"):
        pops.resolve(pops.validate(case), layout=layout)


def test_diffusion_sharing_storage_adapter_is_not_a_local_authority():
    from tests.python.unit.codegen.test_diffusion_program import resolved_heat

    resolved, subject, _ = resolved_heat()
    block, = resolved.blocks
    fake_local = replace(block, spatial=StateStorage())
    assert not ResolvedAMRStateStorage.supports_block(fake_local)
    with pytest.raises(TypeError, match="flux-free"):
        ResolvedAMRStateStorage(fake_local, subject)


def test_pointwise_plan_does_not_invent_gradient_stencil():
    case, layout, state = _case()
    resolved = pops.resolve(pops.validate(case), layout=layout)
    projection = resolved.layout_plan.project(resolved.layout_plan.layouts[0].handle)
    block, = resolved.blocks
    context = AMRTaggingResolutionContext(owner=case.owner_path.canonical(),
        layout_plan=projection, numerics=(block.numerics,), resolve=case.resolve,
        state_storage=(ResolvedAMRStateStorage(block, case.resolve(state)),))
    with pytest.raises(ValueError, match="no resolved spatial discretization"):
        context._discrete_context(case.resolve(state))
