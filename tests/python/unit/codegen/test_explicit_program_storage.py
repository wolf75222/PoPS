"""State storage is a representation contract, not a fabricated zero equation."""
import pops
import pytest

from pops.codegen.program_models import ProgramModelGraph
from pops.domain import Rectangle
from pops.frames import Cartesian2D
from pops.layouts import Uniform
from pops.mesh import CartesianGrid, PeriodicAxes
from pops.numerics import DiscretizationPlan, StateStorage
from pops.time import FixedDt


def case_with_storage(depth=2):
    frame = Rectangle("storage_box", (0., 0.), (1., 1.)).frame(Cartesian2D())
    model = pops.Model("equations_in_program", frame=frame)
    state = model.state("U", components=("a", "b"))
    case = pops.Case("explicit_storage")
    block = case.block("material", model)
    plan = DiscretizationPlan()
    plan.states.add(state, StateStorage(ghost_depth=depth))
    case.numerics(plan, block=block)
    P = pops.Program("authored_equations")
    u = P.state(block[state])
    P.commit(u.next, P.value("evolution", (u.n[0]+u.n[1], u.n[1]), at=u.next.point))
    P.step_strategy(FixedDt(.01))
    case.program(P)
    layout = Uniform(CartesianGrid(frame=frame, cells=(8, 8), periodic=PeriodicAxes(frame.axes)))
    return case, layout, model, state, plan


def test_program_storage_has_no_fabricated_rate_and_retains_ghost_authority():
    case, layout, model, _, _ = case_with_storage(3)
    assert not model._rate_contracts
    resolved = pops.resolve(pops.validate(case), layout=layout)
    block = resolved.blocks[0]
    assert block.spatial.ghost_depth == 3
    data = block.numerics.to_data()
    assert data["rates"] == [] and len(data["states"]) == 1
    native = ProgramModelGraph.from_resolved_blocks(resolved.blocks).model_for_block("material")._m
    assert native._program_state_ghost_depth == 3
    assert not native._flux


def test_storage_does_not_hide_an_unrealized_physical_balance():
    from pops.math import ddt
    case, _, model, state, plan = case_with_storage()
    source = model.source("actual_source", on=state, value=(state[0], state[1]))
    model.rate("actual_equation", equation=ddt(state) == source)
    with pytest.raises(ValueError, match="coverage"):
        plan.validate_for(model)


def test_storage_rejects_foreign_state_and_duplicate_authorities():
    _, _, model, state, plan = case_with_storage()
    with pytest.raises(ValueError, match="already"):
        plan.states.add(state, StateStorage())
    other = pops.Model("other", frame=Cartesian2D())
    foreign = other.state("U", components=("a", "b"))
    bad = DiscretizationPlan()
    bad.states.add(foreign, StateStorage())
    with pytest.raises(ValueError, match="another Model"):
        bad.validate_for(model)
