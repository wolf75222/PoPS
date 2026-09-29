"""Independent selection test for local storage beside another State's flux."""

import pops

from pops.codegen.program_models import ProgramModelGraph
from pops.domain import Rectangle
from pops.frames import Cartesian2D
from pops.layouts import Uniform
from pops.mesh import CartesianGrid, PeriodicAxes
from pops.numerics import StateStorage
from pops.time import FixedDt


def test_local_state_selection_survives_a_spatial_neighbour_through_lowering():
    frame = Rectangle("mixed_storage_review_box", (0., 0.), (1., 1.)).frame(Cartesian2D())
    model = pops.Model("mixed_storage_review", frame=frame)
    a = model.species("A", state=("a",))
    b = model.species("B", state=("b",))
    model.flux("transport", frame=frame, state=b,
               components={axis: (b[0],) for axis in frame.axes})

    case = pops.Case("mixed_storage_review")
    program = pops.Program("identity")
    for name, state in (("first_a", a), ("b", b), ("second_a", a)):
        block = case.block(name, model, states=(state,))
        value = program.state(block[state])
        program.commit(value.next, program.value(name, value.n, at=value.next.point))
    program.step_strategy(FixedDt(.01))
    case.program(program)
    layout = Uniform(CartesianGrid(
        frame=frame, cells=(4, 4), periodic=PeriodicAxes(frame.axes)))

    original_hash = model.module.module_hash()
    resolved = pops.resolve(pops.validate(case), layout=layout)
    assert [isinstance(block.spatial, StateStorage) for block in resolved.blocks] == [True, False, True]
    graph = ProgramModelGraph.from_resolved_blocks(resolved.blocks)
    for name in ("first_a", "second_a"):
        carrier = graph.model_for_block(name)._m
        assert not carrier._flux
        assert carrier._program_state_ghost_depth == 1
    assert model.module.module_hash() == original_hash
    assert all(spec["spatial"] is None for _, spec in case._blocks.items())
