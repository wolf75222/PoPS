"""One physical authority must retain each block's independent numerical body."""
from __future__ import annotations

import pops
import pytest

from pops.codegen.program_models import ProgramModelGraph
from pops.codegen.user_reconstruction_lowering import (
    emit_user_reconstruction_policy, user_reconstruction_source_identity,
)
from pops.domain import Rectangle
from pops.frames import Cartesian2D
from pops.layouts import Uniform
from pops.math import ddt, div
from pops.mesh import CartesianGrid, PeriodicAxes
from pops.numerics import DiscretizationPlan, reconstruction, riemann, variables
from pops.numerics.spatial import FiniteVolume
from pops.representations import Conservative
from pops.spaces import CellState
from pops.time import FixedDt


def resolved_case(*, reverse=False):
    frame = Rectangle("domain", lower=(0., 0.), upper=(1., 1.)).frame(Cartesian2D())
    model = pops.Model("shared_transport", frame=frame)
    state = model.state("U", components=("q",), representation=Conservative(),
                        space=CellState(frame=frame))
    flux = model.flux("advection", frame=frame, state=state,
        components={axis: (.7 * state[0],) for axis in frame.axes},
        waves={axis: (.7 + 0 * state[0],) for axis in frame.axes})
    rate = model.rate("balance", equation=ddt(state) == -div(flux))
    first = reconstruction.User(lambda sample: sample(0), formal_order=1, name="same_name")
    second = reconstruction.User(lambda sample: sample(0) + .25 * (sample(1) - sample(-1)),
                                 formal_order=2, name="same_name")
    authored = {"first": first, "second": second}
    case = pops.Case("shared_model_distinct_reconstructions")
    program = pops.Program("two_independent_steps")
    names = ("second", "first") if reverse else ("first", "second")
    for name in names:
        block = case.block(name, model)
        numerics = DiscretizationPlan()
        numerics.rates.add(rate, FiniteVolume(
            flux=flux, variables=variables.Conservative(state),
            reconstruction=authored[name], riemann=riemann.Rusanov()))
        case.numerics(numerics, block=block)
        temporal = program.state(block[state])
        rhs = program.value(name + "_rhs", rate(temporal.n), at=temporal.n.point)
        result = program.value(name + "_next", temporal.n + program.dt * rhs,
                               at=temporal.next.point)
        program.commit(temporal.next, result)
    program.step_strategy(FixedDt(1.e-3))
    case.program(program)
    layout = Uniform(CartesianGrid(frame=frame, cells=(8, 8),
                                   periodic=PeriodicAxes(frame.axes)))
    resolved = pops.resolve(pops.validate(case), layout=layout)
    return resolved, {name: recipe.options["source_identity"] for name, recipe in authored.items()}


@pytest.mark.parametrize("reverse", (False, True), ids=("first_then_second", "second_then_first"))
def test_same_model_keeps_each_blocks_authored_reconstruction_during_program_lowering(reverse):
    resolved, expected = resolved_case(reverse=reverse)
    assert expected["first"] != expected["second"]
    assert resolved.blocks[0].model is resolved.blocks[1].model
    assert resolved.blocks[0].numerics.identity != resolved.blocks[1].numerics.identity

    graph = ProgramModelGraph.from_resolved_blocks(resolved.blocks)
    emitted = {}
    for name, identity in expected.items():
        model = graph.model_for_block(name)
        assert user_reconstruction_source_identity(model) == identity
        emitted[name] = emit_user_reconstruction_policy(model)
        assert identity in emitted[name]
        other = "second" if name == "first" else "first"
        assert expected[other] not in emitted[name]
    assert "sample(-1)" not in emitted["first"]
    assert "sample(-1)" in emitted["second"]
    assert "sample(1)" in emitted["second"]
