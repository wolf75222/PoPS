"""Canonical Program and ConsumerGraph authorities survive resolution intact."""
from __future__ import annotations

import pops
from pops.diagnostics import Integral
from pops.domain import Rectangle
from pops.frames import Cartesian2D
from pops.layouts import Uniform
from pops.output import ConsumerGraph, HDF5, ScientificOutput
from pops.output._consumer_contracts import ConsumerKind
from pops.representations import Conservative
from pops.spaces import CellState
from pops.time import every
from tests.python.support.layout_plan import cartesian_grid


def test_block_time_and_diagnostics_never_drop_from_resolved_plan():
    """The final single authorities replace, and never emulate, per-block side channels."""
    domain = Rectangle("unit", (0.0, 0.0), (1.0, 1.0))
    frame = domain.frame(Cartesian2D())
    model = pops.Model("transport", frame=frame)
    state = model.state(
        "U",
        components=("u",),
        representation=Conservative(),
        space=CellState(frame=frame),
    )
    case = pops.Case("total-semantic-accounting")
    block = case.block("tracer", model)
    qualified_state = block[state]
    program = pops.Program("macro-time")
    schedule = every(4, clock=program.clock)
    graph = ConsumerGraph.from_consumers((
        ScientificOutput(
            format=HDF5(parallel=False),
            schedule=schedule,
            fields=(qualified_state,),
            diagnostics=(Integral(block=block, cadence=schedule),),
            target="outputs/tracer",
        ),
    ))
    case.program(program)
    case.consumers(graph)

    validated = pops.validate(case)
    resolved = pops.resolve(validated, layout=Uniform(cartesian_grid(n=8)))

    assert resolved.time is program
    assert resolved.consumer_graph is not graph
    assert resolved.consumer_graph.is_resolved
    output, = resolved.consumer_graph.nodes
    assert output.kind is ConsumerKind.SCIENTIFIC_OUTPUT
    assert output.quantities[0].reference == case.resolve(qualified_state)
    assert output.output_format_data["provider_id"] == "pops.output.hdf5.v1"
    diagnostic, = output.to_data()["diagnostics"]
    assert diagnostic["references"] == [case.resolve(block).canonical_identity()]
    assert case.snapshot.to_dict()["consumers"]["phase"] == "authoring"
    resolved.verify()
