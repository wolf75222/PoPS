from __future__ import annotations

import pytest

import pops
from pops.diagnostics import Integral
from pops.domain import Rectangle
from pops.frames import Cartesian2D
from pops.mesh import normalize_layout_plan
from pops.layouts import Uniform
from pops.output import Checkpoint, ConsumerGraph, HDF5, ScientificOutput
from pops.output._consumer_contracts import ConsumerKind, ParallelMode
from pops.representations import Conservative
from pops.spaces import CellState
from pops.time import Clock, every
from tests.python.support.layout_plan import cartesian_grid


def _case():
    domain = Rectangle("unit", (0.0, 0.0), (1.0, 1.0))
    frame = domain.frame(Cartesian2D())
    model = pops.Model("transport", frame=frame)
    state = model.state(
        "U", components=("u",), representation=Conservative(),
        space=CellState(frame=frame))
    case = pops.Case("consumer-authoring")
    block = case.block("tracer", model)
    return case, block, block[state]


def test_direct_consumers_resolve_references_layout_levels_and_parallel_mode():
    case, block, state = _case()
    clock = Clock("macro", owner=case.owner_path)
    output_schedule = every(10, clock=clock)
    diagnostic = Integral(block=block, cadence=output_schedule)
    graph = ConsumerGraph.from_consumers((
        ScientificOutput(
            format=HDF5(parallel=True),
            schedule=output_schedule,
            fields=(state,),
            diagnostics=(diagnostic,),
            target="state/tracer",
        ),
        Checkpoint(
            schedule=every(100, clock=clock),
            target="checkpoints/restart",
            bit_identical=True,
        ),
    ))
    case.consumers(graph)
    pops.validate(case)

    subjects = case.layout_subjects()
    layout = normalize_layout_plan(
        Uniform(cartesian_grid(n=8)),
        owner=case.owner_path.canonical(),
        states=subjects.states,
        fields=subjects.fields,
        blocks=subjects.blocks,
        handle_resolver=case.resolve,
    )
    resolved = graph.resolve(case.resolve, layout, owner=case.owner_path.canonical())

    assert resolved.is_resolved
    output = next(node for node in resolved.nodes
                  if node.kind is ConsumerKind.SCIENTIFIC_OUTPUT)
    checkpoint = next(node for node in resolved.nodes
                      if node.kind is ConsumerKind.CHECKPOINT)
    assert output.parallel_mode is ParallelMode.COLLECTIVE
    assert output.quantities[0].reference == case.resolve(state)
    assert output.quantities[0].levels == (0,)
    assert output.operation is None
    assert output.output_format_data["provider_id"] == "pops.output.hdf5.v1"
    diagnostic_data, = output.to_data()["diagnostics"]
    assert diagnostic_data["references"] == [case.resolve(block).canonical_identity()]
    assert diagnostic_data["descriptor"]["scheme"] == "integral"
    assert checkpoint.output_format is None
    assert checkpoint.operation_data["provider_id"] == "pops.restart.accepted-state-v3"
    assert checkpoint.operation_data["bit_identical"] is True
    assert case.snapshot.to_dict()["consumers"]["phase"] == "authoring"


def test_consumer_protocol_is_required_and_schedule_authority_is_unique():
    case, block, state = _case()
    clock = Clock("macro", owner=case.owner_path)
    with pytest.raises(TypeError, match="consumer_authoring"):
        ConsumerGraph.from_consumers((object(),))
    with pytest.raises(ValueError, match="same schedule"):
        ScientificOutput(
            format=HDF5(),
            schedule=every(10, clock=clock),
            fields=(state,),
            diagnostics=(Integral(block=block, cadence=every(5, clock=clock)),),
            target="state/tracer",
        )


def test_output_format_options_refuse_python_truthiness_coercion() -> None:
    with pytest.raises(TypeError, match="exact bool"):
        HDF5(parallel="false")
