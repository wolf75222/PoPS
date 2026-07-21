from __future__ import annotations

import pytest

import pops
from pops.diagnostics import Integral
from pops.domain import Rectangle
from pops.frames import Cartesian2D
from pops.mesh import LayoutPlanBuilder, normalize_layout_plan
from pops.layouts import Uniform
from pops.output import Checkpoint, ConsumerGraph, HDF5, ParaView, ScientificOutput
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
            format=HDF5(mode=ParallelMode.COLLECTIVE),
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
    diagnostic_quantity, = output.diagnostic_quantities
    assert diagnostic_quantity.reference == case.resolve(state)
    assert diagnostic_quantity.levels == (0,)
    assert diagnostic_quantity.execution["operations"] == ({
        "name": "integral",
        "reduction": "sum",
        "transform": "identity",
        "metric_weighted": True,
    },)
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


def test_embedded_diagnostic_identity_is_qualified_by_its_consumer():
    case, block, state = _case()
    clock = Clock("macro", owner=case.owner_path)
    schedule = every(10, clock=clock)
    diagnostic = Integral(block=block, cadence=schedule)
    graph = ConsumerGraph.from_consumers(tuple(
        ScientificOutput(
            format=HDF5(),
            schedule=schedule,
            fields=(state,),
            diagnostics=(diagnostic,),
            target=target,
        )
        for target in ("state/first", "state/second")
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
    diagnostics = tuple(node.diagnostic_quantities[0] for node in resolved.nodes)
    assert len({value.handle.qualified_id for value in diagnostics}) == 2
    assert all(
        node.handle.local_id in diagnostic.handle.owner_path.segments
        for node, diagnostic in zip(resolved.nodes, diagnostics, strict=True)
    )


def test_output_format_options_refuse_python_truthiness_coercion() -> None:
    with pytest.raises(TypeError, match="exact pops.output.ParallelMode"):
        HDF5(mode="serial")
    with pytest.raises(TypeError, match="exact bool or None"):
        HDF5(series=1)
    assert HDF5().consumer_data()["options"] == {"mode": "serial", "series": True}
    assert ParaView().consumer_data()["options"] == {"mode": "serial", "series": True}
    assert ParaView(mode=ParallelMode.PER_RANK).consumer_data()["options"] == {
        "mode": "per_rank", "series": False,
    }


def test_scientific_output_target_is_logical_and_format_independent() -> None:
    case, _block, state = _case()
    schedule = every(1, clock=Clock("macro", owner=case.owner_path))
    with pytest.raises(ValueError, match="must not contain a file suffix"):
        ScientificOutput(
            format=ParaView(),
            schedule=schedule,
            fields=(state,),
            target="solution/tracer.vtu",
        )


def test_paraview_single_layout_contract_fails_during_resolution():
    domain = Rectangle("unit", (0.0, 0.0), (1.0, 1.0))
    frame = domain.frame(Cartesian2D())
    model = pops.Model("transport", frame=frame)
    state = model.state(
        "U", components=("u",), representation=Conservative(),
        space=CellState(frame=frame))
    case = pops.Case("consumer-two-layouts")
    first_block = case.block("first", model)
    second_block = case.block("second", model)
    first_state = first_block[state]
    second_state = second_block[state]
    schedule = every(1, clock=Clock("macro", owner=case.owner_path))
    paraview = ConsumerGraph.from_consumers((
        ScientificOutput(
            format=ParaView(),
            schedule=schedule,
            fields=(first_state, second_state),
            target="state/two-layouts",
        ),
    ))
    case.consumers(paraview)
    pops.validate(case)

    first_block = case.resolve(first_block)
    second_block = case.resolve(second_block)
    first_state = case.resolve(first_state)
    second_state = case.resolve(second_state)
    builder = LayoutPlanBuilder(
        case.owner_path.canonical(),
        handle_resolver=case.resolve,
    )
    first_layout = builder.layout("first", Uniform(cartesian_grid(n=8)))
    second_layout = builder.layout("second", Uniform(cartesian_grid(n=8)))
    builder.assign_block(first_block, first_layout)
    builder.assign_state(first_state, first_layout)
    builder.assign_block(second_block, second_layout)
    builder.assign_state(second_state, second_layout)
    layout_plan = builder.resolve(
        states=(first_state, second_state),
        blocks=(first_block, second_block),
    )

    with pytest.raises(
        ValueError,
        match="accepts one exact layout per consumer",
    ):
        paraview.resolve(
            case.resolve,
            layout_plan,
            owner=case.owner_path.canonical(),
        )

    mixed = ConsumerGraph.from_consumers((
        ScientificOutput(
            format=ParaView(),
            schedule=schedule,
            fields=(first_state,),
            diagnostics=(Integral(block=second_block, cadence=schedule),),
            target="state/mixed-layouts",
        ),
    ))
    with pytest.raises(
        ValueError,
        match="accepts one exact layout per consumer",
    ):
        mixed.resolve(
            case.resolve,
            layout_plan,
            owner=case.owner_path.canonical(),
        )

    hdf5 = ConsumerGraph.from_consumers((
        ScientificOutput(
            format=HDF5(),
            schedule=schedule,
            fields=(first_state, second_state),
            target="state/two-layouts",
        ),
    ))
    assert hdf5.resolve(
        case.resolve,
        layout_plan,
        owner=case.owner_path.canonical(),
    ).is_resolved
