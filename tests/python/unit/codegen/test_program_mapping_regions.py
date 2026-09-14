from fractions import Fraction
import pytest
import pops
from pops.domain import Rectangle
from pops.frames import Cartesian2D
from pops.model import PhysicalDimension, PhysicalSupport
from pops.mesh import AxisQuadrature, PhysicalSupportMap
from pops.codegen.program_mapping_regions import program_map_invocations, plan_program_mapping_regions


def authored_maps():
    frame = Rectangle("domain", (0, 0), (1, 1)).frame(Cartesian2D())
    phase = PhysicalSupport((("x", "domain-x"), ("v", "domain-v")))
    position = PhysicalSupport((("x", "domain-x"),))
    unit = PhysicalDimension()
    case = pops.Case("stage transfers")
    program = pops.Program("same map twice at distinct points")
    states = {}
    for name, support in (("source", phase), ("moment", position), ("back", phase)):
        model = pops.Model(name, frame=frame)
        state = model.state("U", components=("u",), support=support,
                            units=(unit,), sampling="cell_average")
        states[name] = program.state(case.block(name, model)[state])
    reduction = PhysicalSupportMap(phase, position,
        reductions=(AxisQuadrature(1, 0, 1, 4, unit, weights=(1, -1, 2, 0)),))
    extension = PhysicalSupportMap(position, phase)
    first = states["source"].stage("half", point=program.stage("half", c=Fraction(1, 2)))
    program.value(first, 2 * states["source"].n)
    moment = states["moment"].stage("half moment", point=program.stage("half moment", c=Fraction(1, 2)))
    program.map(reduction, source=first, target=moment)
    back = states["back"].stage("half back", point=program.stage("half back", c=Fraction(1, 2)))
    program.map(extension, source=moment, target=back)
    late = states["source"].stage("three quarters", point=program.stage("three quarters", c=Fraction(3, 4)))
    program.value(late, 3 * first)
    second = states["moment"].stage("late moment", point=program.stage("late moment", c=Fraction(3, 4)))
    program.map(reduction, source=late, target=second)
    for name, value in (("source", late), ("moment", second), ("back", back)):
        program.commit(states[name].next,
                       program.value(name + " accepted", value, at=states[name].next.point))
    return program, states, reduction


def test_map_ports_preserve_three_invocations_and_exact_intermediate_inputs():
    program, _, _ = authored_maps()
    program.validate()
    calls = program_map_invocations(program)
    assert len(calls) == 3
    assert len({row.identity for row in calls}) == 3
    assert sorted(row.source.point.time.offset.to_python() for row in calls) == [Fraction(1, 2), Fraction(1, 2), Fraction(3, 4)]
    assert {row.source.inputs[0].op for row in calls} == {"linear_combine", "layout_map_import"}
    assert all(row.target.op == "layout_map_import" for row in calls)


def test_region_schedule_joins_ports_and_preserves_causal_chain():
    program, _, _ = authored_maps()
    plan = plan_program_mapping_regions(program, {name: name for name in ("source", "moment", "back")})
    positions = {event: i for i, event in enumerate(plan.events)}
    for region in plan.regions:
        if region.barrier is not None:
            assert positions[("region", region.layout, region.index)] < positions[("map", region.barrier, 0)]
            assert positions[("map", region.barrier, 0)] < positions[("region", region.layout, region.index + 1)]
    assert len([event for event in plan.events if event[0] == "map"]) == 3
    assert plan.events == plan_program_mapping_regions(program, dict(reversed(list({name: name for name in ("source", "moment", "back")}.items())))).events


def test_map_rejects_time_mismatch_before_mutating_program():
    program, states, reduction = authored_maps()
    target = states["moment"].stage("wrong time", point=program.stage("wrong time", c=Fraction(1, 3)))
    before = len(program._values)
    with pytest.raises(ValueError, match="same exact clock coordinate"):
        program.map(reduction, source=states["source"].n, target=target)
    assert len(program._values) == before
