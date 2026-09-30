"""Source-only owner, geometry coupling and exact projection contracts."""
from fractions import Fraction

import pytest
import pops
from pops.analytic import coordinate, sin, time
from pops.domain import CartesianDomain
from pops.frames import Cartesian1D
from pops.math import ddt, div
from pops.mesh import GeometryEvolution
from pops.time import MovingFieldProjection


def authoring(*, components=("density",)):
    frame = CartesianDomain("line", (0.,), (1.,)).frame(Cartesian1D())
    model = pops.Model("transport", frame=frame)
    state = model.state("u", components=components)
    flux = model.flux("physical", frame=frame, state=state,
                      components={axis: tuple(state) for axis in frame.axes},
                      waves={axis: (1.,)*len(components) for axis in frame.axes})
    rate = model.rate("balance", equation=ddt(state) == -div(flux))
    case = pops.Case("coupled_geometry")
    block = case.block("fluid", model)
    program = pops.Program("ale")
    temporal = program.state(block[state])
    x = coordinate(frame, frame.axes[0])
    evolution = GeometryEvolution((x+.08*sin(6.283185307179586*x)*time(program.clock),))
    geometry = program.geometry_state(temporal.n, evolution=evolution)
    return case, program, temporal, geometry, rate


def policy():
    return MovingFieldProjection((Fraction(1, 2), Fraction(1, 2)), (1, 0))


@pytest.mark.parametrize("components", [("a",), ("a", "b", "c"), ("c", "a", "b")])
def test_coupled_ssa_retains_equation_operator_and_projection(components):
    _, program, state, geometry, rate = authoring(components=components)
    physical = rate(state.n)
    candidate = program.reynolds_update(geometry, physical_rate=physical,
        projection=policy(), geometry_tolerance=1e-13, at=state.next.point)
    program.commit(state.next, candidate)
    assert candidate.vtype == "state_geometry"
    assert candidate.inputs == (geometry, physical)
    assert candidate.state_ref == state.n.state_ref
    assert program._commits[state.n.state_ref] is candidate
    assert physical.attrs["operator_handle"] is not None
    serialized = program._serialize(include_provenance=False)
    assert "moving-field-projection/linear@1" in str(serialized)
    assert "exact_endpoint_displacement@1" in str(serialized)


def test_bare_density_cannot_commit_to_a_moving_physical_state():
    _, program, state, _, _ = authoring()
    raw = program.value("raw", state.n, at=state.next.point)
    with pytest.raises(ValueError, match="coupled state/geometry"):
        program.commit(state.next, raw)
    assert not program._commits


def test_stale_stage_or_foreign_geometry_refuses_without_allocating_a_candidate():
    _, program, state, geometry, rate = authoring()
    stage = program.stage("half", c=Fraction(1, 2))
    sampled = program.value("half_state", state.n, at=stage)
    physical = rate(sampled)
    original = program._serialize(include_provenance=False)
    with pytest.raises(ValueError, match="same exact state/clock/point"):
        program.reynolds_update(geometry, physical_rate=physical,
            projection=policy(), geometry_tolerance=1e-13, at=state.next.point)
    assert program._serialize(include_provenance=False) == original
    _, other, other_state, _, other_rate = authoring()
    with pytest.raises(ValueError, match="different Program"):
        other.reynolds_update(geometry, physical_rate=other_rate(other_state.n),
            projection=policy(), geometry_tolerance=1e-13, at=other_state.next.point)


def test_projection_must_preserve_constants_and_declare_source_measure_weights():
    with pytest.raises(ValueError, match="sum exactly to one"):
        MovingFieldProjection((Fraction(1, 2), Fraction(1, 3)), (1, 0))
    with pytest.raises(TypeError, match="exactly two explicit weights"):
        MovingFieldProjection((1, 0), (1,))
    with pytest.raises(TypeError, match="finite scalar literals"):
        MovingFieldProjection((float("nan"), 0), (1, 0))
