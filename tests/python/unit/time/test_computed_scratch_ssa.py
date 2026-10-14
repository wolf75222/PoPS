"""Adversarial lifetime tests for typed computed-field stencil results."""
import pytest

import pops
from pops.fields import CellMidpoint


def _program(width):
    from pops.domain import CartesianDomain
    from pops.frames import Cartesian1D
    frame = CartesianDomain("scratch_ssa", (0.,), (1.,)).frame(Cartesian1D())
    model = pops.Model("scratch_ssa", frame=frame)
    state = model.state("u", components=tuple(f"q{i}" for i in range(width)), sampling="cell_average")
    case = pops.Case("scratch_ssa")
    block = case.block("q", model)
    program = pops.Program("scratch_ssa")
    return program, program.state(block[state])


@pytest.mark.parametrize("width", (1, 3))
@pytest.mark.parametrize("in_loop", (False, True))
def test_reused_computed_prototype_never_aliases_live_laplacian_results(width, in_loop):
    p, state = _program(width)
    allocations = []

    def body(builder, current, _index=None):
        field = builder.cell_field(current, quadrature=CellMidpoint())
        prototype = builder.scalar_field(like=field)
        first = builder.laplacian(prototype, field)
        doubled = builder.value("twice", 2*current)
        second_field = builder.cell_field(doubled, quadrature=CellMidpoint())
        second = builder.laplacian(prototype, second_field)
        allocations.append((prototype, first, second))
        a = builder.cell_mean(first, template=current, quadrature=CellMidpoint())
        b = builder.cell_mean(second, template=current, quadrature=CellMidpoint())
        return builder.value("three_laplacians", a+b)

    result = p.fold(state.n, 2, body) if in_loop else body(p, state.n)
    p.commit(state.next, p.value("accepted", result, at=state.next.point))
    assert p.validate()
    for prototype, first, second in allocations:
        assert first.inputs[0] is not prototype
        assert second.inputs[0] is not prototype
        assert first.inputs[0] is not second.inputs[0]
        assert first.space == second.space == prototype.space
        assert first.point == second.point == prototype.point
        assert len(first.space.components) == width
        if in_loop:
            assert first.region == second.region != 0


@pytest.mark.parametrize("operation", ("gradient", "divergence"))
def test_other_computed_stencils_also_return_independent_storage(operation):
    # In physical Dim1 both grad and div have one component, so a single typed
    # prototype suffices to expose the same lifetime hazard as the Laplacian.
    p, state = _program(1)
    field = p.cell_field(state.n, quadrature=CellMidpoint())
    prototype = p.scalar_field(like=field)
    first = getattr(p, operation)(prototype, field)
    second = getattr(p, operation)(prototype, field)
    assert first.inputs[0] is not prototype
    assert second.inputs[0] is not prototype
    assert first.inputs[0] is not second.inputs[0]
