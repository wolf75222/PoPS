"""Local State storage must exist in the resolved plan, not only in emitted C++."""

from types import SimpleNamespace
from dataclasses import replace

import pops
import pytest

from pops.codegen.inspect_compiled import _ghost_depth_by_block
from pops.codegen.program_models import ProgramModelGraph
from pops.domain import Rectangle
from pops.frames import Cartesian2D
from pops.layouts import Uniform
from pops.mesh import CartesianGrid, PeriodicAxes
from pops.numerics import StateStorage
from pops.time import FixedDt
from tests.python.support.local_residual_product_case import make_case


@pytest.mark.parametrize("reverse", (False, True))
def test_local_product_resolves_real_storage_before_inspection(reverse):
    case, layout, _ = make_case(reverse=reverse)
    resolved = pops.resolve(pops.validate(case), layout=layout)
    names = tuple(block.name for block in resolved.blocks)
    assert _ghost_depth_by_block(SimpleNamespace(plan=resolved), names) == dict.fromkeys(names, 1)
    graph = ProgramModelGraph.from_resolved_blocks(resolved.blocks)
    for block in resolved.blocks:
        assert isinstance(block.spatial, StateStorage)
        assert block.spatial.ghost_depth == 1
        native = graph.model_for_block(block.name)._m
        assert native._program_state_ghost_depth == block.spatial.ghost_depth
        assert not native._flux


def test_shared_multistate_a_b_a_retains_exact_selection_and_authoring():
    frame = Rectangle("local_states_box", (0., 0.), (1., 1.)).frame(Cartesian2D())
    model = pops.Model("local_states", frame=frame)
    a = model.species("A", state=("a",))
    b = model.species("B", state=("b", "c"))
    case = pops.Case("local_states")
    program = pops.Program("identity")
    for name, state in (("first_a", a), ("b", b), ("second_a", a)):
        block = case.block(name, model, states=(state,))
        value = program.state(block[state])
        program.commit(value.next, program.value(name, value.n, at=value.next.point))
    program.step_strategy(FixedDt(.01))
    case.program(program)
    layout = Uniform(CartesianGrid(frame=frame, cells=(4, 4), periodic=PeriodicAxes(frame.axes)))
    validated = pops.validate(case)
    before = model.module.module_hash()
    resolved = pops.resolve(validated, layout=layout)
    graph = ProgramModelGraph.from_resolved_blocks(resolved.blocks)
    assert _ghost_depth_by_block(SimpleNamespace(plan=resolved), ("first_a", "b", "second_a")) == {
        "first_a": 1, "b": 1, "second_a": 1}
    assert [graph.model_for_block(name)._m.n_vars for name in ("first_a", "b", "second_a")] == [1, 2, 1]
    assert all(isinstance(block.spatial, StateStorage) for block in resolved.blocks)
    assert model.module.module_hash() == before
    assert not getattr(model._dsl._m, "_program_only_storage_axes", ())
    assert not hasattr(model._dsl._m, "_program_state_ghost_depth")
    assert all(spec["spatial"] is None for _, spec in case._blocks.items())


def test_spatial_neighbour_does_not_change_local_state_admission():
    from pops.codegen.state_storage_lowering import resolve_local_state_storage

    frame = Rectangle("mixed_box", (0., 0.), (1., 1.)).frame(Cartesian2D())
    model = pops.Model("mixed_local_spatial", frame=frame)
    model.species("A", state=("a",))
    b = model.species("B", state=("b",))
    model.flux("transport", frame=frame, state=b,
               components={axis: (b[0],) for axis in frame.axes},
               waves={axis: (1.,) for axis in frame.axes})
    before = model.module.module_hash()
    admissions = [resolve_local_state_storage(model, state_space=name) for name in ("A", "B", "A")]
    assert isinstance(admissions[0], StateStorage) and isinstance(admissions[2], StateStorage)
    assert admissions[1] is None
    assert model.module.module_hash() == before
    assert not getattr(model._dsl._m, "_program_only_storage_axes", ())


def test_diffusion_grid_operator_is_not_admitted_as_local_storage():
    from pops import math
    from pops.codegen.state_storage_lowering import resolve_local_state_storage

    frame = Rectangle("diffusion_box", (0., 0.), (1., 1.)).frame(Cartesian2D())
    model = pops.Model("diffusion_state", frame=frame)
    state = model.state("U", components=("u",))
    flux = model.diffusive_flux("conduction", state=state, value=(math.grad(state[0]),))
    model.rate("diffusion", equation=math.ddt(state) == math.div(flux))
    assert resolve_local_state_storage(model, state_space="U") is None


def test_declared_source_stencil_larger_than_one_remains_authoritative():
    from pops import math
    from pops.lib.time import ForwardEuler
    from pops.numerics import DiscretizationPlan

    frame = Rectangle("deep_source_box", (0., 0.), (1., 1.)).frame(Cartesian2D())
    model = pops.Model("deep_source", frame=frame)
    state = model.state("U", components=("u",))
    source = model.source("local", on=state, value=(state[0],))
    rate = model.rate("source", equation=math.ddt(state) == source)
    case = pops.Case("deep_source")
    block = case.block("material", model)
    numerics = DiscretizationPlan()
    numerics.rates.add(rate, StateStorage(ghost_depth=4))
    case.numerics(numerics, block=block)
    program = ForwardEuler(block[state], rate=rate)
    program.step_strategy(FixedDt(.01))
    case.program(program)
    layout = Uniform(CartesianGrid(frame=frame, cells=(4, 4), periodic=PeriodicAxes(frame.axes)))
    resolved = pops.resolve(pops.validate(case), layout=layout)
    assert resolved.blocks[0].spatial.ghost_depth == 4
    assert _ghost_depth_by_block(SimpleNamespace(plan=resolved), ("material",)) == {"material": 4}
    native = ProgramModelGraph.from_resolved_blocks(resolved.blocks).model_for_block("material")._m
    assert native._program_state_ghost_depth == 4
    assert not hasattr(model._dsl._m, "_program_state_ghost_depth")


def test_inspector_still_refuses_a_plan_with_the_storage_authority_removed():
    case, layout, _ = make_case()
    resolved = pops.resolve(pops.validate(case), layout=layout)
    block = replace(resolved.blocks[0], spatial=None)
    incomplete = SimpleNamespace(blocks=(block,), field_plans={})
    with pytest.raises(ValueError, match="no exact ghost depth"):
        _ghost_depth_by_block(SimpleNamespace(plan=incomplete), (block.name,))
