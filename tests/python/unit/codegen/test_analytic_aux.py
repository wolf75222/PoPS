"""Geometry-owned auxiliaries retain frame and re-materialization authority."""
import pytest

from pops.analytic import CellBounds, cos, coordinate, input, sin
from pops.codegen._compile_emit import _emit_auxiliary_route_registration
from pops.codegen.component_provider_packs import resolve_component_provider_packs
from pops.codegen.module_lowering import _module_to_model
from pops.domain import Rectangle
from pops.fields import AnalyticAux, AuxiliaryBoundary
from pops.frames import Cartesian2D
from pops.model import Module


def test_analytic_aux_uses_exact_level_geometry_and_no_transferred_input():
    frame = Rectangle("mapped disk", lower=(0., 0.), upper=(16., 6.283185307179586)).frame(Cartesian2D())
    radial, angular = frame.axes
    bounds = CellBounds(frame)
    theta = coordinate(frame, angular)
    width = bounds.upper(angular) - bounds.lower(angular)
    value = cos(theta) * sin(width / 2.) / (width / 2.)
    module = Module("geometric-coefficient")
    module.state_space("U", ("q",), frame=frame.canonical_id)
    target = module.aux_handle(module.aux_field("radial_cos", frame=frame.canonical_id))
    producer = AnalyticAux(target, value, frame=frame,
                           boundary=AuxiliaryBoundary(width=2, kind="foextrap"))
    module.aux_provider(producer)
    packs = resolve_component_provider_packs(module)
    (route,) = packs.auxiliary_routes.values()
    assert route["dependencies"] == ()
    assert producer.restart_policy == producer.regrid_policy == "recompute"
    carrier = _module_to_model(module)._m
    source = _emit_auxiliary_route_registration(carrier, target="amr_system")
    assert "context.storage.geometry == nullptr" in source
    assert "geometry.cell_coordinate(1, index[1])" in source
    assert "geometry.face_coordinate(1, index[1] + 1)" in source
    assert "geometry.lower()[0]" in source
    assert "Kokkos::parallel_reduce" in source
    assert "AuxiliaryFreshness::evaluation" in source
    assert "std::vector<Dependency>{}" in source


def test_analytic_aux_refuses_foreign_frame_and_unauthenticated_bounds():
    frame = Rectangle("one", lower=(0., 0.), upper=(1., 1.)).frame(Cartesian2D())
    other = Rectangle("two", lower=(0., 0.), upper=(2., 1.)).frame(Cartesian2D())
    module = Module("geometric-coefficient")
    target = module.aux_handle(module.aux_field("x", frame=frame.canonical_id))
    with pytest.raises(ValueError, match="another frame"):
        AnalyticAux(target, coordinate(other, other.axes[0]), frame=frame)
    with pytest.raises(ValueError, match="unauthenticated"):
        AnalyticAux(target, input(0, "invented"), frame=frame)
    with pytest.raises(ValueError, match="target auxiliary space"):
        module.aux_provider(AnalyticAux(target, coordinate(other, other.axes[0]), frame=other))


def test_blackboard_retains_typed_auxiliary_without_fabricating_input_field():
    import pops
    from pops.math import Var
    from pops.codegen.module_lowering import lower_and_validate

    frame = Rectangle("metric", lower=(0., 0.), upper=(1., 1.)).frame(Cartesian2D())
    model = pops.Model("metric transport", frame=frame)
    state = model.state("U", components=("q",))
    speed = Var("speed", "aux")
    flux = model.flux("transport", frame=frame, state=state,
        components={frame.axes[0]: (speed * state[0],), frame.axes[1]: (0 * state[0],)},
        waves={frame.axes[0]: (speed,), frame.axes[1]: (0 * state[0],)})
    module = model.module
    target = module.aux_handle(module.aux_field("speed", frame=frame.canonical_id))
    module.aux_provider(AnalyticAux(target, coordinate(frame, frame.axes[0]), frame=frame))
    module.operator_registry().get(flux.reg_name).requirements["aux"] = ("speed",)
    original_hash = module.module_hash()
    emitter, retained = lower_and_validate(model)
    assert retained is module and retained.module_hash() == original_hash
    assert all(not space.components for space in module.field_spaces().values())
    assert emitter.check()
    assert emitter._m._provider_components == ["speed"]
    assert model._dsl._m._provider_components == []
