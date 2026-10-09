"""Captured numerical coefficients keep exact owner and Program block routes."""
from __future__ import annotations

import pops
import pytest

from pops.codegen.program_models import ProgramModelGraph
from pops.codegen.program_emit_params import program_param_entries
from pops.codegen.user_reconstruction_lowering import emit_user_reconstruction_policy
from pops.codegen.user_riemann_lowering import emit_user_face_policy
from pops.domain import Rectangle
from pops.frames import Cartesian2D
from pops.layouts import Uniform
from pops.math import ddt, div
from pops.mesh import CartesianGrid, PeriodicAxes
from pops.numerics import DiscretizationPlan, reconstruction, riemann, variables
from pops.numerics.spatial import FiniteVolume
from pops.params import RuntimeParam
from pops.representations import Conservative
from pops.spaces import CellState
from pops.time import FixedDt


def _case(*, foreign=False, face=False, width=1, cross=False):
    frame = Rectangle("domain", lower=(0., 0.), upper=(1., 1.)).frame(Cartesian2D())
    model = pops.Model("transport", frame=frame)
    U = model.state("U", components=tuple("q%d" % index for index in range(width)),
                    representation=Conservative(),
                    space=CellState(frame=frame))
    alpha = model.value(model.param(RuntimeParam("alpha", default=.25)))
    if foreign:
        other = pops.Model("other", frame=frame)
        alpha = other.value(other.param(RuntimeParam("alpha", default=.25)))
    F = model.flux("advection", frame=frame, state=U,
                   components={axis: tuple(U) for axis in frame.axes},
                   waves={axis: tuple(1. + 0 * item for item in U) for axis in frame.axes})
    rate = model.rate("balance", equation=ddt(U) == -div(F))
    recipe = reconstruction.User(lambda sample: sample(0) + alpha *
                                 (sample(1) - sample(-1)), formal_order=1)
    face_body = (
        (lambda left, right, flux_left, flux_right, speed:
            (flux_left[0] + alpha * right[1], flux_right[1] - left[0]))
        if cross else
        (lambda left, right, flux_left, flux_right, speed:
            .5 * (flux_left + flux_right) - .5 * speed * (right - left)
            + alpha * (right - left)))
    face_recipe = riemann.User(body=face_body, state=U,
                               stability=lambda left, right, fl, fr, speed: speed + alpha) if face else riemann.Rusanov()
    case = pops.Case("captured_numerical_coefficient")
    program = pops.Program("steps")
    for name in ("first", "second"):
        block = case.block(name, model)
        numerics = DiscretizationPlan()
        numerics.rates.add(rate, FiniteVolume(
            flux=F, variables=variables.Conservative(U), reconstruction=recipe,
            riemann=face_recipe))
        case.numerics(numerics, block=block)
        temporal = program.state(block[U])
        rhs = program.value(name + "_rhs", rate(temporal.n), at=temporal.n.point)
        next_value = program.value(name + "_next", temporal.n + program.dt * rhs,
                                   at=temporal.next.point)
        program.commit(temporal.next, next_value)
    program.step_strategy(FixedDt(1.e-3))
    case.program(program)
    layout = Uniform(CartesianGrid(frame=frame, cells=(8, 8),
                                   periodic=PeriodicAxes(frame.axes)))
    return case, layout, recipe, alpha, program, face_recipe


def test_same_runtime_capture_has_distinct_program_block_slots_and_device_read():
    case, layout, recipe, alpha, program, _ = _case()
    resolved = pops.resolve(pops.validate(case), layout=layout)
    graph = ProgramModelGraph.from_resolved_blocks(resolved.blocks)
    for name in ("first", "second"):
        emitted = emit_user_reconstruction_policy(graph.model_for_block(name))
        assert "params.get(0)" in emitted
        assert recipe.options["source_identity"] in emitted
        assert str(alpha.value) not in emitted
    entries = program_param_entries(program, graph)
    assert {(row[0], row[1]) for row in entries} == {(0, "alpha"), (1, "alpha")}


def test_foreign_model_same_named_runtime_capture_is_refused():
    case, layout, _, _, _, _ = _case(foreign=True)
    with pytest.raises(ValueError, match="another model|belongs to"):
        resolved = pops.resolve(pops.validate(case), layout=layout)
        ProgramModelGraph.from_resolved_blocks(resolved.blocks)


def test_composed_reconstruction_and_face_captures_emit_both_native_routes():
    case, layout, reconstruction_recipe, alpha, program, face_recipe = _case(face=True)
    graph = ProgramModelGraph.from_resolved_blocks(pops.resolve(pops.validate(case), layout=layout).blocks)
    entries = program_param_entries(program, graph)
    assert {(row[0], row[1]) for row in entries} == {(0, "alpha"), (1, "alpha")}
    for name in ("first", "second"):
        model = graph.model_for_block(name)
        source_face = emit_user_face_policy(model)
        assert "params.get(0)" in source_face
        assert face_recipe.options["source_identity"] in source_face
        assert str(alpha.value) not in source_face
        for target in ("system", "amr_system"):
            source = model._m.emit_cpp_native_loader(name="FaceRoute", target=target)
            assert reconstruction_recipe.options["source_identity"] in source
            assert face_recipe.options["source_identity"] in source
            assert "user_reconstruction, user_face" in source


def test_cross_component_face_formula_reaches_both_native_package_targets():
    case, layout, _, _, _, face_recipe = _case(face=True, width=2, cross=True)
    graph = ProgramModelGraph.from_resolved_blocks(pops.resolve(pops.validate(case), layout=layout).blocks)
    for name in ("first", "second"):
        model = graph.model_for_block(name)
        assert len(face_recipe.expression) == 3
        emitted = emit_user_face_policy(model)
        assert "density[1]" in emitted
        assert "right.state[1]" in emitted
        assert "left.state[0]" in emitted
        for target in ("system", "amr_system"):
            source = model._m.emit_cpp_native_loader(name="VectorFaceRoute", target=target)
            assert "struct UserFacePolicy" in source
            assert "user_reconstruction, user_face" in source


def test_authored_face_frequency_guards_the_exact_default_rhs_on_both_targets():
    from pops.codegen.program_codegen import emit_cpp_program

    case, layout, _, _, _, _ = _case(face=True, width=1)
    resolved = pops.resolve(pops.validate(case), layout=layout)
    graph = ProgramModelGraph.from_resolved_blocks(resolved.blocks)
    for target in ("system", "amr_system"):
        source = emit_cpp_program(resolved.time, model=graph, target=target)
        assert source.count("user_face_numerical_stability") == 2
        assert source.count("numerical_face_courant()") == 2
        assert source.count("&user_face_frequency_") == 2
        assert "ctx.max_wave_speed" not in source
