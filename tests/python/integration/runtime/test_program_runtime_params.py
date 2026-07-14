"""Runtime parameters remain qualified through Program codegen and BindSchema."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

import pops.lib.time as libtime
from pops.codegen.program_codegen import emit_cpp_program
from pops.codegen.program_emit_params import program_param_entries
from pops.domain import Rectangle
from pops.frames import Cartesian2D
from pops.math import ddt, div
from pops.model.bind_schema import BindSchema
from pops.params import ConstParam, RuntimeParam
from pops.physics import Model
from pops.problem import Case
from pops.runtime._install_param_routing import route_program_params
from tests.python.integration._final_field_program import compiler_model


def _authoring(declaration, *, name: str):
    frame = Rectangle(
        name + "-domain", lower=(0.0, 0.0), upper=(1.0, 1.0)
    ).frame(Cartesian2D())
    x_axis, y_axis = frame.axes
    model = Model(name + "-model", frame=frame)
    state = model.state("U", components=("rho",))
    (rho,) = state
    parameter = model.param(declaration)
    value = model.value(parameter)
    flux = model.flux(
        "transport",
        frame=frame,
        state=state,
        components={x_axis: (0.0 * rho,), y_axis: (0.0 * rho,)},
        waves={x_axis: (0.0 * rho,), y_axis: (0.0 * rho,)},
    )
    source = model.source("decay", on=state, value=(value * rho,))
    rate = model.rate(
        "explicit_rhs", equation=ddt(state) == -div(flux) + source)

    case = Case(name + "-case")
    block = case.block("gas", model)
    program = libtime.ForwardEuler(block[state], rate=rate)
    case.program(program)
    return model, parameter, case, block, program


def test_runtime_parameter_is_a_program_slot_while_const_parameter_is_inlined() -> None:
    runtime_model, _, _, _, runtime_program = _authoring(
        RuntimeParam("k", default=2.0), name="runtime-decay")
    runtime_source = emit_cpp_program(
        runtime_program, model=compiler_model(runtime_model))

    assert "ctx.program_params(0)" in runtime_source
    assert "params.get(0)" in runtime_source
    assert "pops_program_param_count() { return 1; }" in runtime_source
    assert program_param_entries(
        runtime_program, compiler_model(runtime_model)
    ) == [(0, "k", 0, 2.0)]

    const_model, _, _, _, const_program = _authoring(
        ConstParam("k", 2.0), name="const-decay")
    const_source = emit_cpp_program(
        const_program, model=compiler_model(const_model))
    assert "params.get(" not in const_source
    assert "ctx.program_params(" not in const_source
    assert "pops_program_param_count() { return 0; }" in const_source


def test_bind_schema_routes_only_owner_qualified_parameter_handles() -> None:
    model, parameter, case, block, program = _authoring(
        RuntimeParam("k", default=2.0), name="qualified-decay")
    schema = BindSchema.from_problem(case)
    compile_values = schema.resolve_compile()
    default_values = schema.resolve_bind({}, compile_values=compile_values)
    override_values = schema.resolve_bind(
        {block[parameter]: 7.0}, compile_values=compile_values)
    carrier = SimpleNamespace(
        program=program,
        program_block_routes=((0, "gas"),),
        program_param_routes=tuple(
            program_param_entries(program, compiler_model(model))),
    )

    assert route_program_params(carrier, schema, default_values) == {0: [2.0]}
    assert route_program_params(carrier, schema, override_values) == {0: [7.0]}
    with pytest.raises(TypeError):
        schema.resolve_bind({"k": 9.0}, compile_values=compile_values)
