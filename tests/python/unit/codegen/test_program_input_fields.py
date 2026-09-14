"""Explicit runtime-input observations preserve source and stage authority."""
import pytest
import pops
from pops import math
from pops.domain import Rectangle
from pops.frames import Cartesian2D
from pops.numerics import Diffusion, DiscretizationPlan
from pops.time import Program, FixedDt
from pops.codegen import Production
from pops.layouts import Uniform
from pops.mesh import CartesianGrid, PeriodicAxes


def input_heat():
    frame = Rectangle("domain", lower=(0., 0.), upper=(1., 1.)).frame(Cartesian2D())
    model = pops.Model("variable_heat", frame=frame)
    state = model.state("U", components=("u",))
    coefficient = model.aux("a")
    flux = model.diffusive_flux("flux", state=state, value=coefficient * math.grad(state))
    rate = model.rate("rate", equation=math.ddt(state) == math.div(flux))
    case = pops.Case("case")
    block = case.block("heat", model, states=(state,))
    plan = DiscretizationPlan()
    plan.rates.add(rate, Diffusion(flux=flux))
    case.numerics(plan, block=block)
    program = Program("explicit_input_heat")
    value = program.state(block[state])
    return frame, model, rate, case, program, value


def test_runtime_inputs_resolve_and_emit_an_actual_collective_publication():
    from pops.codegen.module_lowering import lower_and_validate
    from pops.codegen.program_codegen import emit_cpp_program
    frame, model, rate, case, program, value = input_heat()
    fields = program.input_fields(value.n, for_rate=rate)
    rhs = rate(value.n, fields)
    end = program.value("next", value.n + program.dt * rhs, at=value.next.point)
    program.commit(value.next, end)
    program.step_strategy(FixedDt(1e-4))
    case.program(program)
    resolved = pops.resolve(pops.validate(case), layout=Uniform(CartesianGrid(
        frame=frame, cells=(16, 16), periodic=PeriodicAxes(frame.axes))), backend=Production())
    code = emit_cpp_program(resolved.time, model=lower_and_validate(model)[0])
    assert "ctx.prepare_provider_values(" in code
    assert ".apply(" in code
    assert "ctx.solve_fields" not in code
    assert fields.field_context.stage_sources == ((value.n.block, value.n.id),)


def test_input_observation_refuses_foreign_operator_and_stale_stage():
    _, _, rate, _, program, value = input_heat()
    _, _, foreign, _, _, _ = input_heat()
    with pytest.raises(ValueError, match="owner|registry"):
        program.input_fields(value.n, for_rate=foreign)
    fields = program.input_fields(value.n, for_rate=rate)
    first_rhs = program._call(rate, value.n, fields)
    stage = program.value("stage", value.n + program.dt * first_rhs, at=value.next.point)
    with pytest.raises(ValueError, match="field context|stage|point|TimePoint"):
        program._call(rate, stage, fields)


def test_runtime_input_observation_refuses_a_computed_provider_without_fallback():
    from pops.codegen.component_provider_packs import resolve_component_provider_packs
    from pops.model.provider_pack import ProviderPack, ProviderEntry
    from pops.time.input_fields import runtime_input_pack
    _, model, _, _, _, value = input_heat()
    space = model.module.field_spaces()["fields"]
    field = value.n.block[model.module.field_handle(space)]
    pack = resolve_component_provider_packs(model.module).auxiliary
    computed = ProviderPack((key, pack.contract(key), ProviderEntry("field_output", True, 0))
                            for key in pack)
    with pytest.raises(ValueError, match="requires runtime_input"):
        runtime_input_pack(computed, field, space)
