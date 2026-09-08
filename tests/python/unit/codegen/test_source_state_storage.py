"""Source-only scalar state storage keeps its physical frame without fabricating flux."""
import pytest
import pops
from pops import math
from pops.domain import Rectangle
from pops.frames import Cartesian2D
from pops.numerics import StateStorage, DiscretizationPlan
from pops.layouts import Uniform
from pops.mesh import CartesianGrid, PeriodicAxes
from pops.lib.time import ForwardEuler
from pops.time import FixedDt


def source_case():
    frame = Rectangle("source-domain", lower=(0., 0.), upper=(1., 1.)).frame(Cartesian2D())
    model = pops.Model("source-model", frame=frame)
    state = model.state("U", components=("u",))
    source = model.source("injection", on=state, value=(1+0*state[0],))
    rate = model.rate("injection_rate", equation=math.ddt(state) == source)
    case = pops.Case("source-case")
    block = case.block("source", model)
    plan = DiscretizationPlan()
    plan.rates.add(rate, StateStorage())
    case.numerics(plan, block=block)
    program = ForwardEuler(block[state], rate=rate)
    program.step_strategy(FixedDt(.01))
    case.program(program)
    layout = Uniform(CartesianGrid(frame=frame, cells=(16,16), periodic=PeriodicAxes(frame.axes)))
    return case, layout, model


@pytest.mark.parametrize("canonical_module", (False, True))
def test_source_only_public_plan_and_exact_module_emit_real_storage(canonical_module):
    from pops.codegen.module_lowering import lower_and_validate
    from pops.codegen.program_codegen import emit_cpp_program
    from pops.codegen.module_codegen import _emit_bricks
    case, layout, model = source_case()
    resolved = pops.resolve(pops.validate(case), layout=layout)
    emitter, source = lower_and_validate(model.module if canonical_module else model)
    _, body, _ = _emit_bricks(emitter._m)
    assert "program_only_storage = true" in body
    assert "State flux(" not in body
    assert "max_wave_speed(" not in body
    assert emitter._m._program_only_storage_axes == ("x", "y")
    assert not emitter._m._flux and not model._dsl._m._flux
    assert source.module_hash() == model.module.module_hash()
    code = emit_cpp_program(resolved.time, model=emitter)
    assert "ctx.commit_many(" in code
    assert "ctx.rhs_into(" not in code


def test_source_storage_requires_an_authored_frame_before_emission():
    from pops.codegen.module_lowering import lower_and_validate
    model = pops.Model("unframed")
    state = model.state("U", components=("u",))
    source = model.source("injection", on=state, value=(1+0*state[0],))
    model.rate("rate", equation=math.ddt(state) == source)
    with pytest.raises(ValueError, match="authored Cartesian frame"):
        lower_and_validate(model)
