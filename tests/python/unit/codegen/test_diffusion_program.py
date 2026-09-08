"""Real public Case resolution and native Program generation for diffusion."""
import pytest
import pops
from pops import math
from pops.domain import Rectangle
from pops.frames import Cartesian2D
from pops.numerics import Diffusion, DiscretizationPlan
from pops.lib.time import ForwardEuler, SSPRK2
from pops.time import FixedDt
from pops.layouts import Uniform
from pops.mesh import CartesianGrid, PeriodicAxes
from pops.codegen import Production
from pops.initial import InitialCondition
from pops.lib.initial import BindArray
from pops.projection import ConservativeCellAverage


def resolved_heat(*, n=16, coefficient=0.1, method="euler", variable=None):
    frame=Rectangle("diffusion_domain",lower=(0.,0.),upper=(1.,1.)).frame(Cartesian2D())
    model=pops.Model("heat",frame=frame)
    state=model.state("U",components=("u",))
    expression=state if variable is None else variable(state[0])
    flux=model.diffusive_flux("conduction",state=state,value=coefficient*math.grad(expression))
    rate=model.rate("heat_rate",equation=math.ddt(state)==math.div(flux))
    case=pops.Case("heat_case")
    block=case.block("heat",model,states=(state,))
    numerics=DiscretizationPlan()
    numerics.rates.add(rate,Diffusion(flux=flux))
    case.numerics(numerics,block=block)
    program=(ForwardEuler if method=="euler" else SSPRK2)(block[state],rate=rate)
    program.step_strategy(FixedDt(0.1/(coefficient*n*n)))
    case.program(program)
    case.initials.add(InitialCondition(state=block[state],value=BindArray(),
                                      projection=ConservativeCellAverage()))
    resolved=pops.resolve(pops.validate(case),layout=Uniform(CartesianGrid(
        frame=frame,cells=(n,n),periodic=PeriodicAxes(frame.axes))),backend=Production())
    return resolved,block[state],model


@pytest.mark.parametrize("method",["euler","ssprk2"])
def test_scalar_diffusion_public_case_resolves_and_emits_native_faces(method):
    from pops.codegen.program_codegen import emit_cpp_program
    resolved,_,model=resolved_heat(method=method)
    from pops.codegen.module_lowering import lower_and_validate
    emitter,_=lower_and_validate(model)
    code=emit_cpp_program(resolved.time,model=emitter)
    assert "PreparedDiffusion<pops::kNativeDimension>" in code
    assert ".apply(" in code
    assert "ctx.rhs_into(" not in code
    assert not model._dsl._m._flux
    assert all(row.refusal is None for plan in resolved.resolved_operations.values() for row in plan.operations)
