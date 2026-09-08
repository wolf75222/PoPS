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


def resolved_heat(*, n=16, coefficient=0.1, method="euler", variable=None, transport=None):
    frame=Rectangle("diffusion_domain",lower=(0.,0.),upper=(1.,1.)).frame(Cartesian2D())
    model=pops.Model("heat",frame=frame)
    state=model.state("U",components=("u",))
    expression=state if variable is None else variable(state[0])
    flux=model.diffusive_flux("conduction",state=state,value=coefficient*math.grad(expression))
    rhs=math.div(flux)
    transport_method=None
    if transport is not None:
        from pops.numerics import FiniteVolume, variables, reconstruction, riemann
        adv=model.flux("transport",frame=frame,state=state,
            components={axis:(speed*state[0],) for axis,speed in zip(frame.axes,transport,strict=True)},
            waves={axis:(speed,) for axis,speed in zip(frame.axes,transport,strict=True)})
        rhs=-math.div(adv)+rhs
        transport_method=FiniteVolume(flux=adv,variables=variables.Conservative(state),
            reconstruction=reconstruction.FirstOrder(),riemann=riemann.Rusanov())
    rate=model.rate("heat_rate",equation=math.ddt(state)==rhs)
    case=pops.Case("heat_case")
    block=case.block("heat",model,states=(state,))
    numerics=DiscretizationPlan()
    numerics.rates.add(rate,Diffusion(flux=flux,transport=transport_method))
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


@pytest.mark.parametrize("method,weights",[("euler",[1]),("ssprk2",[.5,.5])])
def test_diffusion_accepted_quadrature_does_not_count_predictor_ancestry(method,weights):
    from pops.codegen.program_diffusion_exchanges import accepted_diffusive_quadrature
    resolved,_,_=resolved_heat(method=method)
    rows=accepted_diffusive_quadrature(resolved.time)
    assert [float(weight[1]) for _,weight in rows]==weights
    assert len({value.id for value,_ in rows})==len(weights)


def test_combined_transport_retains_native_riemann_and_adds_stability_frequencies():
    from pops.codegen.module_lowering import lower_and_validate
    from pops.codegen.program_codegen import emit_cpp_program
    resolved,_,model=resolved_heat(transport=(.2,.2))
    code=emit_cpp_program(resolved.time,model=lower_and_validate(model)[0])
    assert "ctx.neg_div_flux_default_into(" in code
    assert "ctx.max_wave_speed(" in code
    assert "combined_transport_diffusion_stability" in code
    assert ".explicit_frequency() + ctx.max_wave_speed" in code
    assert code.index(".stage_accepted_exchanges(")<code.index("ctx.commit_many(")
    assert model._dsl._m._flux
