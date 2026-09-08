"""Physical pair ownership and public resolution of fitted drift/diffusion."""
import pytest
import pops
from pops import math
from pops.domain import CartesianDomain
from pops.frames import Cartesian1D
from pops.numerics import ScharfetterGummel, Diffusion, DiscretizationPlan
from pops.layouts import Uniform
from pops.mesh import CartesianGrid, PeriodicAxes
from pops.codegen import Production
from pops.time import FixedDt, Program


def fitted_case(*,n=16,diffusivity=.1,mobility=.1,potential=None):
    frame=CartesianDomain("fitted_domain",(0.,),(1.,)).frame(Cartesian1D())
    model=pops.Model("fitted_model",frame=frame)
    state=model.state("U",components=("n",))
    phi=model.aux("potential") if potential is None else potential
    drift=model.drift_flux("drift",state=state,mobility=mobility,potential=phi)
    diffusion=model.diffusive_flux("diffusion",state=state,value=diffusivity*math.grad(state))
    rate=model.rate("fitted_rate",equation=math.ddt(state)==-math.div(drift)+math.div(diffusion))
    method=ScharfetterGummel(drift=drift,flux=diffusion)
    case=pops.Case("fitted_case")
    block=case.block("density",model,states=(state,))
    numerics=DiscretizationPlan()
    numerics.rates.add(rate,method)
    case.numerics(numerics,block=block)
    program=Program("ScharfetterGummelEuler")
    value=program.state(block[state])
    observation=program.input_fields(value.n,for_rate=rate)
    rhs=rate(value.n,observation)
    end=program.value("next",value.n+program.dt*rhs,at=value.next.point)
    program.commit(value.next,end)
    program.step_strategy(FixedDt(.05/(diffusivity*n*n)))
    case.program(program)
    resolved=pops.resolve(pops.validate(case),layout=Uniform(CartesianGrid(
        frame=frame,cells=(n,),periodic=PeriodicAxes(frame.axes))),backend=Production())
    return resolved,model,state,drift,diffusion,rate,method


def test_fitted_public_case_preserves_two_physical_occurrences_and_one_face_flux():
    from pops.codegen.module_lowering import lower_and_validate
    from pops.codegen.program_codegen import emit_cpp_program
    resolved,model,_,_,_,rate,_=fitted_case()
    view=model.balance_contract(rate)
    assert [(row.kind,row.coefficient) for row in view.occurrences]==[("drift",-1),("diffusion",1)]
    emitter=lower_and_validate(model)[0]
    code=emit_cpp_program(resolved.time,model=emitter)
    assert ".apply_fitted(" in code
    assert code.count(".stage_accepted_exchanges(")==1
    assert "joint-occurrences:0,1" in code
    assert "ctx.neg_div_flux_default_into(" not in code
    assert not model._dsl._m._flux


def test_fitted_selection_refuses_split_or_repeated_occurrences():
    _,model,_,drift,diffusion,rate,method=fitted_case()
    full=model.balance_contract(rate)
    with pytest.raises(ValueError,match="together exactly once"):
        method.validate_balance_view(full.select(drift))
    with pytest.raises(ValueError,match="together exactly once"):
        method.validate_balance_view(full.select(diffusion))
    with pytest.raises(ValueError,match="diffusion/source"):
        Diffusion(flux=diffusion).validate_balance_view(full)


def test_fitted_pair_rejects_foreign_drift_even_with_matching_names():
    *_,foreign_drift,_,_,_=fitted_case()
    _,_,_,_,diffusion,_,_=fitted_case()
    with pytest.raises(ValueError,match="exact Dim1 scalar state"):
        ScharfetterGummel(drift=foreign_drift,flux=diffusion)
