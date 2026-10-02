"""Actual public path parameter route and strict phase/foreign-owner checks."""
import pops
import pytest
from pops.codegen.program_emit_params import program_param_entries,_qualified_param_identity
from pops.codegen.program_models import ProgramModelGraph
from pops._ir.values import RuntimeParamRef
from pops.moments import CartesianMonomialBasis
from pops.moments.polynomial_path import NormalizedPathInputs,EndpointPathInputs,endpoint_polynomial_path
from pops.numerics import NormalizedPolynomialPath,DiscretizationPlan,PathConservativeFiniteVolume,reconstruction,riemann,variables
from pops.params import RuntimeParam
from pops.math import ddt,div
from pops.layouts import Uniform
from pops.mesh import CartesianGrid,PeriodicAxes
from pops.lib.time import SSPRK2
from pops.time import FixedDt
from tests.python.support.physics_roles import FRAME


def public_case():
    basis=CartesianMonomialBasis(((1,1),(0,0),(0,2),(1,0),(2,0),(0,1)))
    model=pops.Model('parametric_common_path',frame=FRAME)
    state=model.state('moments',components=tuple('q'+str(k) for k in range(6)))
    parameter=model.param(RuntimeParam('gain',default=1.))
    gain=model.value(parameter);rho=state[basis.index((0,0))]
    x,y=FRAME.axes;directions={x:(gain,0),y:(0,0)}
    fluxes={x:tuple(gain*value for value in state),y:tuple(0*value for value in state)}
    matrices={axis:tuple(tuple(g[0]*rho if i==j else 0 for j in range(6)) for i in range(6)) for axis,g in directions.items()}
    flux=model.flux('transport',state=state,frame=FRAME,components=fluxes)
    product=model.nonconservative_product('differential_product',state=state,matrices=matrices)
    endpoint,pair=NormalizedPathInputs(basis),EndpointPathInputs(basis)
    plan=endpoint_polynomial_path(basis,
        flux=tuple(endpoint.direction(0)*endpoint.raw(index) for index in basis.indices),
        integral=tuple(pair.direction(0)*(pair.left((0,0))+pair.right((0,0)))/2*(pair.right(index)-pair.left(index)) for index in basis.indices),
        speed=abs(endpoint.direction(0))*(1+endpoint.density))
    path=NormalizedPolynomialPath(product,frame=FRAME,covectors=directions,plan=plan,
        flux=model.module.operator_registry().get(flux.reg_name).body,matrices=product.law.matrices)
    rate=model.rate('balance',equation=ddt(state)==-div(flux)-product)
    numerics=DiscretizationPlan();numerics.rates.add(rate,PathConservativeFiniteVolume(flux=flux,path=path,
        variables=variables.Conservative(state),reconstruction=reconstruction.FirstOrder(),riemann=riemann.Rusanov()))
    case=pops.Case('parameter_phase_case');block=case.block('transport',model);case.numerics(numerics,block=block)
    program=SSPRK2(block[state],rate=rate);program.step_strategy(FixedDt(1e-4));case.program(program)
    layout=Uniform(CartesianGrid(frame=FRAME,cells=(8,4),periodic=PeriodicAxes(FRAME.axes)))
    return case,layout,model,parameter,block


def test_public_parametric_path_routes_exact_resolved_parameter():
    case,layout,model,parameter,block=public_case()
    resolved=pops.resolve(pops.validate(case),layout=layout)
    graph=ProgramModelGraph.from_resolved_blocks(resolved.blocks)
    assert program_param_entries(resolved.time,graph)==[(0,'gain',0,1.)]


def test_authored_and_resolved_declaration_reads_share_identity_without_mutation():
    case,layout,model,parameter,block=public_case()
    resolved=pops.resolve(pops.validate(case),layout=layout)
    method=resolved.blocks[0].numerics.rates[0].method
    instance=method.path.covectors[0][0]
    authored=model.value(parameter)
    assert authored.handle is parameter and not parameter.is_resolved
    # Exact block authority is the parameter's resolved instance, never first model.
    owner=instance.handle.block_ref
    assert _qualified_param_identity(authored,owner,graph_aware=True)==_qualified_param_identity(instance,owner,graph_aware=True)
    assert authored.handle is parameter and not parameter.is_resolved


def test_same_spelling_foreign_parameter_stays_refused():
    case,layout,model,parameter,block=public_case()
    resolved=pops.resolve(pops.validate(case),layout=layout)
    owner=resolved.blocks[0].numerics.rates[0].method.path.covectors[0][0].handle.block_ref
    foreign=pops.Model('foreign_parameter_owner',frame=FRAME)
    handle=foreign.param(RuntimeParam('gain',default=1.))
    foreign.state('foreign_state',components=('s',))
    foreign.module.module_hash()
    with pytest.raises(ValueError,match='belongs to model owner'):
        _qualified_param_identity(foreign.value(handle),owner,graph_aware=True)
