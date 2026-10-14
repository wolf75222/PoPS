"""Occurrence-preserving composition emits actual path and source operations."""
import pytest
import pops
from pops.math import ddt,div
from pops.numerics import DiscretizationPlan,StateStorage,reconstruction,riemann,variables
from pops.numerics.nonconservative import PathConservativeFiniteVolume
from pops.numerics.symbolic_path import SymbolicPath
from tests.python.unit.physics.test_nonconservative_product_contract import _model


def declarations(*,repeat=False):
    model,state,flux,product=_model()
    source=model.source("decay",on=state,value=(0.,-.3*state[1]))
    equation=-div(flux)-product+source
    if repeat:equation=equation+source
    balance=model.rate("with_relaxation",equation=ddt(state)==equation)
    transport=balance.select(flux,product)
    sources=tuple(balance.select(row) for row in balance.occurrences if row.kind=="source")
    return model,state,flux,product,balance,transport,sources


def assembled(*,repeat=False):
    model,state,flux,product,balance,transport,sources=declarations(repeat=repeat)
    path=SymbolicPath(product,frame=model.frame,quadrature=((.5,1.),),speed=lambda a,b,axis:16.)
    method=PathConservativeFiniteVolume(flux=flux,path=path,variables=variables.Conservative(state),
        reconstruction=reconstruction.FirstOrder(),riemann=riemann.Rusanov())
    case=pops.Case("partitioned_balance");block=case.block("material",model)
    plan=DiscretizationPlan();plan.rates.add(transport,method)
    for source in sources:plan.rates.add(source,StateStorage())
    case.numerics(plan,block=block)
    p=pops.Program("compose_physical_occurrences");q=p.state(block[state])
    rate=block[balance](q.n)
    stage=p.value("predictor",q.n+p.dt*rate,at=q.next.point)
    second=block[balance](stage)
    final=p.value("corrector",q.n+p.dt/2*(rate+second),at=q.next.point)
    p.commit(q.next,final);p.step_strategy(pops.time.FixedDt(.001));case.program(p)
    return case,model,p,rate,second


@pytest.mark.parametrize("repeat",(False,True))
def test_composite_root_retains_every_occurrence_and_emits_both_routes(repeat):
    from pops.layouts import Uniform
    from pops.mesh import CartesianGrid,PeriodicAxes
    from pops.codegen.program_codegen import emit_cpp_program
    from pops.codegen.program_models import ProgramModelGraph
    case,model,p,rate,second=assembled(repeat=repeat)
    assert rate.op=="linear_combine" and rate.vtype=="rhs"
    assert len(rate.inputs)==(3 if repeat else 2)
    assert rate.inputs[0].attrs["path_conservative"]
    assert len(rate.attrs["balance_partitions"])==len(rate.inputs)
    assert rate.inputs[0].inputs[0].id!=second.inputs[0].inputs[0].id
    layout=Uniform(CartesianGrid(frame=model.frame,cells=(4,4),periodic=PeriodicAxes(model.frame.axes)))
    resolved=pops.resolve(pops.validate(case),layout=layout)
    cpp=emit_cpp_program(resolved.time,model_graph=ProgramModelGraph.from_resolved_blocks(resolved.blocks))
    assert "user_face_numerical_stability" in cpp
    assert "rhs_into" in cpp and "pops_shared_source_" in cpp
    from pops.codegen._resolved_operation_inputs import native_route
    declaration=model.module.operator_registry().get("with_relaxation")
    assert native_route(model.module,declaration)==("program:balance_composition",None)


def test_incomplete_or_overlapping_views_cannot_fall_through_default_rhs():
    for overlap in (False,True):
        model,state,flux,product,balance,transport,sources=declarations(repeat=overlap)
        if overlap:
            balance.select(*(row for row in balance.occurrences if row.kind=="source"))
        else:
            # A separate root has no registered source partition.
            missing=model.rate("missing_partition",equation=ddt(state)==-div(flux)-product+
                model.source("other_decay",on=state,value=(0.,-state[1])))
            missing.select(flux,product);balance=missing
        case=pops.Case("unrealized_balance");block=case.block("material",model)
        p=pops.Program("refuse_fallback");q=p.state(block[state])
        with pytest.raises(ValueError,match="exact disjoint registered partition"):
            balance(q.n)


def test_declared_partition_does_not_relax_numerical_exact_once_coverage():
    from tests.python.unit.numerics.test_balance_partition_coverage import _plan
    model,_,_,_,balance,transport,sources=declarations()
    with pytest.raises(ValueError,match="missing=.*occurrences"):
        _plan(transport).validate_for(model)
    with pytest.raises(ValueError,match="duplicate physical balance occurrence"):
        _plan(balance,transport,*sources).validate_for(model)
