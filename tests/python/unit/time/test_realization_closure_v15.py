"""Complete Realize-body closure, including unused partial effects; source only."""
import pytest
import pops.math as pm
from pops.frames import Cartesian2D
from pops.physics import Model
from pops.problem import Case
from pops.realization import Realize, Use
from pops.time import Program, FixedDt
from pops.time.expressions import ProgramExpression
from pops._ir.quantity import PhysicalDimension, PhysicalSupport
from pops.model.spaces import Space
from pops.graph_functions import graph
from pops.representations import ScientificMap, ScientificType, ScientificValue
from pops.time.variant_types import Float64, Product


def fixture(strict=False):
    model=Model("realize_closure",frame=Cartesian2D())
    state=model.state("U",components=("x","y"))
    source=model.source("source",on=state,value=(-state[0],state[0]))
    rate=model.rate("balance",equation=pm.ddt(state)==source)
    case=Case(name="realize_case");block=case.block("material",model,states=(state,))
    p=Program("realize_program");q=p.state(block[state])
    if strict:
        space=Space("scalar",("q",),support=PhysicalSupport(()),units=(PhysicalDimension(),),
                    value_shape=(),sampling="point",representation="coordinates",domain="real")
        typ=ScientificType(Float64,space,"point_value",(),(),{"name":"counting"})
        incoming=ScientificValue(p.float64(2),typ)
        identity=graph(inputs=Product(q=Float64),output=Float64)(lambda g,a:a["q"])
        mapping=ScientificMap(identity,inputs={"q":typ},output=typ,strict=True)
        output=lambda g,value:{"out":mapping(g,q=value)}
    else:
        typ=None;incoming=q.n;output=lambda g,value:{"out":(value[0],value[1])}
    use=Use(rate.occurrences[0],"cell","stage","inventory",scientific_type=typ)
    return model,p,q,use,incoming,output


def recipe(use, incoming, body, *, strict=False, extra=None):
    return Realize("closure",owns=(use,),bindings={"q":incoming,**(extra or {})},
                   body=body,outputs={use:("out",)},strict=strict)


def emit(model,p):
    from pops.codegen.module_lowering import lower_and_validate
    from pops.codegen.program_codegen import emit_cpp_program
    lowered,_=lower_and_validate(model,facade=model)
    return emit_cpp_program(p,model=lowered)


def unused(p,hidden,kind):
    if kind=="direct": pm.log(hidden)
    elif kind=="fold": p.fold(p.float64(1),2,lambda g,x,k:pm.log(hidden)+x)
    elif kind=="branch": p.branch(p.boolean(True),lambda g:pm.log(hidden),lambda g:g.float64(0))
    elif kind=="capture": p.capture(lambda g:pm.log(hidden))
    else: raise AssertionError(kind)


@pytest.mark.parametrize("strict",(False,True))
@pytest.mark.parametrize("kind",("direct","fold","branch","capture"))
def test_unused_partial_hidden_capture_is_rejected_atomically(strict,kind):
    _,p,q,use,incoming,output=fixture(strict)
    hidden=p.requested_dt();before=p._ir_hash();ids=tuple(p._issued_values);counter=p._next_id
    def body(g,q):
        unused(g,hidden,kind)
        return output(g,q)
    with pytest.raises(ValueError,match="undeclared temporal binding"):
        recipe(use,incoming,body,strict=strict).evaluate(p)
    assert p._ir_hash()==before and tuple(p._issued_values)==ids and p._next_id==counter


@pytest.mark.parametrize("kind",("direct","fold","branch","capture"))
def test_explicit_unused_partial_capture_remains_in_generated_program(kind):
    model,p,q,use,incoming,output=fixture()
    duration=p.requested_dt()
    def body(g,q,duration):
        unused(g,duration,kind)
        return output(g,q)
    value=recipe(use,incoming,body,extra={"duration":duration}).evaluate(p)[use]
    p.commit(q.next,p.value("endpoint",value,at=q.next.point))
    assert p.validate()
    assert "log(" in emit(model,p)


def test_new_ambient_native_input_is_rejected_even_if_unused():
    _,p,q,use,incoming,output=fixture();before=p._ir_hash()
    def body(g,q):
        pm.log(g.requested_dt())
        return output(g,q)
    with pytest.raises(ValueError,match="undeclared native input"):
        recipe(use,incoming,body).evaluate(p)
    assert p._ir_hash()==before


@pytest.mark.parametrize("effect",("step_policy","commit"))
def test_body_cannot_change_root_policy_or_publication(effect):
    _,p,q,use,incoming,output=fixture();p.step_strategy(FixedDt(.1));before=p._ir_hash()
    prior=p._step_strategy
    def body(g,old):
        if effect=="step_policy":g.step_strategy(FixedDt(.2))
        else:g.commit(q.next,g.value("unexpected",old,at=q.next.point))
        return output(g,old)
    realization=Realize("root_effect",owns=(use,),bindings={"old":incoming},body=body,outputs={use:("out",)})
    with pytest.raises(ValueError,match="policy, publication, or declarations"):
        realization.evaluate(p)
    assert p._ir_hash()==before and p._step_strategy is prior and not p._commits


def test_strict_explicit_capture_and_nested_region_arguments_are_valid():
    _,p,q,use,incoming,output=fixture(True);duration=p.requested_dt()
    def body(g,q,duration):
        g.fold(g.product(value=duration),2,
               lambda b,carried,index:b.product(value=carried["value"]+duration))
        g.product()  # A literal empty Product is not an undeclared native input.
        return output(g,q)
    result=recipe(use,incoming,body,strict=True,extra={"duration":duration}).evaluate(p)[use]
    assert result.prog is p


def test_unused_distributed_local_rank_is_scoped_by_the_declared_domain():
    from pops.time._program.distributed_arrays import Distributed
    _,p,q,use,incoming,output=fixture()
    distributed=Distributed(p)
    domain=distributed.domain((p.int64(0),p.int64(2)),capacity=2)
    def body(g,q,domain):
        Distributed(g).local_map(domain,
            lambda b,i:b.capture(lambda c:c.float64(Distributed(c).rank())),byte_capacity=4096)
        return output(g,q)
    result=recipe(use,incoming,body,extra={"domain":domain}).evaluate(p)[use]
    assert result.prog is p


def test_unused_ambient_dt_polynomial_is_rejected_atomically():
    _,p,q,use,incoming,output=fixture();before=p._ir_hash()
    def body(g,q):
        g.value("unused_dt",g.dt*q,at=q.point)
        return output(g,q)
    with pytest.raises(ValueError,match="ambient dt; pass a scalar duration binding"):
        recipe(use,incoming,body).evaluate(p)
    assert p._ir_hash()==before


def test_new_bounded_domain_map_and_fold_are_legitimate_authoring_allocations():
    from pops.time._program.finite_arrays import IndexFamily
    model,p,q,use,incoming,output=fixture()
    def body(g,q):
        family=IndexFamily(capacity=3,ownership="replicated")
        domain=g.range_set(3,family=family)
        values=g.map(domain,lambda b,i:b.float64(b.index_value(i)))
        g.fold(values,2,lambda b,old,k:b.map(b.domain(old),lambda c,i:old[i]+1.))
        g.value("unused_constant_coefficient",2*q)
        return output(g,q)
    result=recipe(use,incoming,body).evaluate(p)[use]
    p.commit(q.next,p.value("accepted",result,at=q.next.point))
    assert p.validate()
    source=emit(model,p)
    # Both maps are device kernels; the fold carrying an Array sequences its iterations (DM7).
    assert source.count('Kokkos::parallel_for("pops_finite_map_')==2
    assert "for (; iteration_" in source
