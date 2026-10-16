"""AD removes only syntactically proved false switching surfaces, never a domain guard."""
import json
import math
import pytest
from pops.graph_functions import GraphFunction, graph
from pops.graph_derivatives import _ComparisonProof, _Forward, jvp
from pops.time import Program
from pops.time.variant_types import Bool, Float64, Int64, Product, Result, Failure


def _comparisons(function):
    proof = _ComparisonProof(function.to_data())
    return [(record, proof.invariant(record)) for record in proof.records.values() if record['op']=='compare']


def _guards(function, monkeypatch):
    guarded=[]; original=_Forward.checked_comparison
    def record(self, condition):
        guarded.append(condition)
        return original(self, condition)
    monkeypatch.setattr(_Forward, 'checked_comparison', record)
    differentiated=jvp(GraphFunction.from_data(json.loads(json.dumps(function.to_data()))))
    GraphFunction.from_data(json.loads(json.dumps(differentiated.to_data())))
    return guarded


@pytest.mark.parametrize('constant', (0., -0., 1.))
def test_constant_comparison_at_equality_has_no_new_domain(constant, monkeypatch):
    @graph(inputs=Product(x=Float64), output=Float64)
    def f(p,a):
        return p.branch(p.float64(constant)>=constant,lambda q:a['x']*a['x'],lambda q:-a['x'])
    assert [invariant for _,invariant in _comparisons(f)] == [True]
    assert not _guards(f,monkeypatch)


@pytest.mark.parametrize('change', (False, True))
def test_equation_identity_through_guarded_product_is_syntactic(change, monkeypatch):
    error=Failure(test=Product(code=Int64))
    @graph(inputs=Product(x=Float64,b=Bool),output=Float64)
    def f(p,a):
        record=p.product(value=a['x'])
        t=Result(Product(value=Float64),error)
        packet=p.require_success(p.branch(a['b'],lambda q:q.variant(t,'ok',record),
            lambda q:q.variant(t,'err',q.variant(error,'test',q.product(code=q.int64(73))))))
        x=packet['value']; y=a['x']
        same=(x+x*x)==(y+y*y+(1. if change else 0.)) if change else (x+x*x)==(y+y*y)
        return p.branch(same,lambda q:x,lambda q:-x)
    assert [v for _,v in _comparisons(f)] == [not change]
    assert len(_guards(f,monkeypatch)) == int(change)


@pytest.mark.parametrize('mode', ('zero_direction','stationary_expression','hidden_jump','absolute_value'))
def test_actual_continuous_surface_still_has_differential_guard(mode,monkeypatch):
    @graph(inputs=Product(x=Float64),output=Float64)
    def f(p,a):
        x=a['x']
        if mode=='stationary_expression':
            return p.branch(x*x>=0,lambda q:x,lambda q:-x)
        if mode=='hidden_jump':
            sign=p.branch(x>=0,lambda q:q.float64(1),lambda q:q.float64(-1))
            return p.branch(sign>0,lambda q:x,lambda q:-x)
        if mode=='absolute_value':
            return p.branch(x>=0,lambda q:x,lambda q:-x)
        return p.branch(x>=0,lambda q:x*x,lambda q:-x)
    guards=_guards(f,monkeypatch)
    assert len(guards)==(2 if mode=='hidden_jump' else 1)
    if mode=='zero_direction':
        # Supplying zero is not a certificate of an open differentiability domain.
        p=Program('zero_direction'); d=jvp(f)
        out=d(p,primal=p.product(x=p.float64(0)),tangent=p.product(x=p.float64(0)))
        assert out.vtype=='product'
        assert any(node['op']=='branch' for node in d.to_data()['nodes'])


def test_integer_cast_keeps_the_real_guard_that_selected_the_discrete_value(monkeypatch):
    @graph(inputs=Product(x=Float64),output=Float64)
    def f(p,a):
        count=p.branch(a['x']>=0,lambda q:q.int64(1),lambda q:q.int64(0))
        chosen=p.float64(count)
        return p.branch(chosen>=0,lambda q:a['x']*2,lambda q:-a['x'])
    assert [v for _,v in _comparisons(f)] == [False,True]
    assert len(_guards(f,monkeypatch))==1


def test_different_invocations_and_loop_outputs_are_not_equal_by_body():
    square=graph(inputs=Product(x=Float64),output=Float64)(lambda p,a:a['x']*a['x'])
    @graph(inputs=Product(x=Float64,y=Float64),output=Bool)
    def f(p,a):
        left=square(p,x=a['x']); right=square(p,x=a['y'])
        return left==right
    assert [v for _,v in _comparisons(f)]==[False]
    @graph(inputs=Product(x=Float64),output=Bool)
    def loop(p,a):
        next_value=p.fold(a['x'],p.int64(2),lambda q,z,i:z*z,max_steps=2)
        return next_value==a['x']
    assert [v for _,v in _comparisons(loop)]==[False]


@pytest.mark.parametrize('error,limit', ((0.,0.),(-0.,0.),(0.,-0.),(1.,1.),(-1.,1.),
    (math.nextafter(1.,0.),1.),(math.nextafter(1.,math.inf),1.),
    (math.nextafter(-1.,0.),1.),(math.nextafter(-1.,-math.inf),1.),
    (1e308,1e308),(-1e308,1e308),(0.,-1.),(math.nan,1.),(0.,math.nan),
    (math.inf,1.),(-math.inf,1.),(math.inf,math.inf),(-math.inf,math.inf)))
def test_residual_interval_predicate_is_ieee_equivalent(error,limit):
    # Nonfinite values are rejected by the unchanged earlier guards. They are
    # included here to expose accidental Python/IEEE predicate differences.
    assert (abs(error)<=limit) == (-limit<=error and error<=limit)


def test_complete_inventory_conversion_keeps_real_selection_and_residual_boundaries(monkeypatch):
    from pops.contributions import FiniteInterval
    from examples.api_v02.inventory_publication_v15 import _nonlinear
    contribution=Product(value=Float64,interval=Product(begin=Float64,end=Float64))
    output=Product(q=Float64,encoded=Float64,target=Float64,defect=Float64,delta=Float64,
        increment=Product(J=contribution,conversion_residual=contribution))
    @graph(inputs=Product(q=Float64,time=Float64,h=Float64),output=output)
    def conversion(p,a):
        return _nonlinear(p,p.product(q=a['q'],step=p.int64(0)),
            FiniteInterval(a['time'],a['time']+a['h'],'physical'),'nonlinear_exact')
    comparisons=_comparisons(conversion)
    assert len(comparisons)==10
    assert sum(invariant for _,invariant in comparisons)==3
    guarded=_guards(conversion,monkeypatch)
    assert len(guarded)==7
    # The equality check remains in primal IR, but its synthetic AD guard is gone.
    identities=[r for r,v in comparisons if v and _ComparisonProof.data(r['attrs'])['cmp']=='==']
    assert len(identities)==1
    assert sum(condition.attrs['cmp']=='<=' for condition in guarded)==2
