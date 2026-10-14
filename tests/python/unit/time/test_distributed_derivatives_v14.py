"""Source-only distributed AD closure; real MPI reception is a separate owner task."""
import pytest
from pops.graph_functions import graph
from pops.graph_derivatives import jvp
from pops.time import Program
from pops.time._program.distributed_arrays import Distributed
from pops.time.variant_types import Float64, Product, type_of


def _array():
    p=Program('distributed_ad_types');d=Distributed(p)
    domain=d.domain((0,1,1,3),capacity=8)
    values=p.map(domain,lambda q,i:q.float64(q.index_value(i)+1),byte_capacity=16384)
    return p,domain,values


@pytest.mark.parametrize('mode',('map_reduce','import','scatter','scan'))
def test_distributed_forward_graph(mode):
    p,domain,prototype=_array()
    @graph(inputs=Product(x=type_of(prototype)),output=Float64)
    def f(g,a):
        x=a['x'];D=g.domain(x)
        if mode=='import':
            req=Distributed(g).requests(D,D,lambda b,i:b.index(D,(b.index_value(i)+1)%b.size(D)))
            x=g.gather(x,req,byte_capacity=16384)
        if mode=='scatter':
            req=Distributed(g).requests(D,D,lambda b,i:b.index(D,b.index_value(i)%2))
            x=g.scatter_reduce(x,req,D,g.float64(0),lambda b,s,v:s+v,
                               order='global_gid_ascending',byte_capacity=16384)
        if mode=='scan':
            x=g.scan(x,g.float64(0),lambda b,s,v:b.product(state=s+v,value=s*v),
                     order='global_gid_ascending',byte_capacity=16384)['values']
        squares=g.map(D,lambda b,i:b.at(x,i)**2,byte_capacity=16384)
        return g.reduce(squares,g.float64(0),lambda b,s,v:s+v,
                        order='global_gid_ascending',byte_capacity=16384)
    with pytest.raises(ValueError,match='transport_byte_capacity'):
        jvp(f)
    derivative=jvp(f,transport_byte_capacity=32768)
    assert derivative.output==Product(value=Float64,tangent=Float64)
    assert 'distributed_map' in str(derivative.to_data())


def test_distributed_transpose_enumerates_global_gids_outside_local_regions():
    from pops.graph_derivatives import vjp, GraphPairings, tangent_type
    p,domain,prototype=_array()
    @graph(inputs=Product(x=type_of(prototype)),output=Float64)
    def energy(g,a):
        x=a['x'];D=g.domain(x)
        requests=Distributed(g).requests(D,D,lambda b,i:b.index(D,b.index_value(i)%2))
        imported=g.gather(x,requests,byte_capacity=16384)
        return g.reduce(imported,g.float64(0),lambda b,s,v:s+v*v,
                        order='global_gid_ascending',byte_capacity=16384)
    ti,to=tangent_type(energy.inputs),tangent_type(energy.output)
    identity=lambda T:graph(inputs=Product(value=T),output=T)(lambda g,a:a['value'])
    pulled=vjp(energy,pairings=GraphPairings(identity(to),identity(ti)),transport_byte_capacity=32768)
    text=str(pulled.to_data())
    assert 'structured_fold' in text and 'distributed_reduce' in text
    assert 'distributed_size' in text


def test_distributed_transpose_uses_existing_cpp_execution_path():
    from pops.graph_derivatives import vjp, GraphPairings, tangent_type
    from pops.codegen.program_codegen import emit_cpp_program
    from typed_program_support import typed_state
    p,domain,prototype=_array()
    @graph(inputs=Product(x=type_of(prototype)),output=Float64)
    def energy(g,a):
        return g.reduce(a['x'],g.float64(0),lambda b,s,v:s+v*v,
                        order='global_gid_ascending',byte_capacity=16384)
    ti,to=tangent_type(energy.inputs),tangent_type(energy.output)
    identity=lambda T:graph(inputs=Product(value=T),output=T)(lambda g,a:a['value'])
    pulled=vjp(energy,pairings=GraphPairings(identity(to),identity(ti)),transport_byte_capacity=32768)
    value=pulled(p,primal=p.product(x=prototype),covector=p.float64(1))['pullback']['x']
    total=p.reduce(value,p.float64(0),lambda q,s,x:s+x,order='global_gid_ascending',byte_capacity=32768)
    p.record_scalar('gradient',total)
    state=typed_state(p,'material',state_name='U')
    p.commit(state.next,p.value('accepted',state.n,at=state.next.point))
    cpp=emit_cpp_program(p)
    assert 'pops::distributed' in cpp or 'distributed::reduce' in cpp
    assert 'for (; iteration_' in cpp


def test_distributed_nested_native_cardinality_transpose_has_global_coordinate_loops():
    from pops.graph_derivatives import vjp,GraphPairings,tangent_type
    p=Program('nested_distributed_ad');D=Distributed(p).domain((0,1,1,3),capacity=3)
    def element(q,i):
        local=q.range_set(q.index_value(i)+1,capacity=3,ownership='replicated')
        return q.map(local,lambda r,j:r.float64(q.index_value(i)+r.index_value(j)+1))
    x=p.map(D,element,byte_capacity=65536)
    @graph(inputs=Product(x=type_of(x)),output=Float64)
    def energy(g,a):
        values=g.map(g.domain(a['x']),lambda b,i:
            b.reduce(b.at(a['x'],i),b.float64(0),lambda r,s,v:s+v*v,order='ascending'),
            byte_capacity=65536)
        return g.reduce(values,g.float64(0),lambda b,s,v:s+v,
                        order='global_gid_ascending',byte_capacity=65536)
    ti,to=tangent_type(energy.inputs),tangent_type(energy.output)
    identity=lambda T:graph(inputs=Product(value=T),output=T)(lambda g,a:a['value'])
    transpose=vjp(energy,pairings=GraphPairings(identity(to),identity(ti)),transport_byte_capacity=131072)
    text=str(transpose.to_data())
    assert text.count('structured_fold')>=2
    assert 'distributed_reduce' in text and 'finite_same_domain' in text


def test_nested_distributed_tangent_and_transpose_do_not_hide_local_collectives():
    from pops.graph_derivatives import vjp,GraphPairings,tangent_type
    p=Program('nested_distributed_identity_ad'); dist=Distributed(p)
    inner=dist.domain((0,1,1,2),capacity=2)
    data=p.map(inner,lambda q,i:q.float64(q.index_value(i)+1),byte_capacity=65536)
    outer=dist.domain((0,1,1,2),capacity=2)
    values=p.map(outer,lambda q,i:q.product(values=data),byte_capacity=65536)
    identity=graph(inputs=Product(x=type_of(values)),output=type_of(values))(lambda q,a:a['x'])
    ti,to=tangent_type(identity.inputs),tangent_type(identity.output)
    riesz=lambda typ:graph(inputs=Product(value=typ),output=typ)(lambda q,a:a['value'])
    tangent=jvp(identity,transport_byte_capacity=131072)
    transpose=vjp(identity,pairings=GraphPairings(riesz(to),riesz(ti)),transport_byte_capacity=131072)
    assert 'structured_fold' in str(tangent.to_data())
    assert str(transpose.to_data()).count('structured_fold')>=4


@pytest.mark.parametrize('mode',('gather','scatter','scan'))
def test_distributed_derivative_has_complete_zero_block_native_image(mode):
    from examples.api_v02.distributed_derivative_v14 import build_distributed_derivative_v14
    spec=build_distributed_derivative_v14(mode,ranks=3)
    assert not spec.case._time._block_indices() and not spec.initial_values
    import pops
    pops.resolve(pops.validate(spec.case),layout=spec.layout)


def test_distributed_derivative_inherits_case_storage_policy_through_closed_graphs():
    import pops
    from pops.codegen.program_codegen import emit_cpp_program
    from examples.api_v02.distributed_derivative_v14 import build_distributed_derivative_v14
    spec=build_distributed_derivative_v14('gather',ranks=3)
    plan=pops.resolve(pops.validate(spec.case),layout=spec.layout)
    cpp=emit_cpp_program(plan.time)
    assert '.configure(268435456,8192)' in cpp
    assert '.configure(67108864,65536)' not in cpp


@pytest.mark.parametrize('wrapped',(False,True))
def test_continuous_request_topology_keeps_native_switch_failure(wrapped):
    """At x=0 request(0/1) is discontinuous; discrete topology is still legal away from it."""
    p,D,prototype=_array()
    @graph(inputs=Product(x=Float64,values=type_of(prototype)),output=Float64)
    def function(g,a):
        def selected(q,x):
            domain=q.domain(a['values'])
            requests=Distributed(q).requests(domain,domain,lambda r,i:
                r.index(domain,r.branch(x>0,lambda s:s.int64(0),lambda s:s.int64(1))))
            imported=q.gather(a['values'],requests,byte_capacity=16384)
            return x*x+q.reduce(imported,q.float64(0),lambda r,total,value:total+value,
                                order='global_gid_ascending',byte_capacity=16384)
        if not wrapped:return selected(g,a['x'])
        trace=g.iterate(a['x'],lambda q,x:selected(q,x),lambda q,x:q.boolean(False),max_steps=1)
        return g.match(trace,ok=lambda q,item:item['state'],err=lambda q,error:3*a['x'])
    derived=jvp(function,differentiate_iterations=True,transport_byte_capacity=32768)
    snapshot=str(derived.to_data())
    assert '91001' in snapshot, 'requests erased the continuous switching-surface Failure'
    # The open branch remains expressible; data-dependent requests are not blanket-rejected.
    tangent=p.product(x=p.float64(1),values=p.map(D,lambda q,i:q.float64(0),byte_capacity=16384))
    result=derived(p,primal=p.product(x=p.float64(.5),values=prototype),tangent=tangent)
    assert result['tangent'].vtype=='scalar'
    from pops.codegen.program_codegen import emit_cpp_program
    from typed_program_support import typed_state
    p.record_scalar('open_branch_derivative',result['tangent'])
    state=typed_state(p,'request_probe',state_name='U')
    p.commit(state.next,p.value('accepted',state.n,at=state.next.point))
    cpp=emit_cpp_program(p)
    assert '91001' in cpp and '::requests(' in cpp and '::map<' in cpp
    if wrapped:
        assert 'differential_transport_' in snapshot
        assert 'variant_require_success' in snapshot
