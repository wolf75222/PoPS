"""Unit checking follows distributed algebra equations, not MPI ownership labels."""
import pytest
from pops._ir.quantity import PhysicalDimension, PhysicalSupport
from pops.model.spaces import Space
from pops.representations import ScientificType, ScientificValue, ScientificMap
from pops.scientific_graph import validate_scientific_program
from pops.graph_functions import graph
from pops.time import Program
from pops.time.variant_types import Float64, Int64, Product, Result, type_of
from pops.time._program.distributed_arrays import Distributed

L=PhysicalDimension((('length',1),));T=PhysicalDimension((('time',1),))


def scientific(finite,unit=L,dofs=None):
    space=Space('coordinate',('q',),support=PhysicalSupport(()),units=(unit,),value_shape=())
    return ScientificType(finite,space,'point_value',(),(),{'name':'counting'},dofs)


@pytest.mark.parametrize('operation',('map','import','csr','reduce','scan','scatter'))
def test_distributed_closed_equations_preserve_component_units(operation):
    p=Program('scientific_distributed_'+operation);d=Distributed(p)
    domain=d.domain((0,1,1,3),capacity=3)
    values=p.require_success(d.local_map(domain,
        lambda b,i:b.variant(Result(Float64,Int64),'ok',b.float64(d.gid(i)+1)),byte_capacity=4096))
    qtype=scientific(type_of(values),dofs=type_of(domain));zero=scientific(Float64)
    def body(b,a):
        dd=Distributed(b);ds=dd.domain_of(a['q'])
        if operation=='map':
            return b.require_success(dd.local_map(ds,lambda c,i:c.variant(Result(Float64,Int64),'ok',
                2*dd.at_local(a['q'],i)),byte_capacity=4096))
        if operation in ('import','csr','scatter'):
            count=dd.size(ds)
            requests=dd.requests(ds,ds,lambda c,i:dd.index(ds,(dd.gid(i)+1)%count))
            if operation=='import':return dd.import_values(a['q'],requests,byte_capacity=4096)
            if operation=='csr':
                csr=dd.csr(ds,requests,lambda c,i:i,nnz_capacity=3,target=ds)
                return dd.import_csr(a['q'],csr,byte_capacity=4096)
            return dd.scatter_reduce(a['q'],requests,ds,a['zero'],lambda c,s,x:s+x,
                order='global_gid_ascending',byte_capacity=4096)
        if operation=='scan':
            return dd.scan(a['q'],a['zero'],lambda c,s,x:c.product(state=s+x,value=s+x),
                order='global_gid_ascending',byte_capacity=4096)['values']
        return dd.reduce(a['q'],a['zero'],lambda c,s,x:s+x,
            order='global_gid_ascending',byte_capacity=4096)
    output=zero if operation=='reduce' else qtype
    function=graph(inputs=Product(q=qtype.finite_type,zero=Float64),output=output.finite_type)(body)
    mapping=ScientificMap(function,inputs={'q':qtype,'zero':zero},output=output,strict=True)
    result=mapping(p,q=ScientificValue(values,qtype),zero=ScientificValue(p.float64(0),zero))
    assert result.scientific_type==output
    assert validate_scientific_program(p,strict=True)
    with pytest.raises(TypeError,match='dimensions'):
        ScientificMap(function,inputs={'q':qtype,'zero':zero},
            output=scientific(output.finite_type,T,dofs=output.dofs),strict=True)


def test_distributed_combiner_cannot_add_incompatible_scientific_quantities():
    p=Program('scientific_reduce_bad');d=Distributed(p);domain=d.domain((0,1,1,3),capacity=3)
    values=p.require_success(d.local_map(domain,
        lambda b,i:b.variant(Result(Float64,Int64),'ok',b.float64(1)),byte_capacity=4096))
    qtype=scientific(type_of(values),dofs=type_of(domain));scalar=scientific(Float64,T)
    function=graph(inputs=Product(q=qtype.finite_type,initial=Float64),output=Float64)(
        lambda b,a:Distributed(b).reduce(a['q'],a['initial'],lambda c,s,x:s+x,
            order='global_gid_ascending',byte_capacity=4096))
    with pytest.raises(TypeError,match='dimensions'):
        ScientificMap(function,inputs={'q':qtype,'initial':scalar},output=scalar,strict=True)
