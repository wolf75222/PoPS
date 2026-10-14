"""Source graph and lowering checks; these are not MPI/native reception receipts."""
import re
import pytest

from pops.time import Program
from pops.time.variant_types import Int64, Product, Result, Variant
from pops.time._program.distributed_arrays import Distributed, GLOBAL_OPS
from pops.codegen.program_codegen import emit_cpp_program
from typed_program_support import typed_state


def program():
    p=Program("distributed_v14")
    q=typed_state(p,"material",state_name="U");q.n
    d=Distributed(p);domain=d.domain((0,1,1,4),capacity=4)
    return p,q,d,domain


def finish(p,q,value):
    p.commit(q.next,p.value("accepted",p.float64(value)*q.n,at=q.next.point))
    assert p.validate()
    return emit_cpp_program(p)


def values(p,d,domain):
    return p.require_success(d.local_map(domain,
        lambda b,i:b.variant(Result(Int64,Int64),"ok",d.gid(i)+1),byte_capacity=65536))


@pytest.mark.parametrize("operation",("fold","scan","iterate","elementary"))
def test_local_regions_lower_without_collectives_inside_each_item(operation):
    p,q,d,domain=program()
    def body(b,i):
        gid=d.gid(i)
        if operation=="fold":
            return b.fold(gid,3,lambda c,s,k:s+k,max_steps=3)
        if operation=="scan":
            local=b.range_set(gid,capacity=4,ownership="replicated")
            out=b.scan(local,b.int64(0),lambda c,s,j:c.product(state=s+c.index_value(j),value=s),order="ascending")
            return out["state"]
        if operation=="iterate":
            out=b.iterate(gid,lambda c,s:s+1,lambda c,s:s>=4,max_steps=4)
            return b.require_success(out)["state"]
        from pops import math as pm
        return b.int64(1)+b.fold(b.int64(0),1,lambda c,s,k:c.branch(pm.exp(c.float64(gid))>0.,lambda z:s+1,lambda z:s))
    mapped=p.require_success(d.local_map(domain,lambda b,i:b.capture(lambda c:body(c,i)),byte_capacity=65536))
    total=d.reduce(mapped,p.int64(0),lambda b,s,x:s+x,order="global_gid_ascending",byte_capacity=65536)
    source=finish(p,q,total)
    assert "distributed::map" in source and "distributed::reduce" in source
    local=source.split("[&](auto distributed_index){",1)[1]
    closing=re.search(r"\}, &distributed_\d+_transport\);",local)
    assert closing is not None
    local=local[:closing.start()]
    assert "MPI_" not in local and "all_reduce" not in local
    assert "continuation_new_domain(" not in local


def test_global_ordered_reduce_scatter_and_scan_are_real_graph_nodes():
    p,q,d,domain=program();a=values(p,d,domain)
    target=d.domain((0,1,1,2),capacity=2)
    indices=d.requests(domain,target,lambda b,i:d.index(target,d.gid(i)%2))
    scattered=d.scatter_reduce(a,indices,target,p.int64(0),lambda b,s,x:s*10+x,
        order="global_gid_ascending",byte_capacity=65536)
    scanned=d.scan(a,p.int64(0),lambda b,s,x:b.product(state=s+x,value=s+x),
        order="global_gid_ascending",byte_capacity=65536)
    total=d.reduce(scattered,scanned["state"],lambda b,s,x:s+x,
        order="global_gid_ascending",byte_capacity=65536)
    source=finish(p,q,total)
    for name in ("distributed_reduce","distributed_scan","distributed_scatter_reduce"):
        assert name in GLOBAL_OPS and name in str(p.to_graph().to_data())
    assert "::scatter_reduce(" in source and "::scan(" in source


def test_collective_inside_a_local_region_is_refused_atomically():
    p,_,d,domain=program();before=p._ir_hash()
    def body(b,i):
        b.fold(b.int64(0),1,lambda c,s,k:(d.domain((0,1,1,2),capacity=2),s)[1])
        return b.variant(Result(Int64,Int64),"ok",b.int64(1))
    with pytest.raises(TypeError,match="collective or unsupported"):
        d.local_map(domain,body,byte_capacity=65536)
    assert before==p._ir_hash()


def test_scatter_requires_exact_source_and_target_families():
    p,_,d,domain=program();a=values(p,d,domain)
    other=d.domain((0,1,1,4),capacity=4)
    indices=d.requests(other,domain,lambda b,i:d.index(domain,d.gid(i)))
    with pytest.raises(TypeError,match="same exact source domain"):
        d.scatter_reduce(a,indices,domain,p.int64(0),lambda b,s,x:s+x,
            order="global_gid_ascending",byte_capacity=65536)
    indices=d.requests(domain,other,lambda b,i:d.index(other,d.gid(i)))
    with pytest.raises(TypeError,match="exact target domain"):
        d.scatter_reduce(a,indices,domain,p.int64(0),lambda b,s,x:s+x,
            order="global_gid_ascending",byte_capacity=65536)


@pytest.mark.parametrize("order",("ascending","unordered","mpi_sum"))
def test_global_order_is_explicit(order):
    p,_,d,domain=program();a=values(p,d,domain)
    with pytest.raises(ValueError,match="global_gid_ascending"):
        d.reduce(a,p.int64(0),lambda b,s,x:s+x,order=order,byte_capacity=65536)


def test_typed_combiner_failure_remains_a_compiled_payload():
    p,q,d,domain=program();a=values(p,d,domain)
    error=Variant(bad_value=Product(value=Int64))
    def combine(b,s,x):
        result=b.branch(x==3,
            lambda z:z.variant(Result(Int64,error),"err",z.variant(error,"bad_value",z.product(value=x))),
            lambda z:z.variant(Result(Int64,error),"ok",s+x))
        return b.require_success(result)
    captured=p.capture(lambda b:d.reduce(a,b.int64(0),combine,order="global_gid_ascending",byte_capacity=65536))
    answer=p.match(captured,ok=lambda b,x:x,err=lambda b,e:b.int64(19))
    source=finish(p,q,answer)
    assert "CollectivePayloadFailure" in source and "bad_value" in source


def test_local_map_cannot_erase_an_escaping_user_error():
    p,_,d,domain=program()
    def body(b,i):
        value=b.require_success(b.variant(Result(Int64,Product(code=Int64)),"err",b.product(code=d.gid(i))))
        return b.variant(Result(Int64,Int64),"ok",value)
    with pytest.raises(TypeError,match="retain every escaping failure"):
        d.local_map(domain,body,byte_capacity=65536)


@pytest.mark.parametrize("ranks", (2, 3))
@pytest.mark.parametrize("kind", ("ordered", "nested", "budget", "typed_reduce", "terminal", "terminal_sparse"))
def test_complete_distributed_reception_graph_resolves_and_emits(kind, ranks):
    import pops
    from examples.api_v02.distributed_algebra_v14 import build_distributed_algebra
    from pops.codegen.program_models import ProgramModelGraph
    spec=build_distributed_algebra(kind,ranks=ranks)
    plan=pops.resolve(pops.validate(spec.case),layout=spec.layout)
    cpp=emit_cpp_program(plan.time,model_graph=ProgramModelGraph.from_resolved_blocks(plan.blocks))
    assert "accepted_after_distributed_algebra" in cpp
    assert "continuation_transport_session()" in cpp
    if kind=="nested":
        assert "continuation_new_local_domain" in cpp
        assert "::distributed_array(" in cpp and "::import_values(" in cpp
    if kind=="ordered":
        assert "::requests_from_array<" in cpp and "::scan(" in cpp and "::scatter_reduce(" in cpp


def test_fresh_ownership_instance_requires_a_live_exact_family_witness():
    p,q,d,old=program()
    new=d.domain((0,2,3,4),family=old)
    assert new.id!=old.id
    assert new.attrs["signature"]["family"]==old.attrs["signature"]["family"]
    assert new.attrs["signature"]["capacity"]==4
    assert new.inputs[-1] is old
    source=finish(p,q,d.size(new))
    assert "->family()==" in source and "->capacity()==4" in source
    before=p._ir_hash()
    with pytest.raises(TypeError,match="exact capacity"):
        d.domain((0,2,3,4),capacity=5,family=old)
    assert p._ir_hash()==before
