"""Construction and local emission tests only; no native compilation or MPI execution."""
import pytest

from pops.time import Program
from pops.time.variant_types import Int64,Float64,Product,Result,Variant
from pops.time._program.distributed_arrays import Distributed,transportable
from pops.codegen.program_emit_distributed_arrays import emit_distributed_op,OPS


def emitted(p):
    var={v.id:'std::int64_t(%s)'%v.attrs['value'] for v in p._values if v.op=='int64_constant'}
    lines=[]
    for v in p._values:
        if v.op in OPS:emit_distributed_op(p,v,var,None,lines,[],{},None,'system')
        elif v.op=='variant_require_success':var[v.id]=var[v.inputs[0].id]['payloads']['ok']
    return '\n'.join(lines),var


def test_true_program_ssa_emits_local_loop_payload_and_exact_int64():
    p=Program('distributed');d=Distributed(p);D=d.domain((0,1,1,2),capacity=3);calls=[]
    T=Result(Product(id=Int64,value=Float64),Product(code=Int64))
    def body(b,i):
        calls.append(i)
        return b.variant(T,'ok',b.product(id=d.gid(i)+(1<<60)+17,value=b.float64(d.gid(i))))
    d.local_map(D,body,byte_capacity=4096)
    src,_=emitted(p)
    assert len(calls)==1
    assert '1152921504606846976' in src
    assert 'distributed::map<pops::kNativeDimension>' in src
    assert 'ctx.distributed_registry()' in src
    local=src.split('[&](auto distributed_index){',1)[1]
    assert 'MPI_' not in local and 'all_reduce' not in local and 'ctx.' not in local
    assert 'pops_finite_instance_version' not in src


def test_inactive_singular_arm_is_a_native_local_branch():
    p=Program('lazy');d=Distributed(p);D=d.domain((0,0,1),capacity=1);T=Result(Int64,Int64)
    d.local_map(D,lambda b,i:b.branch(d.gid(i)==0,
        lambda c:c.variant(T,'ok',c.int64(17)),
        lambda c:c.variant(T,'ok',c.int64(1)//0)),byte_capacity=2048)
    src,_=emitted(p)
    assert 'if ((' in src and 'else {' in src
    assert '::integer(3,' in src


@pytest.mark.parametrize('operation',('distributed_domain','distributed_map'))
def test_collective_body_is_rejected_atomically(operation):
    p=Program('effects');d=Distributed(p);D=d.domain((0,1,2),capacity=2);before=p._ir_hash()
    def body(b,i):
        if operation=='distributed_domain':
            d.domain((0,1,2),capacity=2)
        else:
            d.local_map(D,lambda c,j:c.variant(Result(Int64,Int64),'ok',d.gid(j)),byte_capacity=1024)
        return b.variant(Result(Int64,Int64),'ok',b.int64(1))
    with pytest.raises(TypeError,match='collective or unsupported'):d.local_map(D,body,byte_capacity=1024)
    assert p._ir_hash()==before


@pytest.mark.parametrize('kind',('state','array','index','index_set'))
def test_untyped_or_nontransportable_payload_is_explicitly_refused(kind):
    with pytest.raises(TypeError,match='bounded domain authority'):transportable({'vtype':kind})


def test_requests_import_and_csr_keep_exact_domain_occurrences():
    p=Program('import');d=Distributed(p);D=d.domain((0,1,1,2),capacity=2);Q=d.domain((0,2,2,3),capacity=3)
    source=p.require_success(d.local_map(D,lambda b,i:b.variant(Result(Int64,Int64),'ok',d.gid(i)+(1<<60)),byte_capacity=4096))
    req=d.requests(Q,D,lambda b,i:d.index(D,d.gid(i)%2))
    imported=d.import_values(source,req,byte_capacity=4096)
    C=d.csr(D,req,lambda b,k:k*2,nnz_capacity=3) # source construction only; rank-specific bounds checked natively
    d.import_csr(source,C,byte_capacity=4096)
    src,_=emitted(p)
    assert '::requests(' in src and '::import_values(' in src and '::csr_offsets(' in src and '::import_csr(' in src
    other=d.domain((0,1,1,2),capacity=2)
    wrong=d.requests(Q,other,lambda b,i:d.index(other,d.gid(i)%2))
    with pytest.raises(TypeError,match='exact source'):d.import_values(source,wrong,byte_capacity=4096)
    assert imported.attrs['signature']['family']==Q.attrs['signature']['family']


@pytest.mark.parametrize('capacity',(-1,True,2**63,1.0))
def test_bad_capacity_is_not_coerced(capacity):
    p=Program('capacity');d=Distributed(p)
    with pytest.raises(ValueError,match='exact Int64'):d.domain((0,1),capacity=capacity)
    assert not p._values


@pytest.mark.parametrize('capacity',(0,2**31,2**53+3,2**63-1))
def test_domain_cardinality_capacity_is_independent_of_transport_capacity(capacity):
    p=Program('large_domain_capacity');d=Distributed(p)
    domain=d.domain((0,0,0),capacity=capacity)
    successor=d.domain((0,0,0),family=domain)
    assert successor.attrs['signature']['capacity']==capacity
    source,_=emitted(p)
    assert '.issue(0,%d,'%capacity in source
    # Transport/storage budgets remain independent positive Int64 contracts.
    with pytest.raises(ValueError,match='exact Int64'):
        d.local_map(domain,lambda b,i:b.variant(Result(Int64,Int64),'ok',b.int64(1)),
                    byte_capacity=2**63)


def test_outer_amr_calendar_uses_native_registry():
    p=Program('amr');d=Distributed(p);D=d.domain((0,1),capacity=1)
    lines=[]
    var={v.id:'std::int64_t(%s)'%v.attrs['value'] for v in p._values if v.op=='int64_constant'}
    emit_distributed_op(p,D,var,None,lines,[],{},None,'amr_system')
    assert 'ctx.distributed_registry()' in '\n'.join(lines)


def test_equations_are_part_of_serialized_identity_and_detach():
    from pops.time._program.detach import detach_compiled_program
    def build(constant):
        p=Program('identity');d=Distributed(p);D=d.domain((0,1,2),capacity=2)
        d.local_map(D,lambda b,i:b.variant(Result(Int64,Int64),'ok',d.gid(i)+constant),byte_capacity=4096)
        return p
    p=build(3);q=build(4)
    assert p._ir_hash()!=q._ir_hash()
    assert p._ir_hash()==detach_compiled_program(p)._ir_hash()
    data=p.to_graph().to_data()
    assert 'distributed_map' in str(data) and 'int64_op' in str(data)


def test_full_existing_program_emits_the_staged_distributed_kernel():
    from typed_program_support import typed_state
    from pops.codegen.program_codegen import emit_cpp_program
    p=Program('full_distributed_source');q=typed_state(p,'material',state_name='U');q.n
    d=Distributed(p);D=d.domain((0,1,1,2),capacity=2)
    result=d.local_map(D,lambda b,i:b.variant(Result(Int64,Int64),'ok',d.gid(i)+(1<<60)),byte_capacity=4096)
    p.require_success(result)
    p.commit(q.next,p.value('accepted',q.n,at=q.next.point))
    assert p.validate()
    src=emit_cpp_program(p)
    assert '#include <pops/runtime/program/distributed_array.hpp>' in src
    assert 'ctx.distributed_registry()' in src and '::map<pops::kNativeDimension>' in src
    assert '1152921504606846976' in src


@pytest.mark.parametrize("operation", ("index", "gid", "rank", "at_local"))
@pytest.mark.parametrize("nested_branch", (False, True))
def test_distributed_local_primitive_refuses_a_replicated_region(operation, nested_branch):
    p = Program("local_distributed_scope")
    d = Distributed(p)
    domain = d.domain((0, 1, 2), capacity=2)
    saved = []
    def local(b, index):
        saved.append(index)
        return b.variant(Result(Int64, Int64), "ok", d.gid(index))
    values = p.require_success(d.local_map(domain, local, byte_capacity=4096))
    condition = p.boolean(True)
    def action():
        if operation == "index": return d.index(domain, 0)
        if operation == "gid": return d.gid(saved[0])
        if operation == "rank": return d.rank()
        return d.at_local(values, saved[0])
    before = p._ir_hash()
    with pytest.raises(TypeError, match="requires a lexical distributed local body"):
        if nested_branch:
            p.branch(condition, lambda q: action(), lambda q: q.int64(0))
        else:
            action()
    assert p._ir_hash() == before


@pytest.mark.parametrize('budget',(2**31,2**53+3,2**63-1))
def test_wide_distributed_budgets_are_declared_without_allocating_their_capacity(budget):
    from pops.time._program.detach import detach_compiled_program
    p=Program('wide_transport');d=Distributed(p,storage_bytes=budget,storage_versions=budget)
    domain=d.domain((0,0,0),capacity=0)
    value=d.local_map(domain,lambda b,i:b.variant(Result(Int64,Int64),'ok',b.int64(17)),byte_capacity=budget)
    source,_=emitted(p)
    assert '.configure(%d,%d)'%(budget,budget) in source
    assert value.attrs['byte_capacity']==budget
    assert p.to_graph().to_data()==detach_compiled_program(p).to_graph().to_data()


@pytest.mark.parametrize('budget',(0,-1,True,2**63,1.5))
def test_invalid_wide_budget_is_rejected_before_authoring_changes(budget):
    p=Program('invalid_budget');before=p._ir_hash()
    with pytest.raises(ValueError,match='positive exact Int64'):
        Distributed(p,storage_bytes=budget)
    assert p._ir_hash()==before
