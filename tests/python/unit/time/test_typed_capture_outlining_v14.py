"""Native capture source boundaries, exact operations, and lexical failure targets."""
import re
import pytest
from pops.time import Program
from pops.time.variant_types import Bool, Float64, Int64, Product, Result, Failure, type_of
from pops.continuations import ContinuationSchema
from pops.graph_functions import graph
from pops.codegen.program_codegen import emit_cpp_program
from pops.codegen.program_source_map import source_map
import pops.codegen.program_emit_structured_control as original

def baseline_source(p, monkeypatch):
    with monkeypatch.context() as context:
        context.setattr(original, '_outline_capture_region', lambda lines,start,node_id: None)
        return emit_cpp_program(p)


def strip_outlining(source):
    output=[];stack=[];scaffold=0
    for line in source.splitlines(True):
        stripped=line.strip()
        if stripped.startswith('// pops-outlined-capture begin '):
            stack.append(stripped.rsplit(' ',1)[1]);scaffold=5;continue
        if scaffold:
            scaffold-=1;continue
        if stripped.startswith('// pops-outlined-capture end '):
            assert stack.pop()==stripped.rsplit(' ',1)[1]
            assert output.pop().strip()=='}();'
            continue
        # Each outlined ancestor adds exactly two indentation characters. All
        # original operations/markers keep their content and relative order.
        if line.strip():
            count=2*len(stack);assert line.startswith(' '*count);line=line[count:]
        output.append(line)
    assert not stack and not scaffold
    return ''.join(output)


def test_multiline_schema_entries_keep_each_physical_line_and_blank_byte():
    entries = ["const auto schema = [&] {\n  int value = 7;\n\n  return value;\n}();", "use(schema);"]
    before = "".join("  " + line + "\n" for line in entries)
    original._outline_capture_region(entries, 0, 19)
    after = "".join("  " + line + "\n" for line in entries)
    assert strip_outlining(after) == before


def check_goto_boundaries(source):
    scopes=[];labels={};jumps=[]
    for line in source.splitlines():
        stripped=line.strip()
        if stripped.startswith('// pops-outlined-capture begin '):scopes.append(stripped.rsplit(' ',1)[1])
        elif stripped.startswith('// pops-outlined-capture end '):assert scopes.pop()==stripped.rsplit(' ',1)[1]
        else:
            owner=tuple(scopes)
            for label in re.findall(r'\b(typed_(?:capture|iteration)_done_\d+)\s*:',line):labels[label]=owner
            for label in re.findall(r'\bgoto\s+(typed_(?:capture|iteration)_done_\d+)\s*;',line):jumps.append((label,owner))
    assert jumps
    for label,scope in jumps:assert labels[label]==scope,(label,scope,labels[label])


def program(kind):
    p=Program('capture_outlining_source_'+kind)
    error=Failure(user=Product(code=Int64))
    result=Result(Float64,error)
    def child(q,x):
        return q.capture(lambda b:b.require_success(b.branch(x>0,
            lambda c:c.variant(result,'ok',x+1),
            lambda c:c.variant(result,'err',c.variant(error,'user',c.product(code=c.int64(73)))))))
    if kind=='nested_typed':
        p.capture(lambda b:b.require_success(child(b,b.float64(2))))
    elif kind=='loop_typed':
        p.capture(lambda b:b.require_success(b.iterate(b.float64(1),
            lambda c,x:c.require_success(child(c,x)),lambda c,x:x>=4,max_steps=4))['state'])
    elif kind=='runtime_failure':
        p.capture(lambda b:b.require_success(b.capture(lambda c:c.variant(
            Result(Float64,Failure()),'ok',c.float64(1)/c.float64(0)))))
    elif kind=='owned_product':
        domain=p.range_set(3,capacity=3,ownership='replicated')
        array=p.map(domain,lambda b,i:b.float64(b.index_value(i)))
        payload=p.product(values=array,choice=p.variant(Result(Float64,Failure()),'ok',p.float64(2)))
        p.capture(lambda b:b.variant(Result(type_of(payload),Failure()),'ok',payload))
    else:raise AssertionError(kind)
    # Complete finite-state Program carrier, not a bypass of validate(). The
    # test concerns source control boundaries; no numerical acceptance claimed.
    initial=p.product(value=p.float64(0),time=p.float64(0),step=p.int64(0))
    schema=ContinuationSchema(initial,fields=(),time=('time',),step=('step',))
    base=p.continuation_open(initial,schema);old=p.continuation_read(base)
    proposal=p.product(h=p.requested_dt())
    envelope=p.product(continuation=p.product(value=old['value'],time=old['time']+proposal['h'],step=old['step']+1),
                       increment=p.product(),observations=p.product(),checks=p.product())
    candidate=p.continuation_freeze(base,proposal,envelope)
    @graph(inputs=Product(base=schema.read_type,proposal=type_of(proposal),candidate=type_of(envelope)),
           output=Product(accepted=Bool,checks=Product()))
    def assess(g,args):return g.product(accepted=g.boolean(True),checks=g.product())
    p.continuation_publish(p.continuation_accept(candidate,p.continuation_attest(candidate,assess)))
    return p


@pytest.mark.parametrize('kind',('nested_typed','loop_typed','runtime_failure','owned_product'))
def test_same_instructions_and_local_failure_targets(kind,monkeypatch):
    p=program(kind);before=baseline_source(p,monkeypatch)
    after=emit_cpp_program(p)
    assert strip_outlining(after)==before
    assert source_map(after)['complete']
    # Complete byte equality after removing only wrappers also compares every
    # C17 operation/inventory marker, binding and formal authority.
    if kind in ('nested_typed','loop_typed'):check_goto_boundaries(after)
    assert '__attribute__((noinline))' in after


def test_real_nested_trial_has_local_labels_and_unchanged_control(monkeypatch):
    from examples.api_v02.decision_assess_v14 import build_decision_assess_v14
    spec=build_decision_assess_v14(mode='nested_success')
    p=spec.case._time;before=baseline_source(p,monkeypatch)
    after=emit_cpp_program(p)
    assert strip_outlining(after)==before
    assert source_map(after)['complete']
    # Complete byte equality after removing only wrappers also compares every
    # C17 operation/inventory marker, binding and formal authority.
    check_goto_boundaries(after)
    assert after.count('// pops-outlined-capture begin ')==106


def test_field_capture_keeps_context_owned_scratch_reference(monkeypatch):
    from examples.api_v02.methods_T01_T12 import _storage
    from pops.time import FixedDt
    case,layout,model,block,state,p,q,initial=_storage('capture_field_outlining',('q',),(1.,),sampling='cell_average')
    result=Result(type_of(q.n),Failure())
    captured=p.capture(lambda b:b.variant(result,'ok',b.value('twice',2*q.n)))
    value=p.require_success(captured)
    p.commit(q.next,p.value('accepted',value,at=q.next.point));p.step_strategy(FixedDt(.01));case.program(p)
    before=baseline_source(p,monkeypatch)
    after=emit_cpp_program(p)
    assert strip_outlining(after)==before
    assert source_map(after)['complete']
    assert 'ctx.scratch_state' in after
