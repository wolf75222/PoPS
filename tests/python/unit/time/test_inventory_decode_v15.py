"""Real Program source checks; native acceptance is separate and not claimed here."""
import pytest
import pops
from pops.graph_functions import graph
from pops.time.variant_types import Float64, Int64, Bool, Product, type_of
from pops.inventory_decode import (_proof, _structure, _formals, checked_inventory_decode,
                                   validate_inventory_conversion, DecodeFailure)


def problem(mode='exact', *, array=False, program=None, base_value=None, context=None):
    p=pops.Program('inventory_decode') if program is None else program
    before=p.product() if context is None else context
    base=p.float64(0) if base_value is None else base_value
    application=p.float64(2)
    atol=p.float64(1.e-12)
    if array:
        domain=p.range_set(2,capacity=3,ownership='replicated')
        base=p.map(domain,lambda g,i:g.float64(g.index_value(i)))
        application=p.map(domain,lambda g,i:g.float64(2))
        atol=p.map(domain,lambda g,i:g.float64(1.e-12))
    value_type=type_of(base)
    def unary(g,value,fn):
        return g.map(g.domain(value),lambda h,i:fn(h,h.at(value,i))) if array else fn(g,value)
    @graph(inputs=Product(value=value_type,data=type_of(before)),output=value_type)
    def encode(g,a):return unary(g,a['value'],lambda h,x:x+x*x)
    @graph(inputs=Product(inventory=value_type,data=type_of(before),budget=Int64),output=value_type)
    def decode(g,a):
        import pops.math as pm
        if mode=='approximate':return unary(g,a['inventory'],lambda h,x:h.float64(.9))
        if mode=='wrong':return unary(g,a['inventory'],lambda h,x:h.float64(0))
        if mode=='negative':return unary(g,a['inventory'],lambda h,x:(-1-pm.sqrt(1+4*x))/2)
        return unary(g,a['inventory'],lambda h,x:2*x/(1+pm.sqrt(1+4*x)))
    @graph(inputs=Product(inventory=value_type,data=type_of(before)),output=Bool)
    def domain(g,a):
        if array:
            return g.reduce(a['inventory'],g.boolean(True),
                            lambda h,s,x:h.branch(s,lambda k:x>=0,lambda k:k.boolean(False)),order='ascending')
        return a['inventory']>=0
    @graph(inputs=Product(value=value_type,data=type_of(before)),output=Bool)
    def admissible(g,a):return g.boolean(True)
    @graph(inputs=Product(value=value_type,data=type_of(before)),output=Bool)
    def selection(g,a):
        if array:
            return g.reduce(a['value'],g.boolean(True),
                            lambda h,s,x:h.branch(s,lambda k:x>=0,lambda k:k.boolean(False)),order='ascending')
        return a['value']>=0
    recipient=encode(p,value=base,data=before)
    functions=dict(encode=encode,decode=decode,inventory_domain=domain,admissible=admissible,selection=selection)
    bindings=dict(recipient=recipient,application=application,base_value=base,before_data=before,
                  after_data=before,budget=p.int64(4),atol=atol,rtol=p.float64(0))
    return p,functions,bindings


@pytest.mark.parametrize('mode',('exact','wrong','negative'))
def test_original_equation_and_selection_remain_closed_executable_graphs(mode):
    p,f,a=problem(mode)
    proof=_proof(f,a,None)
    out=proof(p,**a)
    assert out.vtype=='product'
    payload=proof.to_data()
    # No accepted flag or algorithm-supplied residual can remove these graph operations.
    text=str(payload)
    assert 'inventory_residual' in text and str(int(DecodeFailure.UNSELECTED)) in text
    assert str(int(DecodeFailure.OLD_ENCODING)) in text
    q,args=_formals(proof.inputs)
    arguments={key:args[key] for key in a}
    arguments['after_data']=arguments['before_data']
    expected=proof(q,**arguments)
    assert _structure(out,a)==_structure(expected,arguments)


def test_array_original_equation_checks_exact_instances_and_each_coordinate():
    p,f,a=problem(array=True)
    proof=_proof(f,a,None)
    out=proof(p,**a)
    text=str(proof.to_data())
    assert 'finite_same_domain' in text and 'finite_map' in text and 'structured_fold' in text
    assert out.vtype=='product'


def test_plain_callback_is_not_a_closed_decoder_and_does_not_mutate_program():
    p,f,a=problem()
    before=p._ir_hash()
    f['decode']=lambda *a,**kw:None
    with pytest.raises(TypeError,match='closed GraphFunctions'):
        checked_inventory_decode(p,**a,**f)
    assert p._ir_hash()==before


@pytest.mark.parametrize('array',(False,True))
def test_conversion_validator_reconstructs_complete_guard_graph(array):
    p,f,a=problem(array=array)
    result=checked_inventory_decode(p,**a,**f)
    assert result.vtype=='variant'
    markers=[v for v in p._issued_values.values() if v.op=='provenance_bind']
    assert len(markers)==1
    before=(p._next_id,tuple(p._issued_values))
    data=validate_inventory_conversion(markers[0])
    assert data['defect_policy']=='declare_once_identity_no_reapply'
    assert data['application'] is a['application']
    assert (p._next_id,tuple(p._issued_values))==before


def test_copying_proof_to_another_retained_value_is_rejected():
    from pops.ledger_provenance import bind_contract
    p,f,a=problem()
    checked_inventory_decode(p,**a,**f)
    marker=next(v for v in p._issued_values.values() if v.op=='provenance_bind')
    data=dict(marker.attrs['contract'])
    with pytest.raises(ValueError,match='copied onto another'):
        bind_contract(p,a['base_value'],'inventory_conversion',data)


def test_counterfeit_checked_product_cannot_bypass_original_equation():
    from pops.ledger_provenance import bind_contract
    p,f,a=problem()
    checked_inventory_decode(p,**a,**f)
    marker=next(v for v in p._issued_values.values() if v.op=='provenance_bind')
    data=dict(marker.attrs['contract'])
    fake=p.product(value=a['base_value'],defect=p.float64(0),encoded=a['application'],
                   bound=a['atol'],atol=a['atol'],rtol=a['rtol'],budget=a['budget'])
    q=fake['value'];r=fake['defect']
    data.update(checked=fake,retained_input=q,defect=r)
    with pytest.raises(ValueError,match='original-equation/control graph'):
        bind_contract(p,q,'inventory_conversion',data)



def test_changed_equation_descriptor_cannot_relabel_an_existing_proof():
    from pops.inventory_decode import _descriptor
    from pops.ledger_provenance import bind_contract
    p,f,a=problem()
    checked_inventory_decode(p,**a,**f)
    marker=next(v for v in p._issued_values.values() if v.op=='provenance_bind')
    @graph(inputs=f['encode'].inputs,output=f['encode'].output)
    def changed(g,b):return 2*b['value']
    data=dict(marker.attrs['contract']);functions=dict(data['functions'])
    functions['encode']=_descriptor(changed);data['functions']=functions
    p._recording.append([marker])
    try: forged=p._replace_value(marker,attrs=dict(marker.attrs,contract=data))
    finally: p._recording.pop()
    with pytest.raises(ValueError,match='original-equation/control graph'):
        validate_inventory_conversion(forged)


def test_changed_target_binding_cannot_retag_old_equation_check():
    from pops.ledger_provenance import bind_contract
    p,f,a=problem();checked_inventory_decode(p,**a,**f)
    marker=next(v for v in p._issued_values.values() if v.op=='provenance_bind')
    data=dict(marker.attrs['contract']);inputs=dict(data['input_values'])
    target=p.float64(3);inputs['application']=target
    data.update(application=target,input_values=inputs)
    p._recording.append([marker])
    try: forged=p._replace_value(marker,attrs=dict(marker.attrs,contract=data))
    finally: p._recording.pop()
    with pytest.raises(ValueError,match='original-equation/control graph'):
        validate_inventory_conversion(forged)


def test_typed_inner_failure_is_not_replaced_by_an_approval_flag():
    from pops.time.variant_types import Failure,Result
    p,f,a=problem()
    error=Failure(inner=Product(code=Int64))
    @graph(inputs=f['decode'].inputs,output=Result(type_of(a['base_value']),error))
    def failure(g,b):
        return g.variant(Result(type_of(b['inventory']),error),'err',
                         g.variant(error,'inner',g.product(code=g.int64(9007199254740995))))
    f['decode']=failure
    result=checked_inventory_decode(p,**a,**f)
    assert 'inner' in str(value_signature_for_test(result))
    marker=next(v for v in p._issued_values.values() if v.op=='provenance_bind')
    validate_inventory_conversion(marker)
    assert '9007199254740995' in str(failure.to_data())


def value_signature_for_test(value):
    from pops.time._program.structured_control import value_signature
    return value_signature(value)


def test_decoder_internal_finite_workspace_is_alpha_renamed_not_a_new_authority():
    p,f,a=problem()
    original=f['decode']
    p.range_set(1,capacity=1,ownership='replicated')
    @graph(inputs=original.inputs,output=original.output)
    def workspace(g,b):
        d=g.range_set(2,capacity=4,ownership='replicated')
        x=g.map(d,lambda h,i:h.float64(h.index_value(i)))
        total=g.reduce(x,g.float64(0),lambda h,s,v:s+v,order='ascending')
        return original(g,inventory=b['inventory'],data=b['data'],budget=b['budget'])+0*total
    f['decode']=workspace
    checked_inventory_decode(p,**a,**f)
    marker=next(v for v in p._issued_values.values() if v.op=='provenance_bind')
    validate_inventory_conversion(marker)


def test_negative_budget_and_accuracy_have_mandatory_native_guards():
    p,f,a=problem();a['budget']=p.int64(-1);a['atol']=p.float64(-1)
    result=checked_inventory_decode(p,**a,**f)
    marker=next(v for v in p._issued_values.values() if v.op=='provenance_bind')
    validate_inventory_conversion(marker)
    text=str(_proof(f,a,None).to_data())
    assert str(int(DecodeFailure.INVALID_BUDGET)) in text
    assert str(int(DecodeFailure.INVALID_ACCURACY)) in text
    assert result.vtype=='variant'  # Runtime rejection is not claimed by a source check.



def build_inventory_decode_case(mode='exact'):
    from pops.continuations import ContinuationSchema,Candidate
    from pops.contributions import Contribution,Delivery,Increment,FiniteInterval,LinearMap
    from pops.ledger_provenance import InventoryBinding
    from pops.domain import Rectangle
    from pops.frames import Cartesian2D
    from pops.layouts import Uniform
    from pops.mesh import CartesianGrid,PeriodicAxes
    from pops.time import FixedDt
    p=pops.Program('C12_decode_'+mode)
    initial=p.product(q=p.float64(0),time=p.float64(0),step=p.int64(0))
    schema=ContinuationSchema(initial,fields=(),time=('time',),step=('step',))
    base=p.continuation_open(initial,schema);old=p.continuation_read(base)
    p,f,a=problem('approximate' if mode=='approximate' else 'exact',program=p,base_value=old['q'])
    recipient=a['recipient'];h=p.requested_dt();proposal=p.product(h=h)
    window=FiniteInterval(old['time'],old['time']+h,'physical')
    @graph(inputs=Product(value=type_of(recipient)),output=type_of(recipient))
    def identity(g,b):return b['value']
    action=LinearMap.differential(identity,at=p.product(value=recipient),wrt='value')
    flux=Contribution('J',p.float64(2),window)
    ledger=Increment((flux,),(Delivery(flux,recipient,action),))
    a['application']=ledger.apply(p,recipient,name='target_inventory')
    if mode=='approximate':a['atol']=p.float64(.3)
    result=p.require_success(checked_inventory_decode(p,**a,**f))
    retained=result['value'];defect=result['defect']
    residual=Contribution('conversion_residual',defect,window,kind='defect')
    final=Increment((flux,residual),(Delivery(flux,recipient,action),Delivery(residual,recipient,action)))
    if mode=='missing_defect':final=ledger
    if mode=='duplicate_defect':
        duplicate=Contribution('duplicate',defect,window,kind='defect')
        final=Increment((flux,residual,duplicate),(*final.deliveries,Delivery(duplicate,recipient,action)))
    packed=final.pack(p,bindings=(InventoryBinding(recipient,('q',),conversion=retained),))
    value=old['q'] if mode=='hidden_conversion' else retained
    after=p.product(q=value,time=old['time']+h,step=old['step']+1)
    candidate=Candidate(p,base=base,proposal=proposal,continuation=after,increment=packed,
                        observations=p.product(also_computed=retained),checks=p.product())
    envelope=type_of(p.continuation_candidate_value(candidate))
    @graph(inputs=Product(base=schema.read_type,proposal=type_of(proposal),candidate=envelope),
           output=Product(accepted=Bool,checks=Product()))
    def assess(g,b):return g.product(accepted=g.boolean(True),checks=g.product())
    p.continuation_publish(p.continuation_accept(candidate,p.continuation_attest(candidate,assess)))
    p.step_strategy(FixedDt(1))
    case=pops.Case('C12_decode_'+mode);case.program(p)
    frame=Rectangle('execution_context',(0.,0.),(1.,1.)).frame(Cartesian2D())
    layout=Uniform(CartesianGrid(frame=frame,cells=(4,4),periodic=PeriodicAxes(tuple(frame.axes))))
    return case,layout


@pytest.mark.parametrize('mode',('exact','approximate'))
def test_complete_nonlinear_inventory_ledger_resolves_and_emits(mode):
    from pops.codegen.program_codegen import emit_cpp_program
    from pops.codegen.program_models import ProgramModelGraph
    case,layout=build_inventory_decode_case(mode)
    plan=pops.resolve(pops.validate(case),layout=layout)
    source=emit_cpp_program(plan.time,model_graph=ProgramModelGraph.from_resolved_blocks(plan.blocks))
    assert not plan.blocks and 'std::sqrt' in source
    assert str(int(DecodeFailure.UNSELECTED)) in source


@pytest.mark.parametrize('mode',('missing_defect','duplicate_defect','hidden_conversion'))
def test_nonlinear_conversion_requires_exact_path_and_exact_defect_once(mode):
    with pytest.raises(ValueError):
        case,layout=build_inventory_decode_case(mode)
        pops.resolve(pops.validate(case),layout=layout)



def test_closed_conversion_roundtrip_revalidates_actual_references_and_guards():
    from pops.graph_functions import GraphFunction
    p,f,a=problem()
    result=checked_inventory_decode(p,**a,**f)
    inputs=Product(**{key:type_of(value) for key,value in a.items()})
    @graph(inputs=inputs,output=type_of(result))
    def convert(g,args):
        return checked_inventory_decode(g,**{key:args[key] for key in a},**f)
    restored=GraphFunction.from_data(convert.to_data(),inputs=convert.inputs,output=convert.output)
    before=p._ir_hash()
    restored(p,**a)
    assert p._ir_hash()!=before
    markers=[v for v in p._issued_values.values() if v.op=='provenance_bind']
    for marker in markers:validate_inventory_conversion(marker)
    assert len(markers)==2
