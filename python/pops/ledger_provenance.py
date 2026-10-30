"""Mandatory provenance of explicitly declared inventory applications.

This is a compiler check on the actual finite graph, never an equality oracle.
The generic alias opcode retains a contract through graph codecs and rebuilds.
An unannotated user Product makes no claim to be a certified ledger.
"""
from collections.abc import Mapping
from dataclasses import dataclass

from pops.time.values import ProgramValue
from pops.time._program.structured_control import new_typed, value_signature, same_signature
from pops.time._program.value_validation import require_owned

OP = 'provenance_bind'
KINDS = frozenset(('inventory_ledger', 'inventory_application', 'inventory_conversion', 'current_dependency'))

@dataclass(frozen=True)
class InventoryBinding:
    recipient: ProgramValue
    path: tuple[str, ...]
    conversion: ProgramValue | None = None
    def __post_init__(self):
        path=tuple(self.path)
        if not path or any(type(k) is not str or not k for k in path):
            raise TypeError('InventoryBinding requires a nonempty exact Product path')
        if not isinstance(self.recipient,ProgramValue):
            raise TypeError('InventoryBinding recipient must be an exact native value')
        if self.conversion is not None and (not isinstance(self.conversion,ProgramValue)
                or self.conversion.prog is not self.recipient.prog):
            raise TypeError('InventoryBinding conversion belongs to another Program')
        object.__setattr__(self,'path',path)


def _refs(value):
    if isinstance(value,ProgramValue):yield value
    elif isinstance(value,Mapping):
        for item in value.values():yield from _refs(item)
    elif isinstance(value,(tuple,list)):
        for item in value:yield from _refs(item)


def _packet_reference_scanner():
    """Memoize metadata reachability for one read-only packet prefilter.

    Shared structural metadata is a DAG. Only this boolean reachability query may
    collapse repeated references; `_refs` retains its ordered multiplicity for
    contract validation and positional comparisons. Strong source references keep
    identity keys live, and nothing survives the enclosing delivery check.
    """
    cached = {}
    active = set()

    def references(value):
        if isinstance(value, ProgramValue):
            return (value,)
        if not isinstance(value, (Mapping, tuple, list)):
            return ()
        marker = id(value)
        previous = cached.get(marker)
        if previous is not None:
            assert previous[0] is value
            return previous[1]
        if marker in active:
            raise ValueError('cyclic metadata in inventory packet reachability')
        active.add(marker)
        try:
            result = []
            seen = set()
            items = value.values() if isinstance(value, Mapping) else value
            for item in items:
                for ref in references(item):
                    identity = id(ref)
                    if identity not in seen:
                        seen.add(identity)
                        result.append(ref)
            found = tuple(result)
            cached[marker] = (value, found)
            return found
        finally:
            active.remove(marker)

    return references


def contract_json_data(value):
    if isinstance(value,ProgramValue):return {'program_value_id':value.id}
    if isinstance(value,Mapping):return {k:contract_json_data(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [contract_json_data(v) for v in value]
    return value


def _key(value):
    # This key is invocation-local; native loop frames are compared separately.
    if isinstance(value,ProgramValue):return ('ssa',value.id)
    if isinstance(value,Mapping):return tuple((k,_key(v)) for k,v in sorted(value.items()))
    if isinstance(value,(tuple,list)):return tuple(_key(v) for v in value)
    return value


def increment_contract(increment):
    from pops.finite_contributions import FiniteInterval,LinearMap
    from pops.inventory_decode import _descriptor
    contributions=[]
    for c in increment.contributions:
        if type(c.interval) is not FiniteInterval:
            raise NotImplementedError('persistent ledger contract requires explicit finite interval endpoints')
        contributions.append(dict(id=c.id,value=c.value,begin=c.interval.begin,end=c.interval.end,
                                  clock=c.interval.clock,measure=c.measure,kind=c.kind))
    deliveries=[]
    for d in increment.deliveries:
        index=next(i for i,c in enumerate(increment.contributions) if c is d.contribution)
        action=(dict(kind='differential',graph=_descriptor(d.linear_map.function)['graph'],
                     inputs=d.linear_map.function.inputs.signature, output=d.linear_map.function.output.signature,
                     transport_byte_capacity=d.linear_map.transport_byte_capacity, at=d.linear_map.at,wrt=d.linear_map.wrt,
                     differentiate_iterations=d.linear_map.differentiate_iterations)
                if type(d.linear_map) is LinearMap else dict(kind='dense',matrix=d.linear_map))
        deliveries.append(dict(contribution=index,recipient=d.recipient,action=action))
    return dict(schema='pops.inventory-ledger@1',contributions=tuple(contributions),deliveries=tuple(deliveries))


def _validate_ledger(ledger,program):
    if not isinstance(ledger,Mapping) or set(ledger)!={'schema','contributions','deliveries'} or ledger['schema']!='pops.inventory-ledger@1':
        raise TypeError('malformed mandatory inventory ledger contract')
    cs=ledger['contributions'];ds=ledger['deliveries']
    if len({c['id'] for c in cs})!=len(cs):raise ValueError('duplicate ledger contribution')
    seen=set()
    for c in cs:
        if set(c)!={'id','value','begin','end','clock','measure','kind'} or c['measure']!='increment':
            raise TypeError('published ledger requires integrated contributions and exact interval')
        if type(c['id']) is not str or not c['id'] or c['kind'] not in ('exchange','production','defect'):
            raise TypeError('malformed contribution identity or kind')
        from pops.finite_contributions import FiniteInterval, inventory
        inventory(c['value'],'declared contribution')
        FiniteInterval(c['begin'],c['end'],c['clock'])
        if any(_key(c[k]) != _key(cs[0][k]) for k in ('begin','end','clock')):
            raise ValueError('ledger contributions must share their exact declared interval')
    for d in ds:
        if set(d)!={'contribution','recipient','action'} or type(d['contribution']) is not int or not 0<=d['contribution']<len(cs):
            raise TypeError('malformed ledger delivery')
        key=(d['contribution'],d['recipient'].id)
        if key in seen:raise ValueError('duplicate ledger delivery')
        seen.add(key)
    if {i for i,_ in seen}!=set(range(len(cs))):raise ValueError('undelivered ledger contribution')
    for ref in _refs(ledger):require_owned(program,ref,'inventory contract reference')


def bind_contract(program,value,kind,metadata):
    require_owned(program,value,'contract value')
    attrs=dict(contract_kind=kind,contract=dict(metadata))
    validate_contract_operation((value,),attrs,value_signature(value))
    return new_typed(program,value_signature(value),OP,(value,),attrs,point=value.point)


def validate_contract_operation(inputs,attrs,signature):
    if len(inputs)!=1 or attrs.get('contract_kind') not in KINDS or not isinstance(attrs.get('contract'),Mapping):
        raise TypeError('provenance_bind requires its complete mandatory contract')
    value=inputs[0];meta=attrs['contract'];kind=attrs['contract_kind']
    if not same_signature(signature,value_signature(value)):
        raise TypeError('provenance_bind must preserve its exact input type')
    for ref in _refs(meta):require_owned(value.prog,ref,'provenance contract reference')
    if kind=='inventory_application':
        if set(meta)!={'ledger','recipient','computed','byte_capacity'} or meta['computed'] is not value:
            raise ValueError('application proof was copied to a different numerical result')
        _validate_ledger(meta['ledger'],value.prog)
        if not any(d['recipient'] is meta['recipient'] for d in meta['ledger']['deliveries']):
            raise ValueError('application recipient has no exact ledger delivery')
        validate_application(value, meta)
    elif kind=='inventory_ledger':
        if set(meta)!={'ledger','packet','bindings'} or meta['packet'] is not value:
            raise ValueError('ledger proof was copied to another packet')
        _validate_ledger(meta['ledger'],value.prog)
        # The wire packet is prescribed by the original contribution values.
        if value.op!='product' or tuple(value.attrs['fields'])!=tuple(c['id'] for c in meta['ledger']['contributions']):
            raise ValueError('ledger packet lost its exact contribution structure')
        for packed,c in zip(value.inputs,meta['ledger']['contributions']):
            if packed.op!='product' or tuple(packed.attrs['fields'])!=('value','interval') or packed.inputs[0] is not c['value']:
                raise ValueError('ledger packet has a substituted contribution value')
            window=packed.inputs[1]
            if window.op!='product' or tuple(window.attrs['fields'])!=('begin','end') or tuple(x.id for x in window.inputs)!=(c['begin'].id,c['end'].id):
                raise ValueError('ledger packet has a substituted interval')
        recipients={d['recipient'].id for d in meta['ledger']['deliveries']}
        found=set()
        for binding in meta['bindings']:
            if set(binding)!={'recipient','path','conversion'} or binding['recipient'].id not in recipients or binding['recipient'].id in found:
                raise ValueError('binding must name one exact declared recipient')
            found.add(binding['recipient'].id)
            InventoryBinding(binding['recipient'],binding['path'],binding['conversion'])
    elif kind=='current_dependency':
        from pops.current_dependencies import validate_current_operation
        validate_current_operation(inputs,meta,signature)
    elif kind=='inventory_conversion':
        from pops.inventory_decode import validate_inventory_conversion_operation
        validate_inventory_conversion_operation(inputs,meta,signature)



def _map_refs(data, transform):
    if isinstance(data,ProgramValue):return transform(data)
    if isinstance(data,Mapping):return {k:_map_refs(v,transform) for k,v in data.items()}
    if isinstance(data,(tuple,list)):return tuple(_map_refs(v,transform) for v in data)
    return data


def realize_application(program, ledger, recipient, *, byte_capacity=None):
    """One generic implementation used for actual algebra and mandatory reconstruction."""
    from pops.finite_contributions import LinearMap, FiniteInterval, add, inventory
    from pops.graph_functions import GraphFunction
    from pops.time.variant_types import ValueType
    inventory(recipient,'inventory recipient')
    value=recipient
    for delivery in ledger['deliveries']:
        if delivery['recipient'] is not recipient:continue
        action=delivery['action']
        if action['kind']!='differential':
            raise NotImplementedError('finite application requires a closed differential action')
        from pops.inventory_decode import _load
        function=_load(dict(graph=action['graph'],inputs=action['inputs'],output=action['output']))
        # Closed original graph witnesses were authenticated above.
        linear=LinearMap.differential(function,at=action['at'],wrt=action['wrt'],
            transport_byte_capacity=action['transport_byte_capacity'],
            differentiate_iterations=action['differentiate_iterations'])
        contribution=ledger['contributions'][delivery['contribution']]
        value=add(program,value,linear(program,contribution['value']),byte_capacity=byte_capacity)
    if ledger['contributions']:
        first=ledger['contributions'][0]
        value=FiniteInterval(first['begin'],first['end'],first['clock']).bind(program,value,name='inventory_application')
    return program.product(value=value)['value']


def validate_application(value, metadata):
    """Rebuild the declared linear deliveries and compare all actual arithmetic/control.

    A user may edit the closed equations, but cannot copy a receipt to another result
    by editing a free expected-value reference: the declared equations are replayed.
    """
    from pops.graph_functions import _formals
    from pops.time.variant_types import Product, type_of
    from pops.inventory_decode import _structure
    bindings={}
    for ref in (*_refs(metadata['ledger']),metadata['recipient']):
        bindings.setdefault('v'+str(ref.id),ref)
    p,formal=_formals(Product(**{k:type_of(v) for k,v in bindings.items()}))
    expected_args={k:formal[k] for k in bindings}
    mapping={v.id:expected_args[k] for k,v in bindings.items()}
    ledger=_map_refs(metadata['ledger'],lambda v:mapping[v.id])
    expected=realize_application(p,ledger,mapping[metadata['recipient'].id],byte_capacity=metadata['byte_capacity'])
    if _structure(value,bindings)!=_structure(expected,expected_args):
        raise ValueError('inventory application arithmetic/control differs from its declared deliveries')

def rebuild_contract_primal(program, value, attrs, *, replacements=None):
    """Reissue a primal certificate after algebraic AD adds its own control domain.

    The tangent never enters these declarations. Reusing the exact recipient keeps
    the instance of the primal inventory domain, including an empty native Array.
    """
    kind=attrs['contract_kind'];meta=dict(attrs['contract'])
    old_value=value
    if kind=='inventory_application':
        value=realize_application(program,meta['ledger'],meta['recipient'],byte_capacity=meta['byte_capacity'])
        meta['computed']=value
    elif kind=='inventory_ledger':
        meta['packet']=value
    elif kind=='inventory_conversion':
        from pops.inventory_decode import _proof, _load
        functions={key:_load(item) for key,item in meta['functions'].items()}
        checked=_proof(functions,meta['input_values'],meta['byte_capacity'])(program,**meta['input_values'])
        value=checked['value']
        defect=checked['defect']
        if replacements is not None:
            replacements.extend(((meta['checked'],checked),(meta['defect'],defect)))
        meta.update(checked=checked,retained_input=value,defect=defect)
    elif kind=='current_dependency':
        from pops.current_dependencies import _validate_definition
        function=_validate_definition(meta['definition'])
        value=function(program,**meta['input_values'])
        meta['computed']=value
    else:
        raise NotImplementedError('AD primal provenance rule requires its closed generic contract')
    if replacements is not None:
        replacements.append((old_value,value))
    return bind_contract(program,value,kind,meta)


def bind_application(program,value,increment,recipient,*,byte_capacity=None):
    return bind_contract(program,value,'inventory_application',dict(
        ledger=increment_contract(increment),recipient=recipient,computed=value,byte_capacity=byte_capacity))


def bind_ledger(program,packet,increment,bindings=()):
    records=[]
    for binding in bindings:
        if type(binding) is not InventoryBinding:raise TypeError('ledger bindings require InventoryBinding')
        records.append(dict(recipient=binding.recipient,path=binding.path,conversion=binding.conversion))
    return bind_contract(program,packet,'inventory_ledger',dict(
        ledger=increment_contract(increment),packet=packet,bindings=tuple(records)))


def _projection(value,base,path=(),active=None):
    """Resolve only actual Product transport to the exact candidate Base read."""
    active=set() if active is None else active
    key=(value.id,path)
    if key in active:return None
    active.add(key)
    try:
        if value.op=='product_get':return _projection(value.inputs[0],base,(value.attrs['field'],)+path,active)
        if value.op=='product' and path:
            fields=tuple(value.attrs['fields'])
            return _projection(value.inputs[fields.index(path[0])],base,path[1:],active)
        if value.op=='bundle_validate':return _projection(value.inputs[0],base,path,active)
        if value.op=='bundle_remap_validate':return _projection(value.inputs[1],base,path,active)
        if value.op=='continuation_read' and value.inputs[0] is base:return path
        return None
    finally:active.remove(key)


def _compatible(left,right):
    choices=left|right
    return not any((key,not arm) in choices for key,arm in choices)


_TRANSPORT=frozenset(('bundle_validate','bundle_remap_validate','product','product_get','branch','result_match','where','region_argument',
    'variant_construct','variant_payload','variant_require_success','variant_unreachable',
    'result_ok','result_error','result_success','result_payload','iteration_success','iteration_payload',
    'capture_result','structured_fold','structured_iterate','acceptance_guard',
    'continuation_candidate_value','continuation_freeze','continuation_transform',
    'continuation_accepted_candidate','continuation_accept','continuation_publish',
    'continuation_child_base','continuation_read','continuation_open','continuation_consume_increment'))



def _leaf_paths(signature, path=()):
    if signature['vtype']=='product':
        for name,child in signature['members']:
            yield from _leaf_paths(child,(*path,name))
    else:yield path


def _validate_packet_delivery(envelope, values):
    """A packed ledger can be transported/reconstructed, never silently reinterpreted.

    Check every retained coordinate against the original packed coordinate. Merely
    using an old packet in a new expression is not an application certificate.
    """
    from pops.codegen.typed_delivery import DeliveryReference,delivered_frontiers,require_identified
    signature=dict(value_signature(envelope)['members'])['increment']
    def is_ledger(v):return v.op==OP and v.attrs.get('contract_kind')=='inventory_ledger'
    def terminal(v):return is_ledger(v) or v.op=='continuation_consume_increment' or (v.op not in _TRANSPORT and v.op!=OP)
    # A user Product is not a ledger merely because it contains a numerical
    # reduction. Only ask the exact delivery analyser to track a certified packet
    # when such a packet is reachable through the actual computation. Contract
    # metadata is not a numerical edge; consumed child packets are independent
    # authenticated frontiers. This prefilter never grants a certificate.
    reachable={}
    metadata_refs = _packet_reference_scanner()
    def has_packet(node, active=None):
        if node.id in reachable:return reachable[node.id]
        if is_ledger(node):return True
        if node.op=='continuation_consume_increment':return False
        active=set() if active is None else active
        if node.id in active:return True
        active.add(node.id)
        dependencies=list(node.inputs)
        if node.op!=OP:
            for name,attribute in node.attrs.items():
                if not name.endswith('_block') and not name.startswith(('scientific_','realization_')):
                    dependencies.extend(metadata_refs(attribute))
        found=any(has_packet(child,active) for child in dependencies)
        active.remove(node.id);reachable[node.id]=found
        return found
    if not has_packet(envelope):return ()
    obligations={}
    ledger_rows=set()
    for output_path in _leaf_paths(signature):
        rows=delivered_frontiers(envelope,terminal,path=('increment',*output_path),
                                 with_choices=True,with_projection=True)
        for choices,frontier in rows:
            for reference in frontier:
                node=values[reference.value_id]
                if not is_ledger(node):
                    if node.op=='continuation_consume_increment':continue
                    if not has_packet(node):continue
                    ancestry=delivered_frontiers(node,lambda v:is_ledger(v) or v.op=='continuation_consume_increment')
                    if any(any(is_ledger(values[r.value_id]) for r in row) for row in ancestry):
                        raise ValueError('declared ledger packet was numerically modified or stripped')
                    continue
                require_identified((reference,),'declared ledger packet')
                ledger_rows.add((choices,DeliveryReference(reference.value_id,reference.invocation)))
                projection=reference.projection
                if len(output_path)<len(projection) or (projection and output_path[-len(projection):]!=projection):
                    raise ValueError('declared ledger packet lost its exact coordinate names/path')
                prefix=output_path[:-len(projection)] if projection else output_path
                key=(reference.value_id,reference.invocation,prefix,choices)
                obligations.setdefault(key,set()).add(projection)
    # A packet cannot be shortened to one contribution/interval and keep the claim.
    for (node_id,invocation,prefix,choices),seen in obligations.items():
        expected=set(_leaf_paths(value_signature(values[node_id])))
        if seen!=expected:raise ValueError('declared ledger packet has incomplete retained coordinates')
    return tuple((choices,frozenset((reference,))) for choices,reference in ledger_rows)


def _is_exact_transport(value, target):
    from pops.codegen.typed_delivery import delivered_count, delivered_frontiers
    def stop(node):return node is target or (node.op not in _TRANSPORT and node.op!=OP)
    rows=delivered_frontiers(value,stop)
    return bool(rows) and all(delivered_count(row)==1 and next(iter(row)).value_id==target.id for row in rows)


def _identity_action(action):
    from pops.inventory_decode import _load, _projection as unproject
    if action.get('kind')!='differential':return False
    function=_load(dict(graph=action['graph'],inputs=action['inputs'],output=action['output']))
    from pops.graph_functions import _formals
    p,formal=_formals(function.inputs)
    args={key:formal[key] for key,_ in function.inputs.signature['members']}
    result=function(p,**args)
    return unproject(result) is unproject(args[action['wrt']])


def _recipient_ledger(ledger, recipient, *, omit=()):
    """Preserve ordered multiplicity for one exact recipient, independent of other ports."""
    deliveries=[d for d in ledger['deliveries'] if d['recipient'] is recipient and d['contribution'] not in omit]
    used={d['contribution'] for d in deliveries}
    indices=[i for i in range(len(ledger['contributions'])) if i in used]
    positions={index:position for position,index in enumerate(indices)}
    return dict(ledger,contributions=tuple(ledger['contributions'][i] for i in indices),
                deliveries=tuple(dict(d,contribution=positions[d['contribution']]) for d in deliveries))


def _context_rows(envelope, *, path, replacements, choices):
    """Follow explicit contract refs in their owner's actual region/loop frame."""
    from pops.codegen.typed_delivery import delivered_frontiers, require_identified
    def terminal(value):return value.op in (OP,'continuation_read') or value.op not in _TRANSPORT
    def expand(value, projection):
        item=replacements.get(value.id)
        return None if item is None else (item,())
    values={v.id:envelope.prog._canonical_value(v) for v in envelope.prog._issued_values.values()}
    rows=delivered_frontiers(envelope,terminal,path=path,with_choices=True,with_projection=True,expand_match=expand)
    # Other complete ledger packets in a Product are independent obligations.
    # They are not data dependencies of the specific contract reference expanded here.
    return [(c,frozenset(r for r in require_identified(refs,'inventory contract reference') if not (
             values[r.value_id].op==OP and values[r.value_id].attrs.get('contract_kind')=='inventory_ledger')))
            for c,refs in rows if _compatible(choices,c)]


def _same_context_rows(left,right,choices):
    if bool(left)!=bool(right):return False
    for lc,lr in left:
        compatible=[rr for rc,rr in right if _compatible(choices|lc,rc)]
        if not compatible or any(lr!=rr for rr in compatible):return False
    return all(any(_compatible(choices|rc,lc) for lc,lr in left) for rc,rr in right)


def _conversion_obligation(applied, binding, ledger, recipient, base, path, *, envelope, pack, choices):
    from pops.inventory_decode import validate_inventory_conversion
    data=validate_inventory_conversion(applied)
    if binding is None or binding['conversion'] is None:
        raise ValueError('retained conversion differs from its exact explicit inventory binding')
    retained_rows=_context_rows(envelope,path=('continuation',*path),replacements={},choices=choices)
    binding_rows=_context_rows(envelope,path=('increment',),
        replacements={pack.id:binding['conversion']},choices=choices)
    if not _same_context_rows(binding_rows,retained_rows,choices):
        raise ValueError('inventory conversion binding belongs to another native invocation')
    if data['recipient'] is not recipient or _projection(data['base_value'],base)!=path:
        raise ValueError('inventory conversion targets another base value or continuation path')
    application_rows=_context_rows(envelope,path=('continuation',*path),
        replacements={applied.id:data['application']},choices=choices)
    applications=[refs for selected,refs in application_rows]
    if not applications or any(len(row)!=1 for row in applications):
        raise ValueError('inventory conversion lacks one exact original application')
    application_ids={next(iter(row)).value_id for row in applications}
    if len(application_ids)!=1:raise ValueError('inventory conversion original application is ambiguous')
    values={v.id:applied.prog._canonical_value(v) for v in applied.prog._issued_values.values()}
    original=values[next(iter(application_ids))]
    if original.op!=OP or original.attrs.get('contract_kind')!='inventory_application':
        raise ValueError('inventory conversion original value has no application certificate')
    validate_contract_operation(original.inputs,original.attrs,value_signature(original))
    original_contract=original.attrs['contract']
    if original_contract['recipient'] is not recipient:
        raise ValueError('inventory conversion original application belongs to another recipient')
    defect_rows=_context_rows(envelope,path=('continuation',*path),
        replacements={applied.id:data['defect']},choices=choices)
    indices=[i for i,c in enumerate(ledger['contributions']) if _same_context_rows(
        _context_rows(envelope,path=('increment',),replacements={pack.id:c['value']},choices=choices),
        defect_rows,choices)]
    if len(indices)!=1:raise ValueError('conversion defect must be declared exactly once in final ledger')
    index=indices[0];defect=ledger['contributions'][index]
    if defect['kind']!='defect':raise ValueError('conversion residual requires an explicit defect contribution')
    deliveries=[d for d in ledger['deliveries'] if d['contribution']==index]
    if len(deliveries)!=1 or deliveries[0]['recipient'] is not recipient or not _identity_action(deliveries[0]['action']):
        raise ValueError('conversion defect must have one exact identity delivery to its recipient')
    old=original_contract['ledger']
    if old['contributions']:
        window=old['contributions'][0]
        if any(_key(defect[k])!=_key(window[k]) for k in ('begin','end','clock')):
            raise ValueError('conversion defect changed the application interval')
    stripped=_recipient_ledger(ledger,recipient,omit=(index,))
    original_recipient=_recipient_ledger(old,recipient)
    if _key(stripped)!=_key(original_recipient):
        raise ValueError('conversion ledger differs from original application plus its measured defect')
    # Equal static producer IDs alone cannot certify a carried previous iteration.
    # Compare every live parameter of the two declared ledgers in its actual owner frame.
    for prior,current in zip(_refs(original_recipient),_refs(stripped)):
        prior_rows=_context_rows(envelope,path=('continuation',*path),
            replacements={applied.id:data['application'],original.id:prior},choices=choices)
        current_rows=_context_rows(envelope,path=('increment',),
            replacements={pack.id:current},choices=choices)
        if not _same_context_rows(prior_rows,current_rows,choices):
            raise ValueError('conversion original application and final ledger use different native inputs')

def validate_candidate(candidate):
    """Require each declared ledger's application at its exact retained recipient path."""
    from pops.codegen.typed_delivery import delivered_count, delivered_frontiers, require_identified
    if candidate.op!='continuation_freeze':raise TypeError('candidate validation requires its exact freeze')
    base,proposal,envelope=candidate.inputs
    values={v.id:candidate.prog._canonical_value(v) for v in candidate.prog._issued_values.values()}
    rows=_validate_packet_delivery(envelope, values)
    count=0
    for choices,refs in rows:
        for reference in refs:
            pack=values[reference.value_id]
            if pack.op=='continuation_consume_increment':continue
            validate_contract_operation(pack.inputs,pack.attrs,value_signature(pack))
            meta=pack.attrs['contract'];ledger=meta['ledger']
            explicit={b['recipient'].id:b for b in meta['bindings']}
            for recipient in {d['recipient'].id:d['recipient'] for d in ledger['deliveries']}.values():
                binding=explicit.get(recipient.id)
                path=tuple(binding['path']) if binding else _projection(recipient,base)
                if not path:raise ValueError('ledger recipient has no authenticated continuation path; supply InventoryBinding')
                conversion=binding['conversion'] if binding else None
                if conversion is None and _projection(recipient,base)!=path:
                    raise ValueError('ledger binding does not target its exact candidate base recipient')
                def terminal(v):
                    return v.op==OP or (v.op not in _TRANSPORT and v.vtype not in ('bool','int64'))
                deliveries=delivered_frontiers(envelope,terminal,path=('continuation',*path),with_choices=True)
                applicable=[(c,r) for c,r in deliveries if _compatible(choices,c)]
                if not applicable:raise ValueError('ledger recipient has no retained successful value')
                for selected,frontier in applicable:
                    if delivered_count(frontier)!=1:raise ValueError('retained inventory has no unique terminal application proof')
                    applied_ref=next(iter(frontier));applied=values[applied_ref.value_id]
                    require_identified((applied_ref,reference),'ledger/application invocation')
                    if applied.op!=OP:raise ValueError('retained inventory is not the declared ledger application')
                    validate_contract_operation(applied.inputs,applied.attrs,value_signature(applied))
                    if applied_ref.invocation!=reference.invocation:
                        raise ValueError('ledger/application belong to different native loop invocations')
                    if applied.attrs['contract_kind']=='inventory_conversion':
                        _conversion_obligation(applied,binding,ledger,recipient,base,path,
                            envelope=envelope,pack=pack,choices=choices|selected)
                        continue
                    if applied.attrs['contract_kind']!='inventory_application':
                        raise ValueError('retained inventory does not carry an application contract')
                    proof=applied.attrs['contract']
                    if _key(_recipient_ledger(proof['ledger'],recipient))!=_key(_recipient_ledger(ledger,recipient)) or proof['recipient'] is not recipient:
                        raise ValueError('retained inventory uses a different ledger version or recipient')
                count+=1
    from pops.current_dependencies import validate_current_dependencies
    dependencies = validate_current_dependencies(candidate)
    return dict(current_dependencies=dependencies,status='bound_ledger' if count else 'uncertified_user_product',recipient_obligations=count,
                property='exact_application_provenance_not_numerical_equivalence')


def validate_program_contracts(program):
    values={v.id:program._canonical_value(v) for v in program._issued_values.values()}
    for value in values.values():
        if value.op==OP:
            validate_contract_operation(value.inputs,value.attrs,value_signature(value))
            if value.attrs['contract_kind']=='inventory_conversion':
                from pops.inventory_decode import validate_inventory_conversion
                validate_inventory_conversion(value)
        elif value.op=='continuation_freeze':
            actual=validate_candidate(value)
            if value.attrs.get('increment_proof_v15')!=actual:
                raise ValueError('candidate ledger proof is missing, stale, or malformed')
