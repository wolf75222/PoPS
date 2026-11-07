"""Explicit closed equations for retained current dependencies.

A declaration relates a retained target path to named paths in the SAME candidate
continuation. It does not infer physics from names or forbid old stage values.
The graph is used for authoring and replay; a copied metadata record is no proof.
"""
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from pops.graph_functions import GraphFunction, _compatible as type_compatible, _formals
from pops.time._program.structured_control import value_signature
from pops.time._program.value_validation import require_owned
from pops.time.variant_types import Product, type_of


def _path(path):
    path=tuple(path)
    if not path or any(type(key) is not str or not key for key in path):
        raise TypeError('current dependency requires a nonempty named Product path')
    return path


def _type_at(signature,path):
    for key in path:
        if signature['vtype']!='product' or key not in dict(signature['members']):
            raise ValueError('current dependency names a missing Product path')
        signature=dict(signature['members'])[key]
    return signature


def _at(value,path):
    for key in path:value=value[key]
    return value


@dataclass(frozen=True)
class CurrentDependency:
    function: GraphFunction
    target: tuple
    inputs: Mapping

    def __post_init__(self):
        if type(self.function) is not GraphFunction:
            raise TypeError('current dependency requires an exact closed GraphFunction')
        paths={key:_path(path) for key,path in self.inputs.items()}
        if set(paths)!=set(dict(self.function.inputs.signature['members'])):
            raise TypeError('current dependency requires all and only the graph arguments')
        object.__setattr__(self,'target',_path(self.target))
        object.__setattr__(self,'inputs',MappingProxyType(paths))

    def to_data(self):
        from pops.inventory_decode import _descriptor
        return dict(target=self.target,inputs=dict(self.inputs),function=_descriptor(self.function))

    def evaluate(self,program,*,current):
        require_owned(program,current,'current dependency record',vtype='product')
        arguments={key:_at(current,path) for key,path in self.inputs.items()}
        value=self.function(program,**arguments)
        from pops.ledger_provenance import bind_contract
        return bind_contract(program,value,'current_dependency',dict(
            definition=self.to_data(),input_values=arguments,computed=value))


def current_dependencies_data(signature,declarations):
    """Normalize immutable schema requirements; path/type authority is preserved."""
    result=[];targets=[]
    for declaration in declarations:
        if type(declaration) is not CurrentDependency:
            raise TypeError('schema current dependencies require CurrentDependency objects')
        data=declaration.to_data();_validate_definition(data,signature)
        target=data['target']
        if any(target[:len(old)]==old or old[:len(target)]==target for old in targets):
            raise ValueError('current dependency target paths overlap')
        targets.append(target);result.append(data)
    return tuple(result)


def _validate_definition(data,signature=None):
    from pops.inventory_decode import _load
    if not isinstance(data,Mapping) or set(data)!={'target','inputs','function'}:
        raise ValueError('malformed current dependency declaration')
    target=_path(data['target'])
    if not isinstance(data['inputs'],Mapping):
        raise TypeError('current dependency input paths must be named')
    function=_load(data['function'])
    inputs=dict(function.inputs.signature['members'])
    if set(inputs)!=set(data['inputs']):raise ValueError('current dependency argument paths differ')
    for key,path in data['inputs'].items():
        path=_path(path)
        if signature is not None and not type_compatible(inputs[key],_type_at(signature,path)):
            raise TypeError('current dependency input path has a different type or authority')
    if signature is not None and not type_compatible(function.output.signature,_type_at(signature,target)):
        raise TypeError('current dependency target has a different type or authority')
    return function


def validate_current_operation(inputs,metadata,signature):
    """Replay equations from authenticated argument refs, never from computed metadata."""
    from pops.inventory_decode import _structure, _projection
    if len(inputs)!=1 or not isinstance(metadata,Mapping) or set(metadata)!={'definition','input_values','computed'}:
        raise ValueError('malformed current dependency application')
    value=inputs[0]
    if metadata['computed'] is not value:raise ValueError('current dependency receipt was copied to another value')
    function=_validate_definition(metadata['definition'])
    args=metadata['input_values']
    if not isinstance(args,Mapping) or set(args)!=set(dict(function.inputs.signature['members'])):
        raise ValueError('current dependency argument references differ')
    for key,arg in args.items():require_owned(value.prog,arg,'current dependency '+key)
    p,formal=_formals(Product(**{key:type_of(arg) for key,arg in args.items()}))
    expected_args={};seen={}
    for key,arg in args.items():
        identity=_projection(arg).id
        if identity not in seen:seen[identity]=formal[key]
        expected_args[key]=seen[identity]
    expected=function(p,**expected_args)
    if _structure(value,args)!=_structure(expected,expected_args):
        raise ValueError('current dependency arithmetic/control differs from its declared graph')


def _leaves(signature,path=()):
    if signature['vtype'] in ('product','variant'):
        for name,child in signature['members']:yield from _leaves(child,(*path,name))
    else:yield path


def validate_current_dependencies(candidate):
    """Recheck every declared target against the exact current record on each success path."""
    from pops.codegen.typed_delivery import delivered_count, delivered_frontiers, require_identified
    from pops.ledger_provenance import OP, _TRANSPORT, _key, _compatible, validate_contract_operation
    if candidate.op!='continuation_freeze':raise TypeError('current dependencies require the actual candidate freeze')
    base,proposal,envelope=candidate.inputs
    schema=value_signature(base)['schema']
    declarations=schema.get('current_dependencies',())
    values={v.id:candidate.prog._canonical_value(v) for v in candidate.prog._issued_values.values()}
    def terminal(value):return value.op in (OP,'continuation_read') or value.op not in _TRANSPORT
    for definition in declarations:
        _validate_definition(definition,schema['write'])
        rows=delivered_frontiers(envelope,terminal,path=('continuation',*definition['target']),
                                with_choices=True,with_projection=True)
        # An unconditionally failing candidate has no successful publication.
        for choices,frontier in rows:
            if delivered_count(frontier)!=1:raise ValueError('current dependency target lacks one exact application')
            reference=next(iter(frontier));value=values[reference.value_id]
            if reference.projection or value.op!=OP or value.attrs.get('contract_kind')!='current_dependency':
                raise ValueError('current dependency target lost its mandatory application')
            validate_contract_operation(value.inputs,value.attrs,value_signature(value))
            proof=value.attrs['contract']
            if _key(proof['definition'])!=_key(definition):
                raise ValueError('current dependency target uses a different declared graph or path')
            for name,path in definition['inputs'].items():
                argument=proof['input_values'][name]
                # Compare structural coordinates, not global reachability. Variant
                # tags are included as projection paths, so equal payloads do not
                # make different injections interchangeable.
                for leaf in _leaves(value_signature(argument)):
                    actual=delivered_frontiers(envelope,terminal,path=('continuation',*definition['target']),
                        with_choices=True,with_projection=True,
                        expand_match=lambda node,path: (argument,leaf) if node is value else None)
                    current=delivered_frontiers(envelope,terminal,path=('continuation',*path,*leaf),
                                                with_choices=True,with_projection=True)
                    actual=[(c,require_identified(r,'current dependency input')) for c,r in actual if _compatible(choices,c)]
                    current=[(c,require_identified(r,'current dependency input')) for c,r in current if _compatible(choices,c)]
                    if bool(actual)!=bool(current):raise ValueError('current dependency input has a different current variant')
                    for ac,ar in actual:
                        compatible=[cr for cc,cr in current if _compatible(choices|ac,cc)]
                        if not compatible or any(ar!=cr for cr in compatible):
                            raise ValueError('current dependency input is stale or belongs to another current path')
                    for cc,cr in current:
                        if not any(_compatible(choices|cc,ac) for ac,ar in actual):
                            raise ValueError('current dependency current path has no matching argument')
    return len(declarations)
