"""Explicit immutable-data snapshot boundary @1; never an executable callback encoding."""
from dataclasses import fields, is_dataclass
from decimal import Decimal
from enum import Enum
from fractions import Fraction
import json
import math
from pathlib import PurePath
from types import MappingProxyType, FunctionType


def immutable_data(value, *, active=None):
    """Retain every stored dataclass field; reject mutable, opaque and callable leaves."""
    if active is None:
        active=set()
    if value is None or type(value) in (str,bool,int):
        return value
    if type(value) is float:
        if not math.isfinite(value): raise ValueError('immutable-data requires finite floats')
        return {'$float':value.hex()}
    if type(value) is bytes: return {'$bytes':value.hex()}
    if type(value) is Fraction: return {'$fraction':[value.numerator,value.denominator]}
    if type(value) is Decimal:
        if not value.is_finite(): raise ValueError('immutable-data requires finite decimals')
        return {'$decimal':str(value)}
    if isinstance(value,Enum):
        return {'$enum':{'type':type_name(value),'name':value.name,'value':immutable_data(value.value,active=active)}}
    if isinstance(value,PurePath): return {'$path':{'type':type_name(value),'value':str(value)}}
    key=id(value)
    if key in active: raise ValueError('immutable-data reference cycle')
    active.add(key)
    try:
        if type(value) is tuple:
            return {'$tuple':[immutable_data(item,active=active) for item in value]}
        if type(value) is frozenset:
            rows=[immutable_data(item,active=active) for item in value]
            return {'$frozenset':sorted(rows,key=lambda row:json.dumps(row,sort_keys=True))}
        if type(value) is MappingProxyType:
            if any(type(key) is not str for key in value): raise TypeError('immutable-data mapping keys must be strings')
            return {'$mapping':{key:immutable_data(item,active=active) for key,item in sorted(value.items())}}
        if is_dataclass(value) and not isinstance(value,type) and type(value).__dataclass_params__.frozen:
            if callable(value):
                marker=getattr(type(value),'__pops_snapshot_immutable_data__',None)
                if type(marker) is not int or marker != 1 or not callable(getattr(value,'snapshot_data',None)):
                    raise TypeError('immutable-data rejects executable callable record without authenticated data-factory projection')
            names={field.name for field in fields(value)}
            slots=set()
            for base in type(value).__mro__:
                declared=vars(base).get('__slots__',())
                slots.update((declared,) if isinstance(declared,str) else declared)
            if any(name not in names and name not in ('__dict__','__weakref__') and hasattr(value,name) for name in slots):
                raise TypeError('immutable-data record has undeclared stored slots')
            if hasattr(value,'__dict__') and set(vars(value))-names:
                raise TypeError('immutable-data record has undeclared stored fields')
            stored = {'type':type_name(value),'implementation':record_code(type(value), active=active), 'fields':{
                name:immutable_data(getattr(value,name),active=active) for name in sorted(names)}}
            if callable(value):
                receipt = declared_data(value.snapshot_data())
                after = {'type':type_name(value),'implementation':record_code(type(value), active=active), 'fields':{
                    name:immutable_data(getattr(value,name),active=active) for name in sorted(names)}}
                if stored != after:
                    raise ValueError('immutable-data factory projection mutated stored authority')
                stored['declared'] = receipt
            return {'$record':stored}
        raise TypeError('immutable-data rejects mutable, opaque or callable leaf %s'%type_name(value))
    finally:
        active.remove(key)


def type_name(value):
    return type(value).__module__+'.'+type(value).__qualname__


def snapshot_projection(value):
    marker=getattr(type(value),'__pops_snapshot_immutable_data__',None)
    if marker is None: return None
    if type(marker) is not int or marker != 1:
        raise TypeError('unknown immutable-data snapshot contract')
    if not is_dataclass(value) or not type(value).__dataclass_params__.frozen:
        raise TypeError('immutable-data snapshot requires a frozen dataclass root')
    before=immutable_data(value)
    projection=getattr(value,'snapshot_data',None)
    if not callable(projection): raise TypeError('immutable-data snapshot requires authenticated snapshot_data')
    data=projection()
    after=immutable_data(value)
    if before != after: raise ValueError('immutable-data snapshot projection mutated stored authority')
    return {'contract':'pops.snapshot.immutable-data@1','stored':before,'declared':declared_data(data)}


def record_code(cls, *, active):
    """Data-record method bytecode identity; runtime callable leaves remain prohibited."""
    from ._snapshot_callable import callable_projection
    from ._snapshot_module_dependency import framework_dependency_projection
    from ._snapshot_module_fingerprint import module_implementation_fingerprint
    import sys

    def canonical_dependency(value, *, path, **context):
        if type(value) is dict and ('.__annotations__' in path or '.__kwdefaults__' in path or '.__dict__' in path):
            return {key:canonical_dependency(item,path=path+'.'+str(key)) for key,item in value.items()}
        module = getattr(value, '__module__', None)
        if callable(value) and isinstance(module, str):
            if module == 'pops' or module.startswith('pops.'):
                return framework_dependency_projection(value, path=path)
            if module.split('.', 1)[0] in sys.stdlib_module_names:
                return {'implementation': module_implementation_fingerprint(module, path=path)}
        from ._snapshot_canonical import _canonical
        return _canonical(value,path=path,active=active,handle_resolver=None,artifact=False)
    generated={'__init__','__repr__','__eq__','__hash__','__setattr__','__delattr__',
               '__lt__','__le__','__gt__','__ge__','__getstate__','__setstate__'}
    result={}
    for base in reversed(cls.__mro__):
        for name, member in vars(base).items():
            if name in generated: continue
            if isinstance(member,(staticmethod,classmethod)): member=member.__func__
            if isinstance(member,property): member=member.fget
            if isinstance(member,FunctionType):
                result[name]=callable_projection(member, path=cls.__module__+'.'+cls.__qualname__+'.'+name,
                    active=active, handle_resolver=None, artifact=False,
                    canonical=canonical_dependency, dependency_cache={})
    return result


def declared_data(value, active=None):
    """Snapshot receipt is strict data, not another object/behavior escape hatch."""
    if active is None: active=set()
    if type(value) in (dict,list,tuple,MappingProxyType):
        if id(value) in active: raise ValueError('immutable-data declared receipt cycle')
        active.add(id(value))
        try:
            if type(value) in (dict,MappingProxyType):
                if any(type(key) is not str for key in value): raise TypeError('immutable-data receipt keys must be strings')
                return {key:declared_data(item,active) for key,item in value.items()}
            return [declared_data(item,active) for item in value]
        finally: active.remove(id(value))
    if is_dataclass(value) or callable(value): raise TypeError('immutable-data receipt cannot carry records or callbacks')
    return immutable_data(value)
