"""Structural coordinate algebra for exact AD on live finite native domains.

All loops below elaborate Program map/fold operations.  Python visits the *type*, never
the runtime cardinality.  A cotangent uses the primal's actual domain and Variant tag;
neither equal capacity nor equal cardinality is an authority conversion.
"""
from __future__ import annotations

from contextvars import ContextVar
from functools import wraps

_TRANSPORT_BYTES = ContextVar("pops_ad_transport_bytes", default=None)

def with_transport(byte_capacity):
    """Scope an explicit elaboration option; every use is serialized in native IR attrs."""
    if byte_capacity is not None and (type(byte_capacity) is not int or byte_capacity <= 0):
        raise ValueError("AD transport_byte_capacity must be a positive Python Int64 budget")
    def decorate(builder):
        @wraps(builder)
        def build(*args, **kwargs):
            token = _TRANSPORT_BYTES.set(byte_capacity)
            try: return builder(*args, **kwargs)
            finally: _TRANSPORT_BYTES.reset(token)
        return build
    return decorate

def transport_capacity():
    capacity = _TRANSPORT_BYTES.get()
    if capacity is None:
        raise ValueError("distributed AD requires explicit transport_byte_capacity")
    return capacity

def array_map(p, domain, body):
    if value_signature(domain).get("ownership") == "replicated":
        return p.map(domain, body)
    from pops.time._program.distributed_arrays import Distributed
    result = Distributed(p).local_map(domain,lambda q,i:q.capture(lambda r:body(r,i)),
                                      byte_capacity=transport_capacity())
    return _transport_result(p,result)

def _transport_result(p, result, prefix="differential_transport"):
    """Name internal AD transport failures without colliding with application error tags.

    The complete typed payload is retained. A structural type digest makes two
    incompatible rank/gid/cause payloads distinct; it never selects scientific behavior.
    """
    import hashlib
    import json
    from pops.time.variant_types import Failure, Result
    from pops._differential_results import _raise
    if result.vtype != 'variant':
        return p.require_success(result)
    members=dict(value_signature(result)['members'])
    success,error=ValueType(members['ok']),ValueType(members['err'])
    digest=hashlib.sha256(json.dumps(error.to_data(),sort_keys=True,separators=(',',':')).encode()).hexdigest()
    tag=prefix+'_'+digest
    errors=Failure(**{tag:error})
    return p.match(result,ok=lambda q,value:value,
        err=lambda q,cause:_raise(q,success,errors,q.variant(errors,tag,cause)))


def array_reduce(p, values, initial, body):
    if value_signature(values).get("ownership") == "replicated":
        return p.reduce(values, initial, body, order="ascending")
    return p.reduce(values, initial, body, order="global_gid_ascending", byte_capacity=transport_capacity())

from pops.time._program.structured_control import new_typed, value_signature
from pops.time.variant_types import Float64, Product, ValueType


DOMAIN_MISMATCH = 91002
TAG_MISMATCH = 91003
_DISCRETE = frozenset({'bool', 'int64', 'index', 'index_set'})


def _contains_distributed(signature):
    kind = signature['vtype']
    if kind == 'array':
        return signature.get('ownership') != 'replicated' or _contains_distributed(signature['element'])
    return kind in ('product', 'variant') and any(_contains_distributed(child) for _, child in signature['members'])


def _select_global(p, array, gid):
    """Read one element on every rank; distributed ownership is never ignored."""
    if value_signature(array).get('ownership') == 'replicated':
        return p.at(array, p.index(p.domain(array), gid))
    from pops.time.variant_types import Variant
    from pops._differential_results import _tag
    element = ValueType(value_signature(array)['element'])
    option = Variant(missing=Product(), present=element)
    tagged = array_map(p, p.domain(array), lambda q, i: q.product(gid=q.index_value(i), value=q.at(array, i)))
    selected = array_reduce(p, tagged, p.variant(option, 'missing', p.product()), lambda q, state, item:
        q.branch(item['gid'] == gid, lambda r: r.variant(option, 'present', item['value']), lambda r: state))
    return _tag(p, selected, 'present', element, lambda q, value: value)


def _structural_array_map(p, primal, body, output_element):
    """Map structure with nested collectives in globally ordered, uniform regions.

    The optional staging array allows a different result element type without inventing
    a placeholder payload. Every live destination is assigned once before projection.
    """
    sig = value_signature(primal)
    if not _contains_distributed(sig['element']):
        return array_map(p, p.domain(primal), lambda q, i: body(q, q.at(primal, i), q.index_value(i)))
    from pops.time.variant_types import Variant
    from pops._differential_results import _tag
    domain = p.domain(primal)
    option = Variant(missing=Product(), present=output_element)
    initial = array_map(p, domain, lambda q, i: q.variant(option, 'missing', q.product()))
    def step(q, current, gid):
        item = _select_global(q, primal, gid)
        result = body(q, item, gid)
        return array_map(q, domain, lambda r, i: r.branch(r.index_value(i) == gid,
            lambda z: z.variant(option, 'present', result), lambda z: z.at(current, i)))
    complete = p.fold(initial, p.size(domain), step, max_steps=sig['capacity'])
    return array_map(p, domain, lambda q, i:
        _tag(q, q.at(complete, i), 'present', output_element, lambda r, value: value))


def _tangent(signature):
    from pops.graph_derivatives import tangent_type
    return tangent_type(ValueType(signature))


def _constant(p, signature, value):
    # The zero/one inhabits the same scalar fibre, including its declared point.  No
    # multiplication by a possibly nonfinite primal is used to fabricate a zero.
    return new_typed(p, signature, 'finite_real', (), {'value': float(value)})


def require(p, value, predicate, code):
    """A native checked identity, retaining the value only on the successful branch."""
    return p.require_success(p.branch(predicate,
        lambda q: q.success(value), lambda q: q.failure(value, code)))


def same_domains(p, left, right):
    """Check the live instance even for empty domains (no element read can do that)."""
    return require(p, right, p.same_domain(p.domain(left), p.domain(right)), DOMAIN_MISMATCH)


def _tagged(p, value, tag, body, *, template):
    """Project one required tag through a guarded match, rejecting all other tags."""
    members = dict(value_signature(value)['members'])
    return p.match(value, **{name: (body if name == tag else
        lambda q, ignored: q.require_success(q.failure(template(q), TAG_MISMATCH)))
        for name in members})


def zero_like(p, primal):
    signature = value_signature(primal)
    kind = signature['vtype']
    if kind == 'scalar':
        return _constant(p, signature, 0.)
    if kind in _DISCRETE:
        return p.product()
    if kind == 'product':
        return p.product(**{key: zero_like(p, primal[key]) for key, _ in signature['members']})
    if kind == 'array':
        return _structural_array_map(p, primal, lambda q, item, i: zero_like(q, item), _tangent(signature['element']))
    if kind == 'variant':
        typ = _tangent(signature)
        return p.match(primal, **{tag: lambda q, value, tag=tag:
            q.variant(typ, tag, zero_like(q, value)) for tag, _ in signature['members']})
    raise NotImplementedError('no finite coordinate zero for '+kind)


def check_tangent(p, primal, direction):
    """Check dependent runtime shape, preserving the supplied tangent's real values."""
    signature = value_signature(primal)
    kind = signature['vtype']
    if kind in _DISCRETE or kind == 'scalar':
        return direction
    if kind == 'product':
        return p.product(**{key: check_tangent(p, primal[key], direction[key])
                            for key, _ in signature['members']})
    if kind == 'array':
        direction = same_domains(p, primal, direction)
        return _structural_array_map(p, primal, lambda q, item, i:
            check_tangent(q, item, _select_global(q, direction, i) if _contains_distributed(signature['element']) else q.at(direction, q.index(q.domain(direction),i))), _tangent(signature['element']))
    if kind == 'variant':
        typ = _tangent(signature)
        def arm(tag):
            return lambda q, value: _tagged(q, direction, tag,
                lambda r, tangent: r.variant(typ, tag, check_tangent(r, value, tangent)),
                template=lambda r: r.variant(typ, tag, zero_like(r, value)))
        return p.match(primal, **{tag: arm(tag) for tag, _ in signature['members']})
    raise NotImplementedError('no finite tangent identity check for '+kind)


def coordinate_map(p, primal, scalar, path=(), *, leaf_type=None, leaf_initial=None):
    """Return T(primal), evaluating scalar(P, coordinate-path) per live real coordinate."""
    signature = value_signature(primal)
    kind = signature['vtype']
    if kind == 'scalar':
        return scalar(p, path)
    if kind in _DISCRETE:
        return p.product()
    if kind == 'product':
        return p.product(**{key: coordinate_map(p, primal[key], scalar, (*path, ('field', key)), leaf_type=leaf_type, leaf_initial=leaf_initial)
                            for key, _ in signature['members']})
    if kind == 'array':
        if signature.get('ownership') != 'replicated':
            return _distributed_coordinate_map(p, primal, scalar, path, leaf_type=leaf_type, leaf_initial=leaf_initial)
        return _structural_array_map(p, primal, lambda q, item, i:
            coordinate_map(q, item, scalar, (*path, ('index', q.index(q.domain(primal),i))), leaf_type=leaf_type, leaf_initial=leaf_initial), mapped_coordinate_type(signature['element'], leaf_type))
    if kind == 'variant':
        typ = mapped_coordinate_type(signature, leaf_type)
        return p.match(primal, **{tag: lambda q, value, tag=tag:
            q.variant(typ, tag, coordinate_map(q, value, scalar, (*path, ('tag', tag)), leaf_type=leaf_type, leaf_initial=leaf_initial))
            for tag, _ in signature['members']})
    raise NotImplementedError('no finite coordinate traversal for '+kind)


def basis_like(p, primal, path):
    """Kronecker direction for one native coordinate, with every original live domain."""
    signature = value_signature(primal)
    kind = signature['vtype']
    if kind == 'scalar':
        if path:
            raise TypeError('coordinate path extends beyond a real scalar')
        return _constant(p, signature, 1.)
    if not path:
        raise TypeError('coordinate path ends before a real scalar')
    step, rest = path[0], path[1:]
    if kind == 'product' and step[0] == 'field':
        return p.product(**{key: (basis_like(p, primal[key], rest) if key == step[1]
                                else zero_like(p, primal[key])) for key, _ in signature['members']})
    if kind == 'array' and step[0] == 'distributed_gid':
        domain, selected_domain, gid = p.domain(primal), step[1][0], step[1][1]
        primal = require(p, primal, p.same_domain(domain, selected_domain), DOMAIN_MISMATCH)
        return _structural_array_map(p, primal, lambda q,item,i:q.branch(i==gid,
            lambda r:basis_like(r,item,rest),lambda r:zero_like(r,item)), _tangent(signature['element']))
    if kind == 'array' and step[0] == 'index':
        domain, selected = p.domain(primal), step[1]
        # The selected Index was issued by coordinate_map on this same primal.
        selected = require(p, selected, p.same_domain(domain, p.domain(selected)), DOMAIN_MISMATCH)
        def entry(q, item, i):
            return q.branch(i == q.index_value(selected),
                lambda r: basis_like(r, item, rest), lambda r: zero_like(r, item))
        return _structural_array_map(p, primal, entry, _tangent(signature['element']))
    if kind == 'variant' and step[0] == 'tag':
        typ = _tangent(signature)
        return p.match(primal, **{tag: lambda q, item, tag=tag:
            q.variant(typ, tag, basis_like(q, item, rest) if tag == step[1] else zero_like(q, item))
            for tag, _ in signature['members']})
    raise TypeError('coordinate path does not inhabit the finite primal')


def coordinate_dot(p, left, right):
    """Coordinate pairing; caller-supplied Riesz maps supply scientific dualities."""
    signature = value_signature(left)
    kind = signature['vtype']
    if kind == 'scalar':
        return left*right
    if kind == 'product':
        return sum((coordinate_dot(p, left[key], right[key]) for key, _ in signature['members']),
                   p.float64(0.))
    if kind == 'array':
        right = same_domains(p, left, right)
        entries = _structural_array_map(p, left, lambda q,item,i: coordinate_dot(q, item,
            _select_global(q,right,i) if _contains_distributed(signature['element']) else q.at(right,q.index(q.domain(right),i))), Float64)
        return array_reduce(p, entries, p.float64(0.), lambda q, carry, item: carry+item)
    if kind == 'variant':
        return p.match(left, **{tag: lambda q, item, tag=tag:
            _tagged(q, right, tag, lambda r, other: coordinate_dot(r, item, other),
                    template=lambda r: r.float64(0.)) for tag, _ in signature['members']})
    raise NotImplementedError('coordinate pairing requires tangent values, got '+kind)


def coordinate_count(p, primal):
    """Native real dimension, including runtime-sized or nested arrays."""
    signature = value_signature(primal)
    kind = signature['vtype']
    if kind == 'scalar': return p.int64(1)
    if kind in _DISCRETE: return p.int64(0)
    if kind == 'product':
        return sum((coordinate_count(p, primal[key]) for key, _ in signature['members']), p.int64(0))
    if kind == 'array':
        from pops.time.variant_types import Int64
        counts = _structural_array_map(p, primal, lambda q,item,i: coordinate_count(q,item), Int64)
        return array_reduce(p, counts, p.int64(0), lambda q, carry, item: carry+item)
    if kind == 'variant':
        return p.match(primal, **{tag: lambda q, value: coordinate_count(q, value)
                                  for tag, _ in signature['members']})
    raise NotImplementedError('no real coordinate dimension for '+kind)


def has_dynamic_coordinates(signature):
    kind = signature['vtype']
    return kind in ('array', 'variant') or (kind == 'product' and
        any(has_dynamic_coordinates(child) for _, child in signature['members']))


def mapped_coordinate_type(signature, leaf_type=None):
    """Structural type of a native coordinate map; arbitrary typed receipts are retained."""
    if leaf_type is None:
        return _tangent(signature)
    from pops.time.variant_types import Variant
    kind = signature['vtype']
    if kind == 'scalar': return leaf_type
    if kind in _DISCRETE: return Product()
    if kind == 'product':
        return Product(**{key: mapped_coordinate_type(child, leaf_type) for key, child in signature['members']})
    if kind == 'variant':
        return Variant(**{key: mapped_coordinate_type(child, leaf_type) for key, child in signature['members']})
    if kind == 'array':
        return ValueType(dict(signature, element=mapped_coordinate_type(signature['element'], leaf_type).signature))
    raise NotImplementedError('coordinate receipt map has no rule for '+kind)


def coordinate_extract(p, primal, mapped, scalar):
    """Replace every coordinate receipt by a scalar without recomputing any column."""
    signature = value_signature(primal)
    kind = signature['vtype']
    if kind == 'scalar': return scalar(p, mapped)
    if kind in _DISCRETE: return p.product()
    if kind == 'product':
        return p.product(**{key: coordinate_extract(p, primal[key], mapped[key], scalar)
                            for key, _ in signature['members']})
    if kind == 'array':
        mapped = same_domains(p, primal, mapped)
        return _structural_array_map(p, primal, lambda q,item,i:
            coordinate_extract(q,item,_select_global(q,mapped,i) if _contains_distributed(signature['element']) else q.at(mapped,q.index(q.domain(mapped),i)),scalar), _tangent(signature['element']))
    if kind == 'variant':
        typ = _tangent(signature)
        return p.match(primal, **{tag: lambda q, item, tag=tag:
            _tagged(q, mapped, tag,
                lambda r, receipt: r.variant(typ, tag, coordinate_extract(r, item, receipt, scalar)),
                template=lambda r: r.variant(typ, tag, zero_like(r, item)))
            for tag, _ in signature['members']})
    raise NotImplementedError('coordinate receipt extraction has no rule for '+kind)


def static_dimension(signature):
    """Return only a dimension proved statically; native cardinalities remain unknown."""
    kind = signature['vtype']
    if kind == 'scalar': return 1
    if kind in _DISCRETE: return 0
    if kind == 'product':
        sizes = [static_dimension(child) for _, child in signature['members']]
        return None if any(size is None for size in sizes) else sum(sizes)
    if kind in ('array', 'variant'): return None
    raise NotImplementedError('no finite real dimension for '+kind)


def _coordinate_fold(p, primal, initial, body, path):
    """Visit live coordinates in native order; the carry may contain distributed arrays."""
    sig=value_signature(primal);kind=sig['vtype']
    if kind=='scalar': return body(p,initial,path)
    if kind in _DISCRETE: return initial
    if kind=='product':
        for key,_ in sig['members']:
            initial=_coordinate_fold(p,primal[key],initial,body,(*path,('field',key)))
        return initial
    if kind=='array' and sig['ownership']=='replicated':
        domain=p.domain(primal)
        return p.fold(initial,p.size(domain),lambda q,carry,k:
            _coordinate_fold(q,q.at(primal,q.index(domain,k)),carry,body,
                             (*path,('index',q.index(domain,k)))),max_steps=sig['capacity'])
    if kind=='variant':
        return p.match(primal,**{tag:lambda q,item,tag=tag:
            _coordinate_fold(q,item,initial,body,(*path,('tag',tag))) for tag,_ in sig['members']})
    if kind=='array':
        domain=p.domain(primal)
        return p.fold(initial,p.size(domain),lambda q,carry,gid:
            _coordinate_fold(q,_select_global(q,primal,gid),carry,body,
                (*path,('distributed_gid',(domain,gid)))),max_steps=sig['capacity'])
    raise NotImplementedError('no finite coordinate fold for '+kind)


def _coordinate_assign(p, primal, current, path, value, *, leaf_type=None):
    sig=value_signature(primal);kind=sig['vtype']
    if kind=='scalar':
        if path: raise TypeError('coordinate assignment extends beyond a scalar')
        if value.vtype=='product':
            return p.product(**{key:value[key] for key,_ in value_signature(value)['members']})
        return new_typed(p,value_signature(value),'finite_real',(value,),{})
    step,rest=path[0],path[1:]
    if kind=='product' and step[0]=='field':
        return p.product(**{key:_coordinate_assign(p,primal[key],current[key],rest,value,leaf_type=leaf_type)
            if key==step[1] else current[key] for key,_ in sig['members']})
    if kind=='array':
        domain=p.domain(primal)
        current=same_domains(p,primal,current)
        if step[0]=='distributed_gid':
            selected_domain,gid=step[1]
            current=require(p,current,p.same_domain(domain,selected_domain),DOMAIN_MISMATCH)
        elif step[0]=='index':
            selected=require(p,step[1],p.same_domain(domain,p.domain(step[1])),DOMAIN_MISMATCH)
            gid=p.index_value(selected)
        else: raise TypeError('invalid array coordinate assignment')
        def assign(q,item,i):
            old=_select_global(q,current,i) if _contains_distributed(sig['element']) else q.at(current,q.index(q.domain(current),i))
            return q.branch(i==gid,
                lambda r:_coordinate_assign(r,item,old,rest,value,leaf_type=leaf_type),lambda r:old)
        return _structural_array_map(p,primal,assign,mapped_coordinate_type(sig['element'],leaf_type))
    if kind=='variant' and step[0]=='tag':
        typ=mapped_coordinate_type(sig,leaf_type)
        return p.match(primal,**{tag:lambda q,item,tag=tag:
            _tagged(q,current,tag,lambda r,old:r.variant(typ,tag,
                _coordinate_assign(r,item,old,rest,value,leaf_type=leaf_type) if tag==step[1] else old),
                template=lambda r:r.variant(typ,tag,zero_like(r,item))) for tag,_ in sig['members']})
    raise TypeError('coordinate assignment does not match the finite primal')


def _distributed_coordinate_map(p, primal, scalar, path, *, leaf_type=None, leaf_initial=None):
    """Global gid order on every rank, including empty ranks; never collective-in-local-map."""
    from pops.time.variant_types import Variant, type_of
    from pops._differential_results import _tag
    sig=value_signature(primal);domain=p.domain(primal);element=ValueType(sig['element'])
    option=Variant(missing=Product(),present=element)
    tagged=array_map(p,domain,lambda q,i:q.product(gid=q.index_value(i),value=q.at(primal,i)))
    initial=zero_like(p,primal) if leaf_type is None else _fill_coordinates(p,primal,leaf_initial)
    def cell(q,carry,gid):
        selected=array_reduce(q,tagged,q.variant(option,'missing',q.product()),lambda r,s,item:
            r.branch(item['gid']==gid,lambda w:w.variant(option,'present',item['value']),lambda w:s))
        sample=_tag(q,selected,'present',element,lambda r,item:item)
        base=(('distributed_gid',(domain,gid)),)
        def coordinate(r,current,relative):
            answer=scalar(r,(*path,*relative))
            return _coordinate_assign(r,primal,current,relative,answer,leaf_type=leaf_type)
        return _coordinate_fold(q,sample,carry,coordinate,base)
    return p.fold(initial,p.size(domain),cell,max_steps=sig['capacity'])


def _fill_coordinates(p, primal, leaf):
    if leaf is None: raise TypeError('coordinate receipt map requires its native initial receipt')
    sig=value_signature(primal);kind=sig['vtype']
    if kind=='scalar':
        return p.product(**{key:leaf[key] for key,_ in value_signature(leaf)['members']})
    if kind in _DISCRETE:return p.product()
    if kind=='product':return p.product(**{key:_fill_coordinates(p,primal[key],leaf) for key,_ in sig['members']})
    if kind=='array':
        from pops.time.variant_types import type_of
        return _structural_array_map(p,primal,lambda q,item,i:_fill_coordinates(q,item,leaf),
                                     mapped_coordinate_type(sig['element'],type_of(leaf)))
    if kind=='variant':
        from pops.time.variant_types import type_of
        typ=mapped_coordinate_type(sig,type_of(leaf))
        return p.match(primal,**{tag:lambda q,item,tag=tag:q.variant(typ,tag,_fill_coordinates(q,item,leaf))
            for tag,_ in sig['members']})
    raise TypeError('coordinate receipt map cannot fill '+kind)
