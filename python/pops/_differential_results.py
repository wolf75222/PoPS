"""Closed Result transport for forward AD: authentic errors and their tangents stay distinct."""
from pops.time._program.structured_control import value_signature
from pops.time.variant_types import Product, Result, ValueType, Variant, type_of


def _tangent(typ):
    from pops.graph_derivatives import tangent_type
    return tangent_type(typ)


def pair_type(typ):
    return Product(value=typ, tangent=_tangent(typ))


def _paired_tag(typ, tag):
    # A differentiated error payload is a different type from the original error.
    # Distinguish them at every derivative order without overwriting an application
    # tag or a fresh differentiation-domain failure in the same captured region.
    import hashlib
    import json
    key=json.dumps(typ.to_data(),sort_keys=True,separators=(',',':')).encode()
    return 'differentiated_'+hashlib.sha256(key).hexdigest()+'_'+tag


def error_type(typ):
    sig = typ.signature
    if sig['vtype'] != 'variant':
        return Variant(**{_paired_tag(typ,'user'):pair_type(typ)})
    return Variant(**{('runtime' if tag=='runtime' else _paired_tag(typ,tag)):
                      ValueType(child) if tag == 'runtime' else pair_type(ValueType(child))
                      for tag, child in sig['members']})


def _raise(p, output, errors, error):
    return p.require_success(p.variant(Result(output, errors), 'err', error))


def _tag(p, value, tag, output, body):
    from pops.graph_derivatives import DerivativeErrors
    def wrong(q, ignored):
        return _raise(q, output, DerivativeErrors,
            q.variant(DerivativeErrors, 'differential', q.product(code=q.int64(91003))))
    return p.match(value, **{key:body if key == tag else wrong
                            for key, _ in value_signature(value)['members']})


def _pack_error(p, value, tangent):
    typ = type_of(value); paired = error_type(typ)
    if value.vtype != 'variant':
        return p.variant(paired, _paired_tag(typ,'user'), p.product(value=value, tangent=tangent))
    def arm(tag):
        return lambda q, payload: (q.variant(paired, tag, payload) if tag == 'runtime' else
            _tag(q, tangent, tag, paired, lambda r, direction:
                r.variant(paired, _paired_tag(typ,tag), r.product(value=payload, tangent=direction))))
    return p.match(value, **{tag:arm(tag) for tag, _ in typ.signature['members']})


def _unpack_error(p, paired, original):
    from pops._finite_differential import zero_like
    output = pair_type(original)
    if paired.vtype != 'variant':
        from pops._finite_differential import static_dimension
        if type_of(paired) == original and static_dimension(original.signature) == 0:
            return p.product(value=paired,tangent=zero_like(p,paired))
        raise TypeError('unpaired Result error cannot fabricate a real payload derivative')
    original_members = dict(original.signature.get('members', ()))
    decoded_tags={('runtime' if tag=='runtime' else _paired_tag(original,tag)):tag for tag in original_members}
    def arm(tag):
        if original.signature['vtype'] != 'variant' and tag == _paired_tag(original,'user'):
            return lambda q, payload: payload
        if original.signature['vtype'] == 'variant' and tag in decoded_tags:
            original_tag=decoded_tags[tag]
            def project(q, payload):
                value = payload if original_tag == 'runtime' else payload['value']
                direction = zero_like(q, value) if original_tag == 'runtime' else payload['tangent']
                return q.product(value=q.variant(original, original_tag, value),
                    tangent=q.variant(_tangent(original), original_tag, direction))
            return project
        # Extra AD-domain errors are effects of differentiation. They must propagate,
        # never be disguised as one of the original application's mathematical errors.
        return lambda q, payload: _raise(q, output, type_of(paired),
                                         q.variant(type_of(paired), tag, payload))
    return p.match(paired, **{tag:arm(tag) for tag, _ in value_signature(paired)['members']})


def paired_result(p, value, tangent):
    """Result(T,E), tangent(Result(T,E)) -> Result((T,Tdot), paired(E))."""
    sig = value_signature(value)
    members = dict(sig['members'])
    success, error = ValueType(members['ok']), ValueType(members['err'])
    output = Result(pair_type(success), error_type(error))
    def ok(q, item):
        return _tag(q, tangent, 'ok', output, lambda r, direction:
            r.variant(output, 'ok', r.product(value=item, tangent=direction)))
    def err(q, failure):
        return _tag(q, tangent, 'err', output, lambda r, direction:
            r.variant(output, 'err', _pack_error(r, failure, direction)))
    return p.match(value, ok=ok, err=err)


def split_result(p, paired, original):
    """Recover the full primal Result and tangent, retaining runtime failures faithfully."""
    error = ValueType(dict(original.signature['members'])['err'])
    tangent_type = _tangent(original)
    def ok(q, item):
        return q.product(value=q.variant(original, 'ok', item['value']),
                         tangent=q.variant(tangent_type, 'ok', item['tangent']))
    def err(q, failure):
        split = _unpack_error(q, failure, error)
        return q.product(value=q.variant(original, 'err', split['value']),
                         tangent=q.variant(tangent_type, 'err', split['tangent']))
    return p.match(paired, ok=ok, err=err)
