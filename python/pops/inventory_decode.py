"""Checked reconstruction of a finite inventory, as ordinary closed Program algebra.

The equations are the encode GraphFunction. This module creates the original residual
itself; a decoder's successful return is not evidence of that equation. A successful
conversion retains its measured defect. Its publication requires a ledger containing
that exact defect once, delivered identically to the same recipient, without applying
it to the already reconstructed value. Ledger/path enforcement belongs to
ledger_provenance, not to this numerical conversion.

No numerical Python evaluator, model dispatch, native solver or C++ opcode is added.
"""
from collections.abc import Mapping
from enum import IntEnum
import json

from pops.graph_functions import GraphFunction, graph, _Codec, _formals, _walk_type
from pops.time._authoring import atomic_authoring
from pops.time._program.structured_control import same_signature, value_signature
from pops.time._program.value_validation import require_owned
from pops.time.variant_types import Failure, Int64, Product, Result, ValueType, type_of
from pops.time.values import ProgramValue


class DecodeFailure(IntEnum):
    INVALID_BUDGET = 15101
    INVALID_ACCURACY = 15102
    INVALID_DOMAIN = 15103
    INADMISSIBLE = 15104
    UNSELECTED = 15105
    OLD_ENCODING = 15106
    NONFINITE = 15107
    DOMAIN_INSTANCE = 15108


DecodeErrors = Failure(inventory_decode=Product(code=Int64))
_GRAPH_NAMES = ('encode', 'decode', 'inventory_domain', 'admissible', 'selection')
_INPUT_NAMES = ('recipient', 'application', 'base_value', 'before_data', 'after_data',
                'budget', 'atol', 'rtol')


def _gate(p, value, predicate, code):
    require_owned(p, predicate, 'inventory decode predicate', vtype='bool')
    result = Result(type_of(value), DecodeErrors)
    return p.require_success(p.branch(predicate,
        lambda g: g.variant(result, 'ok', value),
        lambda g: g.variant(result, 'err', g.variant(DecodeErrors, 'inventory_decode',
                                                  g.product(code=g.int64(int(code)))))))


def _and(p, a, b):
    return p.branch(a, lambda g: b, lambda g: g.boolean(False))


def _options(value, capacity):
    if value_signature(value).get('ownership', 'replicated') == 'replicated':
        return {}
    if type(capacity) is not int or capacity < 1:
        raise ValueError('distributed inventory conversion requires a positive byte_capacity')
    return {'byte_capacity': capacity}


def _inventory(value):
    from pops.finite_contributions import inventory
    return inventory(value, 'inventory reconstruction')


def _unary(p, value, operation, capacity):
    if value.vtype == 'scalar':
        return operation(p, value)
    if value.vtype == 'product':
        return p.product(**{k: _unary(p, value[k], operation, capacity)
                            for k, _ in value_signature(value)['members']})
    return p.map(p.domain(value), lambda g, i: _unary(g, g.at(value, i), operation, capacity),
                 **_options(value, capacity))


def _binary(p, left, right, operation, capacity):
    if not same_signature(value_signature(left), value_signature(right)):
        raise TypeError('inventory coordinates must have identical types and authorities')
    if left.vtype == 'scalar':
        return operation(p, left, right)
    if left.vtype == 'product':
        return p.product(**{k: _binary(p, left[k], right[k], operation, capacity)
                            for k, _ in value_signature(left)['members']})
    domain = p.domain(left)
    right = _gate(p, right, p.same_domain(domain, p.domain(right)), DecodeFailure.DOMAIN_INSTANCE)
    return p.map(domain, lambda g, i: _binary(g, g.at(left, i), g.at(right, i), operation, capacity),
                 **_options(left, capacity))


def _all(p, value, capacity):
    if value.vtype == 'bool':
        return value
    if value.vtype == 'product':
        out = p.boolean(True)
        for key, _ in value_signature(value)['members']:
            out = _and(p, out, _all(p, value[key], capacity))
        return out
    leaves = p.map(p.domain(value), lambda g, i: _all(g, g.at(value, i), capacity),
                   **_options(value, capacity))
    distributed = value_signature(value)['ownership'] != 'replicated'
    return p.reduce(leaves, p.boolean(True), lambda g, a, b: _and(g, a, b),
                    order='global_gid_ascending' if distributed else 'ascending',
                    **_options(value, capacity))


def _finite(p, value, capacity):
    return _all(p, _unary(p, value, lambda g, x: g.isfinite(x), capacity), capacity)


def _abs(p, value):
    # A lazy finite scalar branch; no absolute-value callback enters native code.
    return p.branch(value >= 0, lambda g: value, lambda g: -value)


def _checked_body(p, a, functions, capacity):
    # The state carried by each guard makes every later expression depend on it.
    packet = p.product(**a)
    packet = _gate(p, packet, packet['budget'] >= 0, DecodeFailure.INVALID_BUDGET)
    packet = _gate(p, packet, p.isfinite(packet['rtol']), DecodeFailure.INVALID_ACCURACY)
    packet = _gate(p, packet, packet['rtol'] >= 0, DecodeFailure.INVALID_ACCURACY)
    packet = _gate(p, packet, _finite(p, packet['atol'], capacity), DecodeFailure.INVALID_ACCURACY)
    packet = _gate(p, packet, _all(p, _unary(p, packet['atol'], lambda g, x: x >= 0, capacity), capacity),
                   DecodeFailure.INVALID_ACCURACY)
    for key in ('base_value', 'recipient', 'application'):
        packet = _gate(p, packet, _finite(p, packet[key], capacity), DecodeFailure.NONFINITE)
    for key, code in (('admissible', DecodeFailure.INADMISSIBLE),
                      ('selection', DecodeFailure.UNSELECTED)):
        packet = _gate(p, packet, functions[key](p, value=packet['base_value'],
                                                 data=packet['before_data']), code)
    old_encoded = functions['encode'](p, value=packet['base_value'], data=packet['before_data'])
    _inventory(old_encoded)
    exact_old = _all(p, _binary(p, old_encoded, packet['recipient'], lambda g, x, y: x == y, capacity), capacity)
    packet = _gate(p, packet, exact_old, DecodeFailure.OLD_ENCODING)
    packet = _gate(p, packet, functions['inventory_domain'](p, inventory=packet['application'],
                                                           data=packet['after_data']), DecodeFailure.INVALID_DOMAIN)
    proposed = functions['decode'](p, inventory=packet['application'], data=packet['after_data'],
                                    budget=packet['budget'])
    if proposed.vtype == 'variant':
        if set(dict(value_signature(proposed)['members'])) != {'ok', 'err'}:
            raise TypeError('inventory decoder variant output must be a typed Result')
        proposed = p.require_success(proposed)
    _inventory(proposed)
    if not same_signature(value_signature(proposed), value_signature(a['base_value'])):
        raise TypeError('inventory decoder changed retained representation or support')
    proposed = _gate(p, proposed, _finite(p, proposed, capacity), DecodeFailure.NONFINITE)
    for key, code in (('admissible', DecodeFailure.INADMISSIBLE),
                      ('selection', DecodeFailure.UNSELECTED)):
        proposed = _gate(p, proposed, functions[key](p, value=proposed, data=packet['after_data']), code)
    encoded = functions['encode'](p, value=proposed, data=packet['after_data'])
    _inventory(encoded)
    defect = _binary(p, encoded, packet['application'], lambda g, x, y: x-y, capacity)
    bound = _binary(p, packet['atol'], packet['application'],
                    lambda g, absolute, target: absolute + packet['rtol'] * _abs(g, target), capacity)
    checked = p.product(value=proposed, defect=defect, encoded=encoded, bound=bound,
                        atol=packet['atol'], rtol=packet['rtol'], budget=packet['budget'])
    checked = _gate(p, checked, _finite(p, defect, capacity), DecodeFailure.NONFINITE)
    checked = _gate(p, checked, _finite(p, bound, capacity), DecodeFailure.INVALID_ACCURACY)
    # The equivalent closed interval avoids introducing the artificial kink of
    # abs(defect) at an exactly solved residual. Both finite guards above and the
    # original defect/bound in the success/error payload remain unchanged.
    accepted = _all(p, _binary(p, checked['defect'], checked['bound'],
                              lambda g, error, limit: _and(g, -limit <= error, error <= limit),
                              capacity), capacity)
    residual_errors = Failure(inventory_residual=Product(defect=type_of(defect), bound=type_of(bound)))
    result = Result(type_of(checked), residual_errors)
    return p.require_success(p.branch(accepted,
        lambda g: g.variant(result, 'ok', checked),
        lambda g: g.variant(result, 'err', g.variant(residual_errors, 'inventory_residual',
                            g.product(defect=checked['defect'], bound=checked['bound'])))))


def _proof(functions, bindings, capacity):
    inputs = Product(**{key: type_of(bindings[key]) for key in _INPUT_NAMES})
    output = Product(value=type_of(bindings['base_value']), defect=type_of(bindings['application']),
                     encoded=type_of(bindings['application']), bound=type_of(bindings['application']),
                     atol=type_of(bindings['atol']), rtol=type_of(bindings['rtol']),
                     budget=type_of(bindings['budget']))
    return graph(inputs=inputs, output=output)(lambda p, args:
        _checked_body(p, {key: args[key] for key in _INPUT_NAMES}, functions, capacity))


def _descriptor(function):
    if type(function) is not GraphFunction:
        raise TypeError('inventory conversion requires closed GraphFunctions, not callbacks')
    snapshot = function.to_data()
    snapshot.pop('inputs'); snapshot.pop('output')
    # Dependent type witnesses are remapped by the enclosing graph codec. A second
    # frozen JSON copy of their headers would retain stale points/domain families.
    return {'graph': snapshot, 'inputs': function.inputs.signature,
            'output': function.output.signature}


def _load(data):
    if not isinstance(data, Mapping) or set(data) != {'graph', 'inputs', 'output'}:
        raise ValueError('malformed inventory conversion GraphFunction')
    def thaw(value):
        if isinstance(value, Mapping): return {k: thaw(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)): return [thaw(v) for v in value]
        return value
    snapshot = thaw(data['graph'])
    if 'inputs' in snapshot or 'output' in snapshot:
        raise ValueError('inventory graph headers must come from their exact live type witnesses')
    inputs, output = ValueType(data['inputs']), ValueType(data['output'])
    snapshot.update(inputs=inputs.to_data(), output=output.to_data())
    return GraphFunction.from_data(snapshot, inputs=inputs, output=output)


def _projection(value):
    # Only exact Product aliases are transparent. Never remove a guard/branch.
    while value.op == 'product_get' and value.inputs[0].op == 'product':
        owner = value.inputs[0]
        value = owner.inputs[tuple(owner.attrs['fields']).index(value.attrs['field'])]
    return value


def _structure(value, bindings):
    """Complete alpha-normalized operation data; not a hash or a numerical proof.

    Exact argument identities are terminals. Regions and SSA references are renamed,
    while operations, constants, errors, support signatures and every check remain.
    """
    p = value.prog
    # Named argument order is not semantic; repeated bindings have one stable name.
    anchors = {}
    for key in sorted(bindings): anchors.setdefault(_projection(bindings[key]).id, key)
    memo, nodes, regions, families = {}, [], {}, {}
    local_families, visited = set(), set()
    def discover(x):
        if isinstance(x, ProgramValue):
            x = _projection(p._canonical_value(x))
            if x.id in anchors or x.id in visited: return
            visited.add(x.id)
            if (x.op == 'finite_range_set' and len(x.inputs) == 1) or (
                    x.op == 'distributed_domain' and not x.attrs.get('family_witness')):
                local_families.add(x.attrs['signature']['family'])
            discover(x.inputs); discover(x.attrs)
        elif isinstance(x, Mapping):
            for child in x.values(): discover(child)
        elif isinstance(x, (tuple, list)):
            for child in x: discover(child)
    discover(value)
    type_anchors = tuple(_walk_type({key: value_signature(bindings[key]) for key in sorted(bindings)}))
    codec = _Codec(program=p, anchors=type_anchors)
    def encode(x):
        if isinstance(x, ProgramValue):
            x = _projection(p._canonical_value(x))
            if x.id in anchors: return {'argument': anchors[x.id]}
            if x.id in memo: return {'ref': memo[x.id]}
            idx = len(nodes); memo[x.id] = idx; nodes.append(None)
            attrs = {}
            for key, child in sorted(x.attrs.items()):
                if key.endswith('_region'):
                    attrs[key] = regions.setdefault(child, len(regions))
                else:
                    attrs[key] = encode(child)
            nodes[idx] = {'op': x.op, 'vtype': x.vtype, 'inputs': encode(x.inputs),
                          'attrs': attrs, 'point': encode(x.point)}
            return {'ref': idx}
        if isinstance(x, Mapping):
            pairs = sorted(x.items(), key=lambda item: json.dumps(codec.encode(item[0]), sort_keys=True))
            return {'mapping': [[encode(k), ({'local_family': families.setdefault(v, len(families))}
                            if k == 'family' and v in local_families else encode(v))]
                    for k, v in pairs]}
        if isinstance(x, (tuple, list)): return [encode(v) for v in x]
        from pops.time.points import TimePoint, StagePoint
        if isinstance(x, (TimePoint, StagePoint)):
            # Re-elaboration may recreate an equal point object. Preserve its full
            # structural clock/stage data instead of depending on Python identity.
            return {'point': encode(x.to_data())}
        if x is None or type(x) in (str, int, bool):
            return {'atom_type': type(x).__name__, 'value': x}
        return codec.encode(x)
    return {'root': encode(value), 'nodes': nodes}


@atomic_authoring
def checked_inventory_decode(program, *, recipient, application, base_value, encode, decode,
                             inventory_domain, admissible, selection, before_data, after_data,
                             budget, atol, rtol, byte_capacity=None):
    """Return Result(value, defect, encoded, bound, accuracy, budget).

    encode(value,data) is shared by the before/after equations. decode(inventory,
    data,budget) proposes a value or typed Result; it cannot supply the residual.
    inventory_domain(inventory,data), admissible(value,data), selection(value,data)
    are mandatory Bool graphs. In particular selection distinguishes multiple roots.
    atol has one real coordinate per inventory coordinate; rtol is dimensionless.
    Every coordinate uses |R| <= atol + rtol*|application|, including exact live
    Array domain checks. Empty arrays satisfy the universal coordinate condition.
    """
    bindings = dict(recipient=recipient, application=application, base_value=base_value,
                    before_data=before_data, after_data=after_data, budget=budget, atol=atol, rtol=rtol)
    for key, value in bindings.items(): require_owned(program, value, 'inventory conversion '+key)
    for key in ('recipient', 'application', 'base_value', 'atol'): _inventory(bindings[key])
    require_owned(program, budget, 'inventory conversion budget', vtype='int64')
    require_owned(program, rtol, 'inventory conversion rtol', vtype='scalar')
    if not same_signature(value_signature(recipient), value_signature(application)) or not same_signature(
            value_signature(application), value_signature(atol)):
        raise TypeError('inventory target, recipient and atol must have the exact same coordinates')
    functions = dict(encode=encode, decode=decode, inventory_domain=inventory_domain,
                     admissible=admissible, selection=selection)
    descriptors = {key: _descriptor(functions[key]) for key in _GRAPH_NAMES}
    proof = _proof(functions, bindings, byte_capacity)
    def body(p):
        checked = proof(p, **bindings)
        retained = checked['value']
        defect = checked['defect']
        from pops.ledger_provenance import bind_contract
        metadata = dict(schema_version=1, recipient=recipient, application=application,
                        base_value=base_value, retained_input=retained, defect=defect,
                        checked=checked, input_values=bindings, functions=descriptors,
                        byte_capacity=byte_capacity, defect_policy='declare_once_identity_no_reapply')
        marked = bind_contract(p, retained, 'inventory_conversion', metadata)
        return p.product(value=marked, defect=defect, encoded=checked['encoded'], bound=checked['bound'],
                         atol=checked['atol'], rtol=checked['rtol'], budget=checked['budget'])
    return program.capture(body)


def validate_inventory_conversion(value):
    """Revalidate a provenance_bind after graph loading/inlining; never trust attrs alone."""
    require_owned(value.prog, value, 'inventory conversion marker')
    if value.op != 'provenance_bind' or value.attrs.get('contract_kind') != 'inventory_conversion':
        raise ValueError('inventory conversion requires its exact provenance_bind operation')
    return validate_inventory_conversion_operation(value.inputs, value.attrs.get('contract'), value_signature(value))


def validate_inventory_conversion_operation(inputs, data, signature):
    """The same mandatory check before a loaded alias operation can be issued."""
    if len(inputs) != 1 or not same_signature(signature, value_signature(inputs[0])):
        raise ValueError('inventory conversion must preserve its exact retained type')
    value = inputs[0]
    keys = {'schema_version','recipient','application','base_value','retained_input','defect',
            'checked','input_values','functions','byte_capacity','defect_policy'}
    if not isinstance(data, Mapping) or set(data) != keys or data['schema_version'] != 1 \
            or data['defect_policy'] != 'declare_once_identity_no_reapply':
        raise ValueError('malformed inventory conversion contract')
    bindings = data['input_values']
    if not isinstance(bindings, Mapping) or set(bindings) != set(_INPUT_NAMES):
        raise ValueError('malformed inventory conversion arguments')
    for key, arg in bindings.items(): require_owned(value.prog, arg, 'inventory conversion '+key)
    for key in ('recipient','application','base_value'):
        if data[key] is not bindings[key]: raise ValueError('inventory conversion argument identity changed')
    if data['retained_input'] is not value:
        raise ValueError('inventory conversion was copied onto another retained value')
    checked = data['checked']
    require_owned(value.prog, checked, 'inventory conversion checked result', vtype='product')
    for name, actual in (('value',data['retained_input']),('defect',data['defect'])):
        if actual.op != 'product_get' or (len(actual.inputs) != 1 or actual.inputs[0] is not checked) or actual.attrs.get('field') != name:
            raise ValueError('inventory conversion is not an exact checked-result projection')
    functions = data['functions']
    if not isinstance(functions, Mapping) or set(functions) != set(_GRAPH_NAMES):
        raise ValueError('malformed inventory conversion equations')
    functions = {key: _load(functions[key]) for key in _GRAPH_NAMES}
    proof = _proof(functions, bindings, data['byte_capacity'])
    p, formal = _formals(proof.inputs)
    expected_args = {key: formal[key] for key in _INPUT_NAMES}
    # Repeated actual arguments must remain repeated arguments during comparison.
    seen = {}
    for key in sorted(bindings):
        actual = bindings[key]
        identity = _projection(actual).id
        if identity in seen: expected_args[key] = expected_args[seen[identity]]
        else: seen[identity] = key
    expected = proof(p, **expected_args)
    if _structure(checked, bindings) != _structure(expected, expected_args):
        raise ValueError('inventory conversion original-equation/control graph was modified')
    return data


__all__ = ['checked_inventory_decode', 'validate_inventory_conversion', 'DecodeFailure', 'DecodeErrors']
