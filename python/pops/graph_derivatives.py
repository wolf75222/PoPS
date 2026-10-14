"""Derivatives of closed finite Program graphs, compiled by the ordinary PoPS backend.

Forward transformation follows SSA, lazy branches and finite native arrays. Explicitly
requested bounded-iteration AD differentiates the executed algorithm on an open branch
domain; it never infers an implicit relation from that algorithm. Discrete values have the zero-dimensional tangent Product(); a numerical direction
for an Int64/Bool/Index is a type error. Reverse action below is an explicit finite transposition
of forward columns with authored pairings, not a reverse-mode performance claim.
"""
from __future__ import annotations

from dataclasses import dataclass

from pops.graph_functions import GraphFunction, _Codec, graph
from pops.time._program.structured_control import new_typed, value_signature
from pops.time.variant_types import Bool, Failure, Float64, Int64, Product, Result, ValueType, Variant, as_type, type_of


_DISCRETE = frozenset({'bool', 'int64', 'index_set', 'index'})
_NONDIF_CODE = 91001
DerivativeErrors = Failure(differential=Product(code=Int64))


def tangent_type(value_type):
    """Finite coordinate tangent type; scientific field tangent rules are not fabricated."""
    signature = as_type(value_type).signature
    kind = signature['vtype']
    if kind == 'scalar':
        return ValueType(signature)
    if kind in _DISCRETE:
        return Product()
    if kind == 'product':
        return Product(**{key: tangent_type(ValueType(child)) for key, child in signature['members']})
    if kind == 'array':
        return ValueType(dict(signature, element=tangent_type(ValueType(signature['element'])).signature))
    if kind == 'variant':
        return Variant(**{key: tangent_type(ValueType(child)) for key, child in signature['members']})
    raise NotImplementedError('no graph tangent rule for value type '+kind)


def _raw(attrs, name):
    return dict(attrs['$map'])[name]


def _ref(attrs, name):
    return _raw(attrs, name)['$ref']


def _zero(p, signature):
    kind = signature['vtype']
    if kind == 'scalar': return p.float64(0.)
    if kind == 'int64': return p.int64(0)
    if kind == 'bool': return p.boolean(False)
    if kind == 'product':
        return p.product(**{key: _zero(p, child) for key, child in signature['members']})
    raise NotImplementedError('no zero constructor for dynamic dependent type '+kind)


@dataclass(frozen=True)
class _IterationTangent:
    paired: object


class _ComparisonProof:
    """Syntactic equality/invariance on the original, closed graph.

    This is not a numerical evaluator or an equality assumption. It only follows
    exact Product projections, successful Result transports and pure scalar
    arithmetic. Primal guards remain executed. Unknown operations remain opaque;
    a zero *direction* never proves local constancy of a predicate.
    """
    def __init__(self, snapshot):
        self.records, self.memo, self.keys = {}, {}, {}
        def add(nodes):
            for node in nodes:
                self.records[node['id']] = node
                for block in node['blocks'].values():
                    add(block)
        add(snapshot['nodes'])

    @staticmethod
    def data(value):
        if isinstance(value, dict):
            if set(value) == {'$map'}:
                return {k: _ComparisonProof.data(v) for k, v in value['$map']}
            if set(value) == {'$tuple'}:
                return tuple(_ComparisonProof.data(v) for v in value['$tuple'])
        return value

    @staticmethod
    def frozen(value):
        if isinstance(value, dict):
            return tuple((k, _ComparisonProof.frozen(v)) for k, v in sorted(value.items()))
        if isinstance(value, (list, tuple)):
            return tuple(_ComparisonProof.frozen(v) for v in value)
        return (type(value).__name__, value)

    def key(self, descriptor, varying):
        # Interned DAG keys keep repeated projections linear in graph size.
        return self.keys.setdefault(descriptor, len(self.keys)), varying

    def opaque(self, index, path):
        return self.key(('opaque', index, path), True)

    def join(self, entries, index, path):
        live = [x for x in entries if x is not None]
        if live and all(x == live[0] for x in live[1:]):
            return live[0]
        return self.opaque(index, path)

    def expression(self, index, path=()):
        key = index, path
        if key in self.memo:
            return self.memo[key]
        # Cyclic/dependent constructs are opaque; no finite loop is unrolled.
        self.memo[key] = self.opaque(index, path)
        record = self.records.get(index)
        if record is None:
            return self.memo[key]
        op, xs = record['op'], record['inputs']
        a = self.data(record['attrs'])
        answer = self.memo[key]
        if op == 'product_get':
            answer = self.expression(xs[0], (('field', a['field']), *path))
        elif op == 'provenance_bind':
            answer = self.expression(xs[0], path)
        elif op == 'product' and path and path[0][0] == 'field':
            fields = a['fields']
            if path[0][1] in fields:
                answer = self.expression(xs[fields.index(path[0][1])], path[1:])
        elif op == 'variant_require_success':
            answer = self.expression(xs[0], (('tag', 'ok'), *path))
        elif op == 'variant_payload':
            answer = self.expression(xs[0], (('tag', a['tag']), *path))
        elif op == 'variant_construct' and path and path[0][0] == 'tag':
            answer = self.expression(xs[0], path[1:]) if a['tag'] == path[0][1] else None
        elif op == 'variant_unreachable':
            answer = None
        elif op == 'branch':
            answer = self.join([self.expression(a[k]['$ref'], path)
                                for k in ('true_result', 'false_result')], index, path)
        elif op == 'capture_result' and record['vtype'] == 'variant' and path:
            if a['body_is_variant_result']:
                answer = self.expression(a['body']['$ref'], path)
            elif path[0] == ('tag', 'ok'):
                answer = self.expression(a['body']['$ref'], path[1:])
        elif not path and op == 'finite_real':
            if not xs:
                answer = self.key(('literal', self.frozen(a['value'])), False)
            elif self.records.get(xs[0], {}).get('vtype') == 'int64':
                answer = self.key(('discrete_cast', xs[0]), False)
            else:
                answer = self.expression(xs[0])
        elif not path and op == 'scalar_op':
            arguments = [self.expression(xs[item]) if kind == 'v' else
                         self.key(('literal', self.frozen(item)), False)
                         for kind, item in a['operands']]
            if all(x is not None for x in arguments):
                answer = self.key(('scalar_op', a['fn'], tuple(x[0] for x in arguments)),
                                  any(x[1] for x in arguments))
        elif not path and op == 'scalar_expression':
            arguments = [self.expression(x) for x in xs]
            if all(x is not None for x in arguments):
                answer = self.key(('scalar_expression', self.frozen(a), tuple(x[0] for x in arguments)),
                                  any(x[1] for x in arguments))
        self.memo[key] = answer
        return answer

    def invariant(self, record):
        xs = record['inputs']
        left = self.expression(xs[0])
        right = (self.expression(xs[1]) if len(xs) == 2 else
                 self.key(('literal', self.frozen(self.data(record['attrs'])['rhs'])), False))
        return (left is not None and right is not None and
                (left[0] == right[0] or not (left[1] or right[1])))


class _Forward:
    def __init__(self, function, program, primal, tangent, *, differentiate_iterations=False):
        self.function, self.p = function, program
        self.differentiate_iterations = differentiate_iterations
        self.codec = _Codec(program=program,
            bindings={'inputs': function.inputs.signature, 'output': function.output.signature},
            point=primal.point)
        self.codec.values[function.to_data()['formal']] = primal
        self.tangents = {function.to_data()['formal']: tangent}
        self.comparison_proof = _ComparisonProof(function.to_data())

    def region(self, records, seeds, output, *, predicate=False):
        values, tangents = dict(self.codec.values), dict(self.tangents)
        try:
            for record in records:
                self.codec.values.pop(record['id'], None)
                self.tangents.pop(record['id'], None)
            for index, pair in seeds.items():
                self.codec.values[index], self.tangents[index] = pair
            for record in records:
                if record['id'] not in seeds:
                    self.node(record)
            value, tangent = self.codec.values[output], self.tangents[output]
            if predicate:
                value = self.condition(value)
            if value.vtype == 'result':
                # A step's recoverable failure must stay a failure. Only the successful arm
                # provides paired data; the native require preserves code and origin.
                value, tangent = self.p.require_success(value), self.p.require_success(tangent)
            return self.p.product(value=value, tangent=tangent)
        finally:
            self.codec.values, self.tangents = values, tangents

    def condition(self, condition):
        # Non-invariant continuous comparisons are guarded at their SSA producer,
        # including those hidden behind Product/Variant projections. An external Bool is
        # a fixed discrete input. Only a syntactic proof on the original graph can
        # discharge a guard; a predicate name or a zero tangent cannot do so.
        if condition.vtype != 'bool':
            raise TypeError('AD control condition must be a native Bool')
        return condition

    def checked_comparison(self, condition):
        left = condition.inputs[0]
        right = (condition.inputs[1] if len(condition.inputs) == 2
                 else condition.attrs['rhs'].to_python())
        p = self.p
        typ = Result(type_of(condition), DerivativeErrors)
        return p.require_success(p.branch(left != right,
            lambda q: q.variant(typ, 'ok', condition),
            lambda q: q.variant(typ, 'err', q.variant(DerivativeErrors, 'differential',
                q.product(code=q.int64(_NONDIF_CODE))))))

    def validate_iteration_differential(self, paired):
        # An AD-domain error must never disappear into an unrelated primal budget fallback.
        # The distinct closed error tag cannot collide with arbitrary user Int64 codes.
        if paired.vtype != 'variant':
            return
        from pops.time._program.structured_control import failure_signature
        errors = failure_signature(paired)
        if errors['vtype'] != 'variant' or not (set(dict(errors['members']))-{'runtime'}):
            return
        # This function is used only for an originally legacy iterate, which had
        # no typed user-error union. Every new non-runtime error therefore comes
        # from the differential (possibly inside a rank/gid transport envelope).
        # Preserve it whole rather than letting the primal budget fallback erase it.
        error_type = ValueType(errors)
        def error(q, payload):
            def reject(tag):
                return lambda r,cause:r.require_success(r.variant(Result(Bool,error_type),'err',
                    r.variant(error_type,tag,cause)))
            return q.match(payload, **{tag: reject(tag) if tag != 'runtime' else
                lambda r, ignored: r.boolean(True) for tag, _ in errors['members']})
        self.p.match(paired, ok=lambda q, ignored: q.boolean(True), err=error)

    def iteration_payload(self, differential):
        result = self.p.require_success(differential.paired)
        return self.p.product(state=result['state']['tangent'], iterations=self.p.product())

    def node(self, record):
        p, op = self.p, record['op']
        refs = record['inputs']
        xs = [self.codec.values[index] for index in refs]
        ds = [self.tangents[index] for index in refs]
        attrs = record['attrs']
        if op == 'provenance_bind':
            from pops.ledger_provenance import rebuild_contract_primal
            contract_attrs = self.codec.decode(attrs)
            replacements = []
            value = rebuild_contract_primal(p, xs[0], contract_attrs, replacements=replacements)
            # A reissued conversion owns one complete checked primal Product.
            # Rebind every previously named projection of that Product, including
            # its measured defect, before later packet/metadata decoding. The
            # tangent remains the differentiated original computation and carries
            # no inventory certificate.
            roots = {id(old): (old,new) for old,new in replacements}
            rebound = {}
            def remap_primal(item):
                if id(item) in rebound:
                    return rebound[id(item)]
                pair = roots.get(id(item))
                if pair is not None and pair[0] is item:
                    answer = pair[1]
                elif item.op == 'product_get':
                    parent = remap_primal(item.inputs[0])
                    answer = item if parent is item.inputs[0] else parent[item.attrs['field']]
                else:
                    answer = item
                rebound[id(item)] = answer
                return answer
            self.codec.values = {key: remap_primal(item) for key,item in self.codec.values.items()}
            tangent = ds[0]
        elif op in ('distributed_map','distributed_requests','distributed_import',
                  'distributed_local_at','distributed_reduce','distributed_scatter_reduce','distributed_scan'):
            from pops._distributed_differential import transform
            value, tangent = transform(self,record,xs,ds)
        elif op in ('branch', 'result_match'):
            def arm(key):
                return lambda q: self.region(record['blocks'][key+'_block'], {}, _ref(attrs, key+'_result'))
            if op == 'result_match':
                # Retain the guarded Result construct, not only its Boolean choice:
                # legacy payload projections are capabilities scoped to its two arms.
                pair = p.match(xs[0], ok=lambda q, ignored: arm('true')(q),
                               err=lambda q, ignored: arm('false')(q))
            else:
                pair = p.branch(self.condition(xs[0]), arm('true'), arm('false'))
            value, tangent = pair['value'], pair['tangent']
        elif op == 'capture_result':
            if record['vtype'] == 'result':
                value = self.codec.inline(record)
                tangent = p.capture(lambda q: self.region(record['blocks']['body_block'], {},
                    _ref(attrs, 'body'))['tangent'])
            else:
                from pops._differential_results import paired_result, split_result
                original = ValueType(self.codec.decode(record['signature']))
                def body(q):
                    pair = self.region(record['blocks']['body_block'], {}, _ref(attrs, 'body'))
                    return paired_result(q, pair['value'], pair['tangent']) if _raw(attrs, 'body_is_variant_result') else pair
                pair = split_result(p, p.capture(body), original)
                value, tangent = pair['value'], pair['tangent']
        elif op == 'structured_iterate':
            if not self.differentiate_iterations:
                raise NotImplementedError('structured_iterate differentiation requires differentiate_iterations=True')
            legacy = record['vtype'] == 'iteration_result'
            if legacy:
                # Preserve the primal's exact legacy error envelope. The paired native trace
                # simultaneously updates state/tangent; explicit AD errors cannot be hidden.
                value = self.codec.inline(record)
            initial = p.product(value=xs[0], tangent=ds[0])
            def stop(q, carried):
                pair = self.region(record['blocks']['cond_block'], {
                    _ref(attrs, 'cond_argument'): (carried['value'], carried['tangent'])},
                    _ref(attrs, 'cond'), predicate=True)
                return pair['value']
            def step(q, carried):
                pair = self.region(record['blocks']['body_block'], {
                    _ref(attrs, 'body_argument'): (carried['value'], carried['tangent'])},
                    _ref(attrs, 'body'))
                if _raw(attrs, 'body_is_variant_result'):
                    from pops._differential_results import paired_result
                    return paired_result(q, pair['value'], pair['tangent'])
                return pair
            paired = p.iterate(initial, step, stop, max_steps=xs[1])
            if legacy:
                tangent = _IterationTangent(paired)
                self.validate_iteration_differential(paired)
            else:
                from pops._differential_results import pair_type, split_result
                from pops.time._program.structured_control import failure_signature
                original = ValueType(self.codec.decode(record['signature']))
                success = ValueType(dict(original.signature['members'])['ok'])
                wrapped = Result(pair_type(success), ValueType(failure_signature(paired)))
                def ok(q, item):
                    primal = q.product(state=item['state']['value'], iterations=item['iterations'])
                    direction = q.product(state=item['state']['tangent'], iterations=q.product())
                    return q.variant(wrapped, 'ok', q.product(value=primal, tangent=direction))
                paired = p.match(paired, ok=ok, err=lambda q, error:q.variant(wrapped, 'err', error))
                pair = split_result(p, paired, original)
                value, tangent = pair['value'], pair['tangent']
        elif op == 'variant_require_success':
            from pops._differential_results import paired_result
            pair = p.require_success(paired_result(p, xs[0], ds[0]))
            value, tangent = pair['value'], pair['tangent']
        elif op in ('iteration_success', 'result_success'):
            value = self.codec.inline(record)
            tangent = (self.iteration_payload(ds[0]) if isinstance(ds[0], _IterationTangent)
                       else p.require_success(ds[0]))
        elif op in ('iteration_payload', 'result_payload'):
            from pops._finite_differential import zero_like
            value = self.codec.inline(record)
            if value.attrs['variant'] == 'ok':
                tangent = (self.iteration_payload(ds[0]) if isinstance(ds[0], _IterationTangent)
                           else p.require_success(ds[0]))
            else:
                tangent = zero_like(p, value)
        elif op in ('result_ok', 'result_error'):
            value = self.codec.inline(record)
            tangent = p.success(ds[0]) if op == 'result_ok' else p.failure(ds[0], xs[1])
        elif op == 'structured_fold':
            initial = p.product(value=xs[0], tangent=ds[0])
            def step(q, carried, index):
                return self.region(record['blocks']['body_block'], {
                    _ref(attrs, 'body_argument'): (carried['value'], carried['tangent']),
                    _ref(attrs, 'body_index'): (index, q.product()),
                }, _ref(attrs, 'body'))
            pair = p.fold(initial, xs[1], step, max_steps=xs[2])
            value, tangent = pair['value'], pair['tangent']
        elif op == 'finite_map':
            domain = xs[0]
            paired = p.map(domain, lambda q, index: self.region(record['blocks']['body_block'],
                {_ref(attrs, 'body_index'): (index, q.product())}, _ref(attrs, 'body')))
            value = p.map(domain, lambda q, index: q.at(paired, index)['value'])
            tangent = p.map(domain, lambda q, index: q.at(paired, index)['tangent'])
        elif op == 'finite_scan':
            initial, domain = xs[0], xs[1]
            def step(q, carried, index):
                pair = self.region(record['blocks']['body_block'], {
                    _ref(attrs, 'body_argument'): (carried['value'], carried['tangent']),
                    _ref(attrs, 'body_index'): (index, q.product())}, _ref(attrs, 'body'))
                return q.product(state=q.product(value=pair['value']['state'], tangent=pair['tangent']['state']),
                                 value=q.product(value=pair['value']['value'], tangent=pair['tangent']['value']))
            paired = p.scan(domain, p.product(value=initial, tangent=ds[0]), step, order='ascending')
            value = p.product(state=paired['state']['value'], values=p.map(domain,
                lambda q, index: q.at(paired['values'], index)['value']))
            tangent = p.product(state=paired['state']['tangent'], values=p.map(domain,
                lambda q, index: q.at(paired['values'], index)['tangent']))
        else:
            if record['blocks'] or op == 'int64_cast':
                raise NotImplementedError('no graph derivative rule for '+op)
            value = self.codec.inline(record)
            if op == 'product':
                tangent = p.product(**dict(zip(value.attrs['fields'], ds)))
            elif op == 'product_get':
                tangent = ds[0][value.attrs['field']]
            elif op == 'scalar_op':
                operands = [(xs[x], ds[x]) if kind == 'v' else (x.to_python(), 0.)
                            for kind, x in value.attrs['operands']]
                (a, da), (b, db) = operands
                operation = value.attrs['fn']
                tangent = {'add': lambda: da+db, 'sub': lambda: da-db,
                           'mul': lambda: da*b+a*db,
                           'div': lambda: (da*b-a*db)/(b*b)}[operation]()
                if not hasattr(tangent, 'vtype'): tangent = p.float64(tangent)
            elif op == 'scalar_expression':
                tangent = _expression_tangent(p, value, ds)
            elif op == 'finite_real':
                tangent = ds[0] if xs and xs[0].vtype == 'scalar' else p.float64(0.)
            elif op == 'finite_at':
                tangent = p.at(ds[0], xs[1])
            elif op == 'variant_construct':
                tangent = p.variant(tangent_type(ValueType(value_signature(value))),
                                    value.attrs['tag'], ds[0])
            elif op == 'variant_payload':
                from pops._finite_differential import _tagged, zero_like
                tag = value.attrs['tag']
                tangent = _tagged(p, ds[0], tag, lambda q, item: item,
                                  template=lambda q: zero_like(q, value))
            elif op == 'variant_unreachable':
                tangent = new_typed(p, tangent_type(ValueType(value_signature(value))).signature,
                                    'variant_unreachable', tuple(ds), {})
            elif value.vtype in _DISCRETE:
                if op == 'compare' and not self.comparison_proof.invariant(record):
                    value = self.checked_comparison(value)
                tangent = p.product()
            else:
                raise NotImplementedError('no graph derivative rule for '+op)
        self.codec.values[record['id']], self.tangents[record['id']] = value, tangent
        return value, tangent


def _expression_tangent(p, value, directions):
    from pops._ir import expr as ir
    from pops.derivatives import jvp as expression_jvp
    from pops.time.expressions import ProgramScalar
    variables = tuple(ProgramScalar(x) for x in value.inputs)
    nodes = []
    binary = {'add': ir.Add, 'sub': ir.Sub, 'mul': ir.Mul, 'div': ir.Div, 'pow': ir.Pow}
    unary = {'neg': ir.Neg, 'sqrt': ir.Sqrt, 'exp': ir.Exp, 'sin': ir.Sin,
             'cos': ir.Cos, 'log': ir.Log, 'expm1': ir.Expm1}
    for encoded in value.attrs['expression_nodes']:
        op = encoded[0]
        if op == 'input': node = variables[encoded[1]]
        elif op == 'literal': node = ir.Const(encoded[1])
        elif op in binary: node = binary[op](nodes[encoded[1]], nodes[encoded[2]])
        elif op in unary: node = unary[op](nodes[encoded[1]])
        else:
            raise NotImplementedError('graph derivative needs an explicit rule for expression '+op)
        nodes.append(node)
    if not variables: return p.float64(0.)
    derivative = expression_jvp(nodes[value.attrs['expressions'][0]], wrt=variables,
                                direction=tuple(ProgramScalar(x) for x in directions))
    return p.scalar_expression('graph_tangent', derivative)


def jvp(function, *, differentiate_iterations=False, transport_byte_capacity=None):
    """Transform a closed graph to ``(primal,direction) -> (value,tangent)``.

    Switching surfaces of direct scalar-comparison branches reject with code 91001,
    except comparisons proved invariant from exact expressions or fixed discrete data. The active
    branch is differentiated on its open region. Discrete loop counts are held fixed. Unsupported
    operations reject during construction; no finite-difference fallback exists.
    """
    if type(function) is not GraphFunction:
        raise TypeError('graph jvp requires an exact GraphFunction')
    if type(differentiate_iterations) is not bool:
        raise TypeError('differentiate_iterations must be an explicit Bool')
    inputs = Product(primal=function.inputs, tangent=tangent_type(function.inputs))
    output = Product(value=function.output, tangent=tangent_type(function.output))
    from pops._finite_differential import with_transport
    @graph(inputs=inputs, output=output)
    @with_transport(transport_byte_capacity)
    def transformed(p, arguments):
        from pops._finite_differential import check_tangent
        direction = check_tangent(p, arguments['primal'], arguments['tangent'])
        compiler = _Forward(function, p, arguments['primal'], direction,
                            differentiate_iterations=differentiate_iterations)
        snapshot = function.to_data()
        for record in snapshot['nodes']:
            compiler.node(record)
        result = snapshot['result']
        return p.product(value=compiler.codec.values[result], tangent=compiler.tangents[result])
    return transformed


@dataclass(frozen=True)
class GraphPairings:
    """Closed typed Riesz maps; explicit even for the Euclidean coordinate convention."""
    output_to_covector: GraphFunction
    input_from_covector: GraphFunction

    def __post_init__(self):
        if type(self.output_to_covector) is not GraphFunction or type(self.input_from_covector) is not GraphFunction:
            raise TypeError('GraphPairings requires two closed GraphFunction maps')


def _leaves(signature, path=()):
    if signature['vtype'] == 'scalar': return [path]
    if signature['vtype'] == 'product':
        return [leaf for key, child in signature['members'] for leaf in _leaves(child, (*path, key))]
    raise NotImplementedError('static coordinate enumeration received a nonstatic type; use native finite coordinate traversal')


def _get(value, path):
    for key in path: value = value[key]
    return value


def _assemble(p, signature, leaves, path=()):
    if signature['vtype'] == 'scalar': return leaves[path]
    return p.product(**{key: _assemble(p, child, leaves, (*path, key))
                        for key, child in signature['members']})


def vjp(function, *, pairings, differentiate_iterations=False, transport_byte_capacity=None):
    """Exact finite transposed JVP columns under explicitly authored dualities.

    Runtime Array cardinalities are traversed by native Program loops. They are never
    unrolled by Python or replaced by a declared capacity. This is not a reverse-mode
    complexity claim; the cost is one directional application per real input coordinate.
    """
    if type(function) is not GraphFunction or type(pairings) is not GraphPairings:
        raise TypeError('graph vjp requires a GraphFunction and explicit GraphPairings')
    ti, to = tangent_type(function.inputs), tangent_type(function.output)
    for mapping, typ in ((pairings.output_to_covector, to), (pairings.input_from_covector, ti)):
        if mapping.inputs != Product(value=typ) or mapping.output != typ:
            raise TypeError('GraphPairings map must have inputs=Product(value=T), output=T')
    forward = jvp(function, differentiate_iterations=differentiate_iterations,
                  transport_byte_capacity=transport_byte_capacity)
    from pops._finite_differential import (basis_like, check_tangent, coordinate_dot,
        coordinate_map, has_dynamic_coordinates, with_transport)
    if has_dynamic_coordinates(function.inputs.signature) or has_dynamic_coordinates(function.output.signature):
        @graph(inputs=Product(primal=function.inputs, covector=to),
               output=Product(value=function.output, pullback=ti))
        @with_transport(transport_byte_capacity)
        def dynamic_transposed(p, args):
            primal = args['primal']
            value = function(p, **{key: primal[key] for key, _ in function.inputs.signature['members']})
            covector = check_tangent(p, value, args['covector'])
            dual = pairings.output_to_covector(p, value=covector)
            dual = check_tangent(p, value, dual)
            def column(q, path):
                result = forward(q, primal=primal, tangent=basis_like(q, primal, path))
                return coordinate_dot(q, result['tangent'], dual)
            covariant = coordinate_map(p, primal, column)
            pullback = pairings.input_from_covector(p, value=covariant)
            return p.product(value=value, pullback=check_tangent(p, primal, pullback))
        return dynamic_transposed
    input_paths, output_paths = _leaves(ti.signature), _leaves(to.signature)
    @graph(inputs=Product(primal=function.inputs, covector=to),
           output=Product(value=function.output, pullback=ti))
    def transposed(p, args):
        dual = pairings.output_to_covector(p, value=args['covector'])
        columns = {}
        value = None
        for path in input_paths:
            direction = _assemble(p, ti.signature,
                {other: p.float64(1. if path == other else 0.) for other in input_paths})
            result = forward(p, primal=args['primal'], tangent=direction)
            value = result['value']
            columns[path] = sum((_get(result['tangent'], other)*_get(dual, other)
                                 for other in output_paths), p.float64(0.))
        if value is None:
            value = function(p, **{key: args['primal'][key] for key, _ in function.inputs.signature['members']})
        pullback = pairings.input_from_covector(p, value=_assemble(p, ti.signature, columns))
        return p.product(value=value, pullback=pullback)
    return transposed


__all__ = ['tangent_type', 'jvp', 'vjp', 'GraphPairings']
