"""Typed delivery of owned Uses (spec 0.3 IR19, D5, B4, CF7).

A ``Use`` owned by a realization leaves a region element by element from a ``map``, through
the final state of a ``fold`` or an ``iterate``, through exactly one arm of a branch or a
match, through the target of a ``scatter_reduce``, or through the plain data operations that
carry it; any other path is refused with ``DeliveryNotStructural``. The analysis never enumerates iterations, map entries or
alternatives, and it has no budget:

* Alternatives form a decision diagram over the named conditions of the Program, each
  keyed by its SSA identifier and native invocation, as the success paths of the b6
  analysis were. Two branches on the same condition stay correlated, contradictory
  choices are never formed, and the diagram shares every subgraph, so independent
  branches cost their number, not the number of their combinations. A leaf holds the
  anonymous alternatives of a path (loop counts, iterations), each a set of delivered
  evaluations.
* A ``map`` delivers its body element by element: one visit of its body, whose evaluations
  carry the ordinal ``each``. A literal empty domain evaluates nothing, so it delivers
  nothing; any other domain, a carried one included, delivers element by element, and an
  empty extent of it covers its empty term.
* A loop body is visited once, with its argument bound to placeholders for the carried
  paths. The carried state is then a fixpoint over the finite set of those paths: the
  ages of evaluations relative to the current iteration are exact up to the number of
  carried paths and summarized beyond (``some`` for one older evaluation, ``many`` for two
  or more, which a set can no longer distinguish). That abstract state is eventually
  periodic, and the transient and period depend on the carried paths, never on the trip
  count, so a loop of 10**4 iterations costs the same as a loop of 3. A trip count within
  the transient keeps exact iteration ordinals; a larger one gives ordinals relative to
  the last iteration (``final``) or the literal count.
* A ``scatter_reduce`` delivers each source element once, to its target. Its combine is
  visited once, as every combination, with its accumulator and its element as placeholders;
  on every path it must combine the same parts, keep both and deliver nothing of its own.

``require_single_delivery`` checks one term (D5): every success path delivers exactly one
evaluation, otherwise ``CoverageNotExact`` names the term. It counts evaluations up to two,
so it reads only the distinct leaves of the diagram. ``delivered_frontiers`` lists the exact
rows (choices and delivered evaluations) for the provenance validators.
"""
from collections.abc import Mapping
from dataclasses import dataclass

from pops._ir.refusals import CoverageNotExact, DeliveryNotStructural
from pops.time.values import ProgramValue


@dataclass(frozen=True)
class DeliveryReference:
    """One delivered evaluation: its SSA value, native invocation and projection."""

    value_id: int
    invocation: tuple
    projection: tuple = ()


@dataclass(frozen=True)
class Ordinal:
    """A symbolic iteration of a region in an invocation step.

    ``final`` with ``offset`` k is the k-th iteration before the last one of a loop whose
    count is not a literal; ``each`` is every element of a map; ``some`` is one evaluation
    of an older iteration and ``many`` two or more of them. The kinds ``current`` and
    ``age`` exist only while a loop body is analysed.
    """

    kind: str
    offset: int = 0

    def __repr__(self):
        return '%s%+d' % (self.kind, -self.offset) if self.offset else self.kind


EACH = Ordinal('each')
SOME = Ordinal('some')
MANY = Ordinal('many')
_CURRENT = Ordinal('current')


def _age(offset):
    return Ordinal('age', offset)


def multiplicity(reference):
    """The number of evaluations a reference stands for, counted up to two."""
    return 2 if any(step[1] == MANY for step in reference.invocation) else 1


def identified(reference):
    """Whether a reference names one evaluation that another reference can be compared with."""
    return not any(step[1] in (SOME, MANY) for step in reference.invocation)


def delivered_count(references):
    """The number of evaluations in a delivered set, counted up to two."""
    return min(2, sum(multiplicity(reference) for reference in references))


def require_identified(references, message):
    """Refuse to compare evaluations that stand for summarized loop iterations (IR13)."""
    from pops._ir.refusals import PropertyNotDerived
    for reference in references:
        if not identified(reference):
            raise PropertyNotDerived(message + '; a summarized loop iteration has no evaluation identity',
                                     region='value#%d' % reference.value_id, property='evaluation identity',
                                     blocking=reference.invocation)
    return references


def term_label(use):
    """A readable name of a Use from its data: balance, term ordinal, kind and support."""
    try:
        occurrence = use['occurrence']
        balance = occurrence['balance']
        name = balance.get('name') or balance.get('local_id')
        payload = occurrence.get('payload') or {}
        return '%s[%s] %s %s (%s, %s, %s, %s)' % (
            name, occurrence['ordinal'], occurrence['kind'], payload.get('local_id', '?'),
            use['locus'], use['temporal_role'], use['representation'], use['partition'])
    except (KeyError, TypeError, AttributeError):
        return repr(use)


def _refs(value):
    if isinstance(value, ProgramValue):
        yield value
    elif isinstance(value, Mapping):
        for item in value.values():
            yield from _refs(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from _refs(item)


def _region_nodes(nodes):
    for value in nodes:
        yield value
        for name, attribute in value.attrs.items():
            if name.endswith('_block'):
                yield from _region_nodes(attribute)


def _rename(invocation, region, rename):
    return tuple(step if step[0] != region else (step[0], rename(step[1])) + step[2:]
                 for step in invocation)


def _step_ordinal(invocation, region):
    for step in invocation:
        if step[0] == region:
            return step[1]
    return None


class _ManyEvaluations:
    """The saturated count of two or more evaluations, without their identities."""

    __slots__ = ()

    def __repr__(self):
        return 'MANY'


_SATURATED = frozenset((_ManyEvaluations(),))


@dataclass(frozen=True)
class _Bag:
    """One alternative: delivered evaluations and still unbound carried loop paths."""

    refs: frozenset = frozenset()
    prev: frozenset = frozenset()


@dataclass(frozen=True)
class _Prev:
    """The carried path of the previous iteration of one loop analysis, or a combine argument."""

    loop: object
    path: tuple


class _Leaf:
    __slots__ = ('alternatives',)

    def __init__(self, alternatives):
        self.alternatives = alternatives


class _Node:
    __slots__ = ('level', 'lo', 'hi')

    def __init__(self, level, lo, hi):
        self.level, self.lo, self.hi = level, lo, hi


_TOP = float('inf')


class _Diagrams:
    """Hash-consed decision diagrams; a variable is a named condition (SSA id, invocation)."""

    def __init__(self, merge):
        self.merge = merge
        self.levels, self.variables = {}, []
        self.nodes, self.leaves = {}, {}
        self.applied, self.chosen = {}, {}
        self.fail = self.leaf(frozenset())
        self.empty = self.leaf(frozenset((_Bag(),)))

    def leaf(self, alternatives):
        found = self.leaves.get(alternatives)
        if found is None:
            found = self.leaves[alternatives] = _Leaf(alternatives)
        return found

    def level(self, variable):
        found = self.levels.get(variable)
        if found is None:
            found = self.levels[variable] = len(self.variables)
            self.variables.append(variable)
        return found

    def node(self, level, lo, hi):
        if lo is hi:
            return lo
        key = (level, id(lo), id(hi))
        found = self.nodes.get(key)
        if found is None:
            found = self.nodes[key] = _Node(level, lo, hi)
        return found

    @staticmethod
    def top(diagram):
        return diagram.level if type(diagram) is _Node else _TOP

    @staticmethod
    def cofactor(diagram, level, arm):
        if type(diagram) is _Node and diagram.level == level:
            return diagram.hi if arm else diagram.lo
        return diagram

    def join(self, left, right):
        return _Bag(self.merge(left.refs, right.refs), left.prev | right.prev)

    def seq(self, left, right):
        """Both parts evaluate on one path: their deliveries add up."""
        if left is self.fail or right is self.fail:
            return self.fail
        if left is self.empty:
            return right
        if right is self.empty:
            return left
        return self._apply(True, left, right)

    def alt(self, left, right):
        """Either part, without a condition that names the choice."""
        if left is self.fail or left is right:
            return right
        if right is self.fail:
            return left
        return self._apply(False, left, right)

    def _apply(self, sequential, left, right):
        key = (sequential, id(left), id(right))
        found = self.applied.get(key)
        if found is not None:
            return found
        if type(left) is _Leaf and type(right) is _Leaf:
            if sequential:
                found = self.leaf(frozenset(self.join(a, b) for a in left.alternatives
                                            for b in right.alternatives))
            else:
                found = self.leaf(left.alternatives | right.alternatives)
        else:
            level = min(self.top(left), self.top(right))
            combine = self.seq if sequential else self.alt
            found = self.node(level,
                combine(self.cofactor(left, level, False), self.cofactor(right, level, False)),
                combine(self.cofactor(left, level, True), self.cofactor(right, level, True)))
        self.applied[key] = found
        return found

    def choose(self, level, hi, lo):
        """``hi`` where the condition of ``level`` holds, ``lo`` elsewhere."""
        if hi is lo:
            return hi
        key = (level, id(hi), id(lo))
        found = self.chosen.get(key)
        if found is not None:
            return found
        top = min(level, self.top(hi), self.top(lo))
        if top == level:
            found = self.node(level, self.cofactor(lo, level, False), self.cofactor(hi, level, True))
        else:
            found = self.node(top,
                self.choose(level, self.cofactor(hi, top, False), self.cofactor(lo, top, False)),
                self.choose(level, self.cofactor(hi, top, True), self.cofactor(lo, top, True)))
        self.chosen[key] = found
        return found

    def rebuild(self, diagram, variable, leaf):
        """Rename or quantify (``variable`` gives None) conditions and replace leaves by diagrams."""
        memo = {}
        def go(current):
            found = memo.get(id(current))
            if found is not None:
                return found
            if type(current) is _Leaf:
                found = leaf(current)
            else:
                lo, hi = go(current.lo), go(current.hi)
                renamed = variable(self.variables[current.level])
                found = (self.alt(lo, hi) if renamed is None else
                         self.choose(self.level(renamed), hi, lo))
            memo[id(current)] = found
            return found
        return go(diagram)

    def distinct_leaves(self, diagram):
        seen, leaves, stack = set(), [], [diagram]
        while stack:
            current = stack.pop()
            if id(current) in seen:
                continue
            seen.add(id(current))
            if type(current) is _Leaf:
                leaves.append(current)
            else:
                stack.extend((current.lo, current.hi))
        return leaves

    def paths(self, diagram, choices=()):
        """Every path to a successful leaf, true arm first, as (choices, leaf)."""
        if type(diagram) is _Leaf:
            if diagram.alternatives:
                yield frozenset(choices), diagram
            return
        variable = self.variables[diagram.level]
        yield from self.paths(diagram.hi, choices + ((variable, True),))
        yield from self.paths(diagram.lo, choices + ((variable, False),))


def _erase(reference):
    return DeliveryReference(reference.value_id,
        tuple((step[0], MANY) + step[2:] if step[1] in (SOME, MANY) else step
              for step in reference.invocation), reference.projection)


def _merge_exact(left, right):
    """Union of evaluations. Equal identified evaluations are one; two summarized ones are many."""
    if not left or left is right and not any(not identified(r) for r in left):
        return right
    if not right:
        return left
    merged = left | right
    summarized = [r for r in merged if not identified(r)]
    if not summarized:
        return merged
    shared = left & right
    groups = {}
    for reference in summarized:
        groups.setdefault(_erase(reference), []).append(reference)
    result = set(r for r in merged if identified(r))
    for erased, members in groups.items():
        if len(members) > 1 or members[0] in shared or members[0] == erased:
            result.add(erased)
        else:
            result.add(members[0])
    return frozenset(result)


def _saturate(references):
    if references is _SATURATED or not references:
        return references
    if len(references) > 1 or delivered_count(references) > 1:
        return _SATURATED
    return references


def _merge_saturated(left, right):
    if left is _SATURATED or right is _SATURATED:
        return _SATURATED
    return _saturate(_merge_exact(left, right))


class _Context:
    """A region layer: its step applies to the region's nodes; bindings of its arguments."""

    __slots__ = ('parent', 'step', 'nodes', 'bindings')

    def __init__(self, parent, step, nodes, bindings):
        self.parent, self.step, self.nodes, self.bindings = parent, step, nodes, bindings


class _LoopState:
    """One loop analysed in one context: its body is visited once, over placeholders."""

    __slots__ = ('loop', 'context', 'body_context', 'prefix', 'key')

    def __init__(self, loop, context, prefix, key, body_context):
        self.loop, self.context, self.prefix, self.key = loop, context, prefix, key
        # The body context binds the loop argument to this state, so it is made from it.
        self.body_context = body_context(self)


class _Formal:
    """A combine argument of one scatter_reduce analysis, bound to a placeholder.

    ``sources`` are the (value, context) pairs whose deliveries it may carry: the initial
    value, the source values and the earlier combinations for the accumulator, the source
    values for the element.
    """

    __slots__ = ('key', 'sources')

    def __init__(self, key):
        self.key = key
        self.sources: tuple = ()


class _Analysis:
    def __init__(self, matches, *, saturate, with_projection=False, expand_match=None, stop_at=None):
        self.matches, self.stop_at = matches, stop_at
        self.expand_match, self.with_projection = expand_match, with_projection
        self.saturate = saturate
        self.dd = _Diagrams(_merge_saturated if saturate else _merge_exact)
        self.root = _Context(None, None, frozenset(), {})
        self.contexts, self.region_sets, self.invocations = {}, {}, {}
        self.memo, self.affect, self.deliver = {}, {}, {}
        self.loop_states, self.sequences, self.formals = {}, {}, {}

    # -- contexts ---------------------------------------------------------------------------
    def nodes_of(self, owner, *blocks):
        key = (owner.id, blocks)
        found = self.region_sets.get(key)
        if found is None:
            found = self.region_sets[key] = frozenset(
                node.id for block in blocks for node in _region_nodes(owner.attrs[block]))
        return found

    def context(self, parent, step, nodes, bindings=()):
        key = (id(parent), step, nodes, tuple(
            (argument, kind, id(target), id(where), prefix)
            for argument, (kind, target, where, prefix) in bindings))
        found = self.contexts.get(key)
        if found is None:
            found = self.contexts[key] = _Context(parent, step, nodes, dict(bindings))
        return found

    def invocation(self, node_id, context):
        key = (node_id, id(context))
        found = self.invocations.get(key)
        if found is None:
            steps, current = [], context
            while current is not None:
                if current.step is not None and node_id in current.nodes:
                    steps.append(current.step)
                current = current.parent
            found = self.invocations[key] = tuple(reversed(steps))
        return found

    @staticmethod
    def binding(node_id, context):
        while context is not None:
            found = context.bindings.get(node_id)
            if found is not None:
                return found
            context = context.parent
        return None

    # -- reachability -----------------------------------------------------------------------
    def dependencies(self, value):
        found = list(value.inputs)
        if value.op != 'provenance_bind':
            for name, attribute in value.attrs.items():
                if not name.endswith('_block') and not name.startswith(('scientific_', 'realization_')):
                    found.extend(_refs(attribute))
        return found

    def reaches(self, value, context, *, terminal, assumed=frozenset(), active=None):
        """Whether a delivery (or, with ``terminal``, a terminal control) may flow from value.

        A carried loop argument may affect what its initial value or its body may; while the
        body is examined, the argument itself is assumed not to (least fixpoint).
        """
        table = self.affect if terminal else self.deliver
        key = (value.id, id(context), assumed)
        if key in table:
            return table[key]
        active = set() if active is None else active
        if key in active:
            return True
        active.add(key)
        binding = self.binding(value.id, context)
        if self.matches(value) or (terminal and value.op in {
                'result_error', 'variant_unreachable', 'variant_construct',
                'structured_fold', 'structured_iterate'}):
            answer = True
        elif self.stop_at is not None and self.stop_at(value):
            answer = False
        elif binding is not None and binding[0] == 'value':
            answer = self.reaches(binding[1], binding[2], terminal=terminal, assumed=assumed, active=active)
        elif binding is not None and binding[0] == 'formal':
            formal = binding[1]
            answer = formal.key not in assumed and any(
                self.reaches(source, where, terminal=terminal, assumed=assumed | {formal.key}, active=active)
                for source, where in formal.sources)
        elif binding is not None:
            state = binding[1]
            answer = state.key not in assumed and (
                self.reaches(state.loop.inputs[0], state.context, terminal=terminal,
                             assumed=assumed, active=active) or
                self.reaches(state.loop.attrs['body'], state.body_context, terminal=terminal,
                             assumed=assumed | {state.key}, active=active))
        else:
            answer = any(self.reaches(item, context, terminal=terminal, assumed=assumed, active=active)
                         for item in self.dependencies(value))
        active.remove(key)
        table[key] = answer
        return answer

    # -- leaves -----------------------------------------------------------------------------
    def evaluation(self, value, path, context):
        reference = DeliveryReference(value.id, self.invocation(value.id, context),
                                      tuple(path) if self.with_projection else ())
        refs = frozenset((reference,))
        return self.dd.leaf(frozenset((_Bag(_saturate(refs) if self.saturate else refs),)))

    def known(self, condition):
        if condition.op == 'bool_constant':
            return condition.attrs['value']
        if condition.op == 'variant_is' and condition.inputs[0].op == 'variant_construct':
            return condition.attrs['tag'] == condition.inputs[0].attrs['tag']
        if condition.op == 'result_is_ok' and condition.inputs[0].op in {'result_ok', 'result_error'}:
            return condition.inputs[0].op == 'result_ok'
        return None

    def level(self, condition, context):
        # A condition is ordered when its choice is entered, before its arms: an outer
        # choice stays above the choices made inside its arms, as in the b6 success paths.
        if self.known(condition) is not None:
            return None
        return self.dd.level((condition.id, self.invocation(condition.id, context)))

    def select(self, condition, hi, lo, context, level=None):
        known = self.known(condition)
        if known is not None:
            return hi if known else lo
        return self.dd.choose(self.level(condition, context) if level is None else level, hi, lo)

    def same_cells(self, left, right, value):
        # A global frontier cannot represent a spatial/component partition of a Use.
        if left is not right:
            raise DeliveryNotStructural('per-cell selection has different delivery frontiers; '
                                        'realize the final selected value explicitly',
                                        region='%s#%d' % (value.op, value.id), construct=value.op)
        return left

    # -- visit ------------------------------------------------------------------------------
    def visit(self, value, path, context):
        key = (value.id, path, id(context))
        found = self.memo.get(key)
        if found is not None:
            return found
        found = self.memo[key] = self._visit(value, path, context)
        return found

    def _visit(self, value, path, context):
        dd = self.dd
        # Terminal failures have no successful delivery frontier. Reachability
        # pruning must not turn that absence into a successful empty frontier.
        if (value.op in {'variant_unreachable', 'result_error'} or
                (value.op == 'variant_construct' and path and path[0] != value.attrs['tag'])):
            return dd.fail
        if not self.reaches(value, context, terminal=True):
            return dd.empty
        binding = self.binding(value.id, context)
        if binding is not None:
            kind, target, where, prefix = binding
            if kind == 'value':
                return self.visit(target, prefix + path, where)
            return dd.leaf(frozenset((_Bag(prev=frozenset((_Prev(target.key, prefix + path),))),)))
        if self.matches(value):
            expanded = None if self.expand_match is None else self.expand_match(value, tuple(path))
            if expanded is None:
                return self.evaluation(value, path, context)
            dependency, projection = expanded
            if not isinstance(dependency, ProgramValue) or dependency.prog is not value.prog:
                raise TypeError('delivery match expansion requires a value from the exact Program')
            projection = tuple(projection)
            if dependency is value and projection == tuple(path):
                raise ValueError('delivery match expansion cannot reproduce itself')
            # Metadata references are followed only by this explicit internal
            # query, in the SAME native invocation as their owner.
            return self.visit(dependency, projection, context)
        if self.stop_at is not None and self.stop_at(value):
            return dd.empty
        op = value.op
        if op == 'product_get':
            return self.visit(value.inputs[0], (value.attrs['field'],) + path, context)
        if op in {'continuation_publish', 'continuation_accept',
                  'continuation_accepted_candidate', 'continuation_candidate_value'}:
            # Assessment/checks are not delivered numerical values.
            return self.visit(value.inputs[0], path, context)
        if op == 'continuation_freeze':
            return self.visit(value.inputs[2], path, context)
        if op in {'continuation_transform', 'bundle_remap_validate'}:
            # Only the transformed/remapped candidate is a retained value. The
            # old Bundle is a validation witness, not a second numerical delivery.
            return self.visit(value.inputs[1], path, context)
        if op in {'bundle_validate', 'continuation_read'}:
            return self.visit(value.inputs[0], path, context)
        if op == 'continuation_child_base':
            return self.visit(value.inputs[0], ('continuation',) + path, context)
        if op in {'continuation_open', 'continuation_proposal', 'continuation_candidate_base',
                  'continuation_attest', 'continuation_assessment_checks',
                  'continuation_assessment_ok'}:
            # The previous accepted image is a boundary, not a new delivery.
            return dd.empty
        if op == 'continuation_consume_increment':
            return self.visit(value.inputs[0], ('increment',) + path, context)
        if op in {'finite_map', 'distributed_map'}:
            return self.mapped(value, path, context)
        if op == 'distributed_scatter_reduce':
            return self.scattered(value, path, context)
        if op in {'finite_domain', 'finite_size', 'finite_same_domain',
                  'finite_index_value', 'finite_index', 'finite_index_argument',
                  'distributed_value_domain', 'distributed_size',
                  'distributed_gid', 'distributed_index', 'distributed_index_argument',
                  'distributed_offset_argument', 'distributed_rank'}:
            # Geometry, cardinality and predicates cannot deliver an inventory.
            return dd.empty
        if path and op == 'product':
            fields = tuple(value.attrs['fields'])
            return self.visit(value.inputs[fields.index(path[0])], path[1:], context)
        if op in {'branch', 'result_match'}:
            condition = value.inputs[0] if op == 'branch' else value.inputs[1]
            level = self.level(condition, context)
            return self.select(condition, self.visit(value.attrs['true_result'], path, context),
                               self.visit(value.attrs['false_result'], path, context), context, level)
        if op == 'where':
            # Its predicate is a control dependency, never an inventory delivery.
            return self.same_cells(self.visit(value.inputs[1], path, context),
                                   self.visit(value.inputs[2], path, context), value)
        if op in {'pointwise_expression', 'scalar_expression'}:
            return self.expressions(value, path, context)
        if op in {'structured_fold', 'structured_iterate'}:
            return self.loop(value, path, context)
        if op == 'variant_require_success':
            return self.visit(value.inputs[0], ('ok',) + path, context)
        if op == 'variant_payload':
            return self.visit(value.inputs[0], (value.attrs['tag'],) + path, context)
        if op in {'result_payload', 'iteration_payload'}:
            # Legacy error payloads contain status code/origin/iteration count only.
            # Traversing the result's success value here would borrow a delivery
            # from a mutually exclusive, possibly unexecuted branch.
            return (dd.empty if value.attrs['variant'] == 'err' else
                    self.visit(value.inputs[0], path, context))
        if op in {'result_is_ok', 'iteration_is_ok', 'variant_is'}:
            return dd.empty
        if op == 'variant_construct':
            return self.visit(value.inputs[0], path[1:] if path else (), context)
        if op in {'provenance_bind', 'acceptance_guard', 'result_ok', 'result_success',
                  'iteration_success', 'region_argument'}:
            return self.visit(value.inputs[0], path, context)
        if op == 'capture_result':
            if path and path[0] == 'err':
                raise DeliveryNotStructural('delivery from captured failure payload requires an '
                                            'explicit failure frontier',
                                            region='capture_result#%d' % value.id, construct=op)
            body_path = path
            if value.vtype == 'variant' and path and path[0] == 'ok':
                body_path = path if value.attrs.get('body_is_variant_result') else path[1:]
            return self.visit(value.attrs['body'], body_path, context)
        if any(name.endswith('_block') and attribute for name, attribute in value.attrs.items()):
            raise DeliveryNotStructural('no delivery rule for native region ' + op,
                                        region='%s#%d' % (op, value.id), construct=op)
        result = dd.empty
        support_only = value.attrs.get('support_only_inputs', ())
        for index, dependency in enumerate(value.inputs):
            if index not in support_only:
                result = dd.seq(result, self.visit(dependency, path, context))
        for name, attribute in value.attrs.items():
            if not name.endswith('_block') and not name.startswith(('scientific_', 'realization_')):
                for dependency in _refs(attribute):
                    result = dd.seq(result, self.visit(dependency, path, context))
        return result

    def expressions(self, value, path, context):
        dd, nodes, cached = self.dd, value.attrs['expression_nodes'], {}
        def expression(index):
            if index in cached:
                return cached[index]
            node = nodes[index]
            op = node[0]
            if op == 'input':
                found = self.visit(value.inputs[node[1]], path, context)
            elif op in {'literal', 'coefficient'}:
                found = dd.empty
            elif op == 'where':
                # Its predicate is a control dependency, never an inventory delivery.
                found = self.same_cells(expression(node[2]), expression(node[3]), value)
            else:
                if op in {'add', 'sub', 'mul', 'div', 'pow', 'minimum', 'maximum', 'and', 'or',
                          'neg', 'sqrt', 'exp', 'sin', 'cos', 'log', 'expm1', 'abs', 'sign',
                          'not', 'rounded'}:
                    children = node[1:]
                elif op == 'compare':
                    children = node[2:]
                elif op == 'finite_linear_v1':
                    children = node[5]
                elif op == 'finite_projection_v1':
                    children = (node[1],)
                else:
                    raise DeliveryNotStructural('no delivery rule for scalar expression ' + str(op),
                                                region='%s#%d' % (value.op, value.id), construct=str(op))
                found = dd.empty
                for child in children:
                    found = dd.seq(found, expression(child))
            cached[index] = found
            return found
        result = dd.empty
        for index in value.attrs['expressions']:
            result = dd.seq(result, expression(index))
        return result

    # -- maps -------------------------------------------------------------------------------
    def mapped(self, value, path, context):
        """A map delivers element by element: its body is visited once, as every element.

        A map that owns no Use transports the nearest Array-valued contribution through its
        element expression, in its caller's invocation; it cannot borrow an unused capture or
        its domain/cardinality as evidence of a numerical delivery.
        """
        nodes = self.nodes_of(value, 'body_block')
        if not any(self.matches(node) for node in _region_nodes(value.attrs['body_block'])):
            return self.visit(value.attrs['body'], path, context)
        element = self.visit(value.attrs['body'], path,
                             self.context(context, (value.id, EACH, 'map'), nodes))
        domain = value.inputs[0]
        if domain.op == 'finite_range_set' and _count(domain.inputs[0]) == 0:
            # A literal empty domain evaluates no element, so it delivers nothing.
            return self.dd.empty
        return element

    # -- scatters ---------------------------------------------------------------------------
    def combine_formals(self, value, context):
        key = (value.id, id(context))
        found = self.formals.get(key)
        if found is None:
            accumulator, element = _Formal(key + ('accumulator',)), _Formal(key + ('element',))
            body = self.context(context, (value.id, EACH, 'combine'), self.nodes_of(value, 'body_block'), (
                (value.attrs['body_argument'].id, ('formal', accumulator, None, ())),
                (value.attrs['body_value'].id, ('formal', element, None, ()))))
            values, initial = value.inputs[0], value.inputs[3]
            accumulator.sources = ((initial, context), (values, context), (value.attrs['body'], body))
            element.sources = ((values, context),)
            found = self.formals[key] = (accumulator, element, body)
        return found

    def scattered(self, value, path, context):
        """A scatter_reduce delivers each source element once, to its target (IR19).

        Its combine is visited once, as every combination, with its accumulator and its
        element bound to placeholders. On every path it must combine the same parts: the
        accumulator at the path of the result (the initial value and the elements already
        combined) and the element, and deliver nothing of its own. Then each source element
        reaches its one target exactly once, and the result delivers what the initial value
        and the source values deliver. The indices and the target domain are addresses, never
        deliveries.
        """
        dd, region = self.dd, '%s#%d' % (value.op, value.id)
        values, initial = value.inputs[0], value.inputs[3]
        accumulator, element, body = self.combine_formals(value, context)
        bags = {bag for found in dd.distinct_leaves(self.visit(value.attrs['body'], path, body))
                for bag in found.alternatives}
        if not bags:
            return dd.fail
        def refuse(message):
            raise DeliveryNotStructural('the combine of a scatter_reduce ' + message,
                                        region=region, construct=value.op)
        if len(bags) > 1:
            refuse('must combine the same parts on every path')
        bag = bags.pop()
        if bag.refs:
            refuse('delivers a Use of its own once per combination')
        kept = _Prev(accumulator.key, path)
        if any(item.loop != element.key and item != kept for item in bag.prev):
            refuse('may combine only its accumulator, at the path of its result, and its element')
        if kept not in bag.prev and (self.reaches(initial, context, terminal=False) or
                                     self.reaches(values, context, terminal=False)):
            refuse('must keep its accumulator: the initial value and the elements already combined')
        delivered = sorted((item.path for item in bag.prev if item.loop == element.key), key=repr)
        if not delivered and self.reaches(values, context, terminal=False):
            refuse('must deliver its element to the target')
        result = self.visit(initial, path, context)
        for element_path in delivered:
            result = dd.seq(result, self.visit(values, element_path, context))
        return result

    # -- loops ------------------------------------------------------------------------------
    def loop(self, value, path, context):
        dd = self.dd
        if not self.reaches(value, context, terminal=False):
            return self.loop_without_delivery(value, path, context)
        is_fold = value.op == 'structured_fold'
        if not is_fold:
            if path and path[0] == 'ok':
                path = path[1:]
            if path and path[0] == 'iterations':
                return dd.empty
            if not path or path[0] != 'state':
                raise DeliveryNotStructural('delivery through iterate requires its exact successful '
                                            'state projection', region='%s#%d' % (value.op, value.id),
                                            construct=value.op)
            path = path[1:]
        literal = _count(value.inputs[1]) if is_fold else None
        capacity = _count(value.inputs[2] if is_fold else value.inputs[1])
        if ((literal is not None and literal < 0) or (capacity is not None and capacity < 0) or
                (literal is not None and capacity is not None and literal > capacity)):
            return dd.fail
        limit = literal if literal is not None else capacity
        condition = None if is_fold else value.attrs['cond']
        nodes = None if is_fold else self.nodes_of(value, 'cond_block')
        def stop(ordinal):
            return self.context(context, (value.id, ordinal, 'stop'), nodes)
        if limit == 0 or (not is_fold and self.known(condition) is True):
            # No iteration runs: the body is not examined.
            initial = self.visit(value.inputs[0], path, context)
            return initial if is_fold else self.select(condition, initial, dd.fail, stop(0))
        sequence = self.sequence(value, path, context)
        if is_fold and literal is not None:
            return sequence.state(literal)[path]
        if is_fold:
            result = dd.fail
            for state in sequence.states(limit):
                result = dd.alt(result, state[path])
            return result
        accepted, continued = dd.fail, dd.empty
        exact, symbolic = sequence.split(limit)
        for iteration in range(exact + 1):
            where = stop(iteration)
            state = sequence.state(iteration)[path]
            accepted = dd.alt(accepted, dd.seq(continued, self.select(condition, state, dd.fail, where)))
            continued = dd.seq(continued, self.select(condition, dd.fail, dd.empty, where))
            if continued is dd.fail:
                return accepted
        where = stop(Ordinal('final'))
        for state in symbolic:
            accepted = dd.alt(accepted, dd.seq(continued, self.select(condition, state[path], dd.fail, where)))
        return accepted

    def loop_state(self, value, context):
        key = (value.id, id(context))
        found = self.loop_states.get(key)
        if found is None:
            found = self.loop_states[key] = _LoopState(
                value, context, ('ok',) if value.attrs.get('body_is_variant_result') else (),
                len(self.loop_states), lambda state: self.context(
                    context, (value.id, _CURRENT), self.nodes_of(value, 'body_block'),
                    ((value.attrs['body_argument'].id, ('state', state, None, ())),)))
        return found

    def sequence(self, value, path, context):
        state = self.loop_state(value, context)
        paths, transfers = [path], {}
        for carried in paths:
            transfers[carried] = self.visit(value.attrs['body'], state.prefix + carried, state.body_context)
            for diagram_leaf in self.dd.distinct_leaves(transfers[carried]):
                for bag in diagram_leaf.alternatives:
                    for previous in bag.prev:
                        if previous.loop == state.key and previous.path not in transfers and previous.path not in paths:
                            paths.append(previous.path)
        key = (state.key, frozenset(paths))
        found = self.sequences.get(key)
        if found is None:
            found = self.sequences[key] = _Sequence(self, state, tuple(paths), transfers)
        return found

    def loop_without_delivery(self, value, path, context):
        """Summarize possible success, without certifying a numerical solve.

        No Use is reachable here, so iterating cannot discover a delivery or
        multiplicity. Numerical stop/count predicates remain opaque: retaining a
        possible empty frontier is conservative for delivery checking. Explicit
        terminal failures are still eliminated; this is not a convergence proof.
        """
        dd = self.dd
        is_fold = value.op == 'structured_fold'
        capacity = _count(value.inputs[2] if is_fold else value.inputs[1])
        if not is_fold:
            if path and path[0] == 'err':
                # A handled failure is data, not a delivery from the success arm.
                return dd.empty
            if path and path[0] == 'ok':
                path = path[1:]
            if path and path[0] == 'state':
                state_path = path[1:]
            elif path and path[0] == 'iterations':
                state_path = ()
            else:
                raise DeliveryNotStructural('delivery through iterate requires its exact successful '
                                            'state projection', region='%s#%d' % (value.op, value.id),
                                            construct=value.op)
        else:
            state_path = path
        initial = self.visit(value.inputs[0], state_path, context)
        for control in value.inputs[1:]:
            initial = dd.seq(initial, self.visit(control, (), context))
        if initial is dd.fail or (capacity is not None and capacity < 0):
            return dd.fail
        previous = ('value', value.inputs[0], context, ())
        if is_fold:
            literal = _count(value.inputs[1])
            if literal is not None and (literal < 0 or (capacity is not None and literal > capacity)):
                return dd.fail
            if literal is None or literal == 0:
                # Zero iterations are possible for a dynamic count.
                return initial
            first = self.visit(value.attrs['body'], state_path, self.context(
                context, (value.id, 1), self.nodes_of(value, 'body_block'),
                ((value.attrs['body_argument'].id, previous),)))
            # A positive fold cannot succeed if its first evaluation terminates.
            # Later numerical branches are not examined in this Use-free graph.
            return dd.empty if dd.seq(initial, first) is not dd.fail else dd.fail
        where = self.context(context, None, frozenset(), ((value.attrs['cond_argument'].id, previous),))
        condition = value.attrs['cond']
        initial = dd.seq(initial, self.visit(condition, (), where))
        if initial is dd.fail:
            return dd.fail
        stopped = self.select(condition, initial, dd.fail, where)
        if capacity == 0 or condition.op == 'bool_constant':
            # A constant-false stop never accepts, regardless of the budget.
            return stopped
        # The unknown predicate may accept before stepping. Do not infer either
        # convergence or exhaustion from its numerical expression or its budget.
        return dd.empty


def _count(value):
    return value.attrs['value'] if value.op == 'int64_constant' else None


class _Sequence:
    """The carried state of one loop after each iteration, and its periodic abstraction.

    ``exact[k]`` maps each carried path to its diagram after k iterations, with literal
    iteration ordinals. ``abstract[k]`` is the same state with ages relative to iteration k,
    exact up to ``horizon`` (the number of carried paths) and summarized beyond. It is
    computed until it repeats: ``abstract[start + period] == abstract[start]``; then every
    later iteration k has the abstract state at ``start + (k - start) % period``.
    """

    def __init__(self, analysis, state, paths, transfers):
        self.analysis, self.dd, self.carrier = analysis, analysis.dd, state
        self.loop, self.paths, self.transfers = state.loop.id, paths, transfers
        self.horizon = len(paths)
        initial = {path: analysis.visit(state.loop.inputs[0], path, state.context) for path in paths}
        self.exact, self.abstract, seen = [initial], [initial], {self._key(initial): 0}
        while True:
            self.exact.append(self._step(self.exact[-1], len(self.exact)))
            following = self._summarize(self._step(self._shift(self.abstract[-1]), _age(0)))
            key = self._key(following)
            if key in seen:
                self.start = seen[key]
                self.period = len(self.abstract) - self.start
                break
            seen[key] = len(self.abstract)
            self.abstract.append(following)

    def _key(self, state):
        return tuple(id(state[path]) for path in self.paths)

    def _step(self, previous, ordinal):
        """One iteration: the body over the previous carried state, at ``ordinal``."""
        dd, loop, key = self.dd, self.loop, self.carrier.key
        def current(old):
            return ordinal if old == _CURRENT else old
        def variable(name):
            return name[0], _rename(name[1], loop, current)
        def leaf(found):
            result = dd.fail
            for bag in found.alternatives:
                refs = bag.refs if bag.refs is _SATURATED else frozenset(
                    DeliveryReference(r.value_id, _rename(r.invocation, loop, current), r.projection)
                    for r in bag.refs)
                own = [item for item in bag.prev if item.loop == key]
                found_bag = dd.leaf(frozenset((_Bag(refs, bag.prev.difference(own)),)))
                for item in own:
                    found_bag = dd.seq(found_bag, previous[item.path])
                result = dd.alt(result, found_bag)
            return result
        return {path: dd.rebuild(self.transfers[path], variable, leaf) for path in self.paths}

    def _map_state(self, state, ordinal, drop=lambda ordinal: False):
        """Rename this loop's ordinals of a state; quantify conditions whose ordinal drops."""
        dd, loop = self.dd, self.loop
        def variable(name):
            old = _step_ordinal(name[1], loop)
            if old is not None and drop(old):
                return None
            return name[0], _rename(name[1], loop, ordinal)
        def leaf(found):
            alternatives = set()
            for bag in found.alternatives:
                merged = bag.refs
                if merged is not _SATURATED:
                    # Two evaluations that become one summarized ordinal are many.
                    merged = frozenset()
                    for r in bag.refs:
                        merged = dd.merge(merged, frozenset((DeliveryReference(
                            r.value_id, _rename(r.invocation, loop, ordinal), r.projection),)))
                alternatives.add(_Bag(merged, bag.prev))
            return dd.leaf(frozenset(alternatives))
        return {path: dd.rebuild(state[path], variable, leaf) for path in self.paths}

    def _shift(self, state):
        return self._map_state(state, lambda old: _age(old.offset + 1) if old.kind == 'age' else old)

    def _summarize(self, state):
        horizon = self.horizon
        def older(old):
            return old.kind == 'age' and old.offset > horizon
        return self._map_state(state, lambda old: SOME if older(old) else old, drop=older)

    def _concrete(self, state, count):
        """The abstract state as the state after ``count`` iterations (None: an unknown count)."""
        if count is None:
            return self._map_state(state, lambda old: Ordinal('final', old.offset) if old.kind == 'age' else old)
        return self._map_state(state, lambda old: count - old.offset if old.kind == 'age' else old)

    def state(self, count):
        if count < len(self.exact):
            return self.exact[count]
        return self._concrete(self.abstract[self.start + (count - self.start) % self.period], count)

    def split(self, limit):
        """Exact iterations [0, n] and the symbolic states of the later ones, up to ``limit``."""
        last = len(self.exact) - 1
        if limit is not None and limit <= last:
            return limit, ()
        later = range(last + 1, last + 1 + self.period if limit is None else min(limit, last + self.period) + 1)
        positions = sorted({self.start + (k - self.start) % self.period for k in later})
        return last, tuple(self._concrete(self.abstract[position], None) for position in positions)

    def states(self, limit):
        exact, symbolic = self.split(limit)
        return tuple(self.exact[count] for count in range(exact + 1)) + symbolic


def _analysis(root, matches, path, **options):
    analysis = _Analysis(matches, **options)
    return analysis, analysis.visit(root, tuple(path), analysis.root)


def _bag_order(bag):
    return (len(bag.refs), sorted((r.value_id, repr(r.invocation), repr(r.projection)) for r in bag.refs))


def delivered_frontiers(root, matches, *, path=(), with_choices=False, with_projection=False,
                        expand_match=None, stop_at=None):
    """The delivered evaluations on every possible success path, with their choices.

    Dynamic loop counts include zero. Count/stop predicates are control, not
    delivery. ``stop_at`` identifies value computations whose inputs are numerical
    data, rather than deliveries of this caller's queried occurrence. Matches are
    inspected first; the optional frontier cannot hide the matched value itself.
    A path to a leaf with several alternatives gives one row per alternative.
    """
    analysis, diagram = _analysis(root, matches, path, saturate=False, with_projection=with_projection,
                                  expand_match=expand_match, stop_at=stop_at)
    rows = []
    for choices, found in analysis.dd.paths(diagram):
        for bag in sorted(found.alternatives, key=_bag_order):
            if bag.prev:
                raise RuntimeError('a carried loop path escaped its loop analysis')
            rows.append((choices, bag.refs) if with_choices else bag.refs)
    return tuple(rows)


def require_single_delivery(root, matches, *, term, path=(), stop_at=None, missing, duplicate):
    """Check that every success path delivers exactly one evaluation of ``term`` (D5, IR19).

    ``missing`` and ``duplicate`` are the caller's diagnostics; the refusal is
    ``CoverageNotExact`` naming the term, checked first for a missing delivery (a path, or
    no successful path at all), then for a duplicate one. An untyped delivery path is
    ``DeliveryNotStructural``, naming the term. Returns the single delivered references.
    """
    try:
        analysis, diagram = _analysis(root, matches, path, saturate=True, stop_at=stop_at)
    except DeliveryNotStructural as error:
        if error.term is not None:
            raise
        raise DeliveryNotStructural(error.detail, region=error.region, construct=error.construct,
                                    term=term) from error
    leaves = [found for found in analysis.dd.distinct_leaves(diagram) if found.alternatives]
    bags = [bag for found in leaves for bag in found.alternatives]
    if not bags or any(not bag.refs for bag in bags):
        raise CoverageNotExact(missing, term=term, count=0)
    if any(bag.refs is _SATURATED for bag in bags):
        raise CoverageNotExact(duplicate, term=term, count=2)
    return frozenset(reference for bag in bags for reference in bag.refs)


__all__ = ['DeliveryReference', 'EACH', 'MANY', 'Ordinal', 'SOME', 'delivered_count', 'delivered_frontiers',
           'identified', 'multiplicity', 'require_identified', 'require_single_delivery', 'term_label']
