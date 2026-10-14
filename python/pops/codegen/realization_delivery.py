"""Finite delivery analysis over success paths and native loop occurrences.

This is a structural check, never an equality or consistency theorem. A loop
body's SSA identifiers recur at runtime; their different evaluations must not
collapse to a single delivery merely because the compiler reused one identifier.
"""
from collections.abc import Mapping
from dataclasses import dataclass

from pops.time.values import ProgramValue


@dataclass(frozen=True)
class DeliveryReference:
    value_id: int
    invocation: tuple
    projection: tuple = ()


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


class _Frame:
    def __init__(self, bindings=None, invocations=None):
        self.bindings = bindings or {}
        self.invocations = invocations or {}


def delivered_frontiers(root, matches, *, path=(), max_alternatives=4096, max_iterations=256, with_choices=False, with_projection=False, expand_match=None):
    """Find the nearest delivered value on every finite possible success path.

    Dynamic loop counts include zero. Count/stop predicates are control, not
    delivery. The explicit analysis budgets fail closed and are independent of
    the numerical loop capacity; no unchecked result is substituted at a limit.
    """
    if type(max_alternatives) is not int or max_alternatives <= 0:
        raise ValueError('realization delivery alternative budget must be a positive integer')
    if type(max_iterations) is not int or max_iterations < 0:
        raise ValueError('realization delivery iteration budget must be a nonnegative integer')
    empty = ((frozenset(), frozenset()),)
    memo = {}
    possible = {}
    delivery_possible = {}

    def may_affect_frontier(value, frame, active=None):
        """Retain delivery and explicit terminal control dependencies transitively.

        An unowned alias of an error can still remove a successful path. Tagged
        injections likewise matter when a caller projects a different arm.
        Prune only graphs without these dependencies; exact projection, success
        paths and invocation checks still run with the same finite budgets.
        Region arguments resolve through their actual caller frame.
        """
        key = (value.id, frame)
        if key in possible:
            return possible[key]
        active = set() if active is None else active
        if key in active:
            return True
        active.add(key)
        if matches(value) or value.op in {'result_error', 'variant_unreachable', 'variant_construct',
                                          'structured_fold', 'structured_iterate'}:
            answer = True
        elif value.id in frame.bindings:
            target, bound_frame, _ = frame.bindings[value.id]
            answer = may_affect_frontier(target, bound_frame, active)
        else:
            dependencies = list(value.inputs)
            for name, attribute in value.attrs.items():
                if value.op != 'provenance_bind' and not name.endswith('_block') and not name.startswith(('scientific_', 'realization_')):
                    dependencies.extend(_refs(attribute))
            answer = any(may_affect_frontier(item, frame, active) for item in dependencies)
        active.remove(key)
        possible[key] = answer
        return answer

    def may_deliver(value, frame, active=None):
        """Track Use reachability separately from fallible control dependencies."""
        key = (value.id, frame)
        if key in delivery_possible:
            return delivery_possible[key]
        active = set() if active is None else active
        if key in active:
            return True
        active.add(key)
        if matches(value):
            answer = True
        elif value.id in frame.bindings:
            target, bound_frame, _ = frame.bindings[value.id]
            answer = may_deliver(target, bound_frame, active)
        else:
            dependencies = list(value.inputs)
            for name, attribute in value.attrs.items():
                if value.op != 'provenance_bind' and not name.endswith('_block') and not name.startswith(('scientific_', 'realization_')):
                    dependencies.extend(_refs(attribute))
            answer = any(may_deliver(item, frame, active) for item in dependencies)
        active.remove(key)
        delivery_possible[key] = answer
        return answer

    def bounded(rows):
        result = tuple(dict.fromkeys(rows))
        if len(result) > max_alternatives:
            raise ValueError('realization delivery analysis exceeds its alternative budget')
        return result

    def union(left, right):
        rows = []
        for choices_a, refs_a in left:
            for choices_b, refs_b in right:
                choices = choices_a | choices_b
                if any((key, not arm) in choices for key, arm in choices):
                    continue
                rows.append((choices, refs_a | refs_b))
                if len(rows) > max_alternatives:
                    rows = list(bounded(rows))
        return bounded(rows)

    def alternative(condition, arm, rows, frame):
        known = None
        if condition.op == 'bool_constant':
            known = condition.attrs['value']
        elif condition.op == 'variant_is' and condition.inputs[0].op == 'variant_construct':
            known = condition.attrs['tag'] == condition.inputs[0].attrs['tag']
        elif condition.op == 'result_is_ok' and condition.inputs[0].op in {'result_ok', 'result_error'}:
            known = condition.inputs[0].op == 'result_ok'
        if known is not None:
            return rows if known is arm else ()
        key = (condition.id, frame.invocations.get(condition.id, ()))
        return union(((frozenset(((key, arm),)), frozenset()),), rows)

    def count(value):
        return value.attrs['value'] if value.op == 'int64_constant' else None

    def same_cell_frontiers(left, right):
        # A global frontier cannot represent a spatial/component partition of a Use.
        # In particular, union(left,right) would hide an undelivered selected arm.
        if set(left) != set(right):
            raise ValueError('per-cell selection has different delivery frontiers; '
                             'realize the final selected value explicitly')
        return left

    def expressions(value, path, frame):
        nodes = value.attrs['expression_nodes']
        cached = {}
        def expression(index):
            if index in cached:
                return cached[index]
            node = nodes[index]
            op = node[0]
            if op == 'input':
                rows = visit(value.inputs[node[1]], path, frame)
            elif op in {'literal', 'coefficient'}:
                rows = empty
            elif op == 'where':
                # Its predicate is a control dependency, never an inventory delivery.
                rows = same_cell_frontiers(expression(node[2]), expression(node[3]))
            else:
                if op in {'add', 'sub', 'mul', 'div', 'pow', 'minimum', 'maximum', 'and', 'or'}:
                    children = node[1:]
                elif op in {'neg', 'sqrt', 'exp', 'sin', 'cos', 'log', 'expm1', 'abs', 'sign', 'not', 'rounded'}:
                    children = node[1:]
                elif op == 'compare':
                    children = node[2:]
                elif op == 'finite_linear_v1':
                    children = node[5]
                elif op == 'finite_projection_v1':
                    children = (node[1],)
                else:
                    raise ValueError('no delivery rule for scalar expression '+str(op))
                rows = empty
                for child in children:
                    rows = union(rows, expression(child))
            cached[index] = rows
            return rows
        result = empty
        for index in value.attrs['expressions']:
            result = union(result, expression(index))
        return result

    def child_frame(loop, ordinal, previous, frame):
        bindings, invocations = dict(frame.bindings), dict(frame.invocations)
        bindings[loop.attrs['body_argument'].id] = previous
        invocation = frame.invocations.get(loop.id, ()) + ((loop.id, ordinal),)
        for node in _region_nodes(loop.attrs['body_block']):
            invocations[node.id] = invocation
        return _Frame(bindings, invocations)

    def loop_without_delivery(value, path, frame):
        """Summarize possible success, without certifying a numerical solve.

        No Use is reachable here, so unrolling cannot discover a delivery or
        multiplicity. Numerical stop/count predicates remain opaque: retaining a
        possible empty frontier is conservative for delivery checking. Explicit
        terminal failures are still eliminated; this is not a convergence proof.
        """
        is_fold = value.op == 'structured_fold'
        capacity = count(value.inputs[2] if is_fold else value.inputs[1])
        if not is_fold:
            if path and path[0] == 'err':
                # A handled failure is data, not a delivery from the success arm.
                return empty
            if path and path[0] == 'ok':
                path = path[1:]
            if path and path[0] == 'state':
                state_path = path[1:]
            elif path and path[0] == 'iterations':
                state_path = ()
            else:
                raise ValueError('delivery through iterate requires its exact successful state projection')
        else:
            state_path = path
        initial = visit(value.inputs[0], state_path, frame)
        controls = empty
        for control in value.inputs[1:]:
            controls = union(controls, visit(control, (), frame))
        initial = union(initial, controls)
        if not initial or (capacity is not None and capacity < 0):
            return ()
        previous = (value.inputs[0], frame, ())
        if is_fold:
            literal = count(value.inputs[1])
            if literal is not None and (literal < 0 or
                    (capacity is not None and literal > capacity)):
                return ()
            if literal is None or literal == 0:
                # Zero iterations are possible for a dynamic count.
                return initial
            first = visit(value.attrs['body'], state_path,
                          child_frame(value, 1, previous, frame))
            # A positive fold cannot succeed if its first evaluation terminates.
            # Later numerical branches are not enumerated in this Use-free graph.
            return empty if union(initial, first) else ()
        bindings = dict(frame.bindings)
        bindings[value.attrs['cond_argument'].id] = previous
        condition_frame = _Frame(bindings, frame.invocations)
        condition = value.attrs['cond']
        initial = union(initial, visit(condition, (), condition_frame))
        if not initial:
            return ()
        stopped = alternative(condition, True, initial, condition_frame)
        if capacity == 0 or condition.op == 'bool_constant':
            # A constant-false stop never accepts, regardless of the budget.
            return stopped
        # The unknown predicate may accept before stepping. Do not infer either
        # convergence or exhaustion from its numerical expression or its budget.
        return empty

    def loop(value, path, frame):
        if not may_deliver(value, frame):
            return loop_without_delivery(value, path, frame)
        is_fold = value.op == 'structured_fold'
        if not is_fold:
            if path and path[0] == 'ok':
                path = path[1:]
            if path and path[0] == 'iterations':
                return empty
            if not path or path[0] != 'state':
                raise ValueError('delivery through iterate requires its exact successful state projection')
            path = path[1:]
        literal = count(value.inputs[1]) if is_fold else None
        capacity = count(value.inputs[2] if is_fold else value.inputs[1])
        limit = literal if literal is not None else capacity
        if limit is None or limit > max_iterations:
            raise ValueError('realization delivery loop exceeds its finite analysis budget')
        if limit < 0 or (literal is not None and capacity is not None and literal > capacity):
            return ()
        previous = (value.inputs[0], frame, ())
        choices, continued = [], ((frozenset(), frozenset()),)
        for iteration in range(limit + 1):
            state, state_frame, prefix = previous
            state_path = prefix + path
            if is_fold:
                if literal is None or iteration == literal:
                    choices.extend(visit(state, state_path, state_frame))
            else:
                bindings = dict(frame.bindings)
                bindings[value.attrs['cond_argument'].id] = previous
                invocations = dict(frame.invocations)
                epoch = frame.invocations.get(value.id, ()) + ((value.id, iteration, 'stop'),)
                for node in _region_nodes(value.attrs['cond_block']):
                    invocations[node.id] = epoch
                condition_frame = _Frame(bindings, invocations)
                condition = value.attrs['cond']
                accepted = alternative(condition, True,
                    visit(state, state_path, state_frame), condition_frame)
                choices.extend(union(continued, accepted))
                continued = union(continued, alternative(condition, False, empty, condition_frame))
                if not continued:
                    break
            if iteration < limit:
                body_frame = child_frame(value, iteration + 1, previous, frame)
                prefix = ('ok',) if value.attrs.get('body_is_variant_result') else ()
                previous = (value.attrs['body'], body_frame, prefix)
        return bounded(choices)

    def visit(value, path=(), frame=None):
        if frame is None:
            frame = _Frame()
        key = (value.id, path, frame)
        if key in memo:
            return memo[key]
        # Terminal failures have no successful delivery frontier. Reachability
        # pruning must not turn that absence into a successful empty frontier.
        if (value.op in {'variant_unreachable', 'result_error'} or
                (value.op == 'variant_construct' and path and path[0] != value.attrs['tag'])):
            memo[key] = ()
            return ()
        if not may_affect_frontier(value, frame):
            memo[key] = empty
            return empty
        if value.id in frame.bindings:
            target, bound_frame, prefix = frame.bindings[value.id]
            result = visit(target, prefix + path, bound_frame)
        elif matches(value):
            expanded = None if expand_match is None else expand_match(value, tuple(path))
            if expanded is not None:
                dependency, projection = expanded
                if not isinstance(dependency, ProgramValue) or dependency.prog is not value.prog:
                    raise TypeError('delivery match expansion requires a value from the exact Program')
                projection = tuple(projection)
                if dependency is value and projection == tuple(path):
                    raise ValueError('delivery match expansion cannot reproduce itself')
                # Metadata references are followed only by this explicit internal
                # query, in the SAME native invocation frame as their owner.
                result = visit(dependency, projection, frame)
            else:
                reference = DeliveryReference(value.id, frame.invocations.get(value.id, ()), tuple(path) if with_projection else ())
                result = ((frozenset(), frozenset((reference,))),)
        elif value.op == 'product_get':
            result = visit(value.inputs[0], (value.attrs['field'],) + path, frame)
        elif value.op in {'continuation_publish', 'continuation_accept',
                          'continuation_accepted_candidate', 'continuation_candidate_value'}:
            # Assessment/checks are not delivered numerical values.
            result = visit(value.inputs[0], path, frame)
        elif value.op == 'continuation_freeze':
            result = visit(value.inputs[2], path, frame)
        elif value.op in {'continuation_transform', 'bundle_remap_validate'}:
            # Only the transformed/remapped candidate is a retained value. The
            # old Bundle is a validation witness, not a second numerical delivery.
            result = visit(value.inputs[1], path, frame)
        elif value.op == 'bundle_validate':
            result = visit(value.inputs[0], path, frame)
        elif value.op == 'continuation_child_base':
            result = visit(value.inputs[0], ('continuation',) + path, frame)
        elif value.op == 'continuation_read':
            result = visit(value.inputs[0], path, frame)
        elif value.op == 'continuation_open':
            # The previous accepted image is a boundary, not a new delivery.
            result = empty
        elif value.op in {'continuation_proposal', 'continuation_candidate_base',
                          'continuation_attest', 'continuation_assessment_checks',
                          'continuation_assessment_ok'}:
            result = empty
        elif value.op == 'continuation_consume_increment':
            result = visit(value.inputs[0], ('increment',) + path, frame)
        elif value.op in {'finite_map', 'distributed_map'}:
            # A pointwise map transports the nearest Array-valued contribution
            # through its element expression. It cannot borrow an unused capture
            # or its domain/cardinality as evidence of a numerical delivery.
            local_owned = any(matches(node) for node in _region_nodes(value.attrs['body_block']))
            if local_owned:
                domain = value.inputs[0]
                length = (count(domain.inputs[0]) if domain.op == 'finite_range_set' else None)
                if length is None:
                    raise ValueError('owned Use inside dynamic map must realize the complete Array value')
                if length > max_iterations:
                    raise ValueError('realization delivery map exceeds its finite analysis budget')
                result = empty
                for iteration in range(length):
                    invocations = dict(frame.invocations)
                    epoch = frame.invocations.get(value.id, ()) + ((value.id, iteration, 'map'),)
                    for node in _region_nodes(value.attrs['body_block']):
                        invocations[node.id] = epoch
                    result = union(result, visit(value.attrs['body'], path,
                                                  _Frame(frame.bindings, invocations)))
            else:
                result = visit(value.attrs['body'], path, frame)
        elif value.op in {'finite_domain', 'finite_size', 'finite_same_domain',
                          'finite_index_value', 'finite_index', 'finite_index_argument',
                          'distributed_value_domain', 'distributed_size',
                          'distributed_gid', 'distributed_index', 'distributed_index_argument',
                          'distributed_offset_argument', 'distributed_rank'}:
            # Geometry, cardinality and predicates cannot deliver an inventory.
            result = empty
        elif path and value.op == 'product':
            fields = tuple(value.attrs['fields'])
            result = visit(value.inputs[fields.index(path[0])], path[1:], frame)
        elif value.op in {'branch', 'result_match'}:
            condition = value.inputs[0] if value.op == 'branch' else value.inputs[1]
            result = bounded((
                *alternative(condition, True, visit(value.attrs['true_result'], path, frame), frame),
                *alternative(condition, False, visit(value.attrs['false_result'], path, frame), frame),
            ))
        elif value.op == 'where':
            result = same_cell_frontiers(visit(value.inputs[1], path, frame),
                                         visit(value.inputs[2], path, frame))
        elif value.op in {'pointwise_expression', 'scalar_expression'}:
            result = expressions(value, path, frame)
        elif value.op in {'structured_fold', 'structured_iterate'}:
            result = loop(value, path, frame)
        elif value.op == 'variant_require_success':
            result = visit(value.inputs[0], ('ok',) + path, frame)
        elif value.op == 'variant_payload':
            result = visit(value.inputs[0], (value.attrs['tag'],) + path, frame)
        elif value.op in {'result_payload', 'iteration_payload'}:
            # Legacy error payloads contain status code/origin/iteration count only.
            # Traversing the result's success value here would borrow a delivery
            # from a mutually exclusive, possibly unexecuted branch.
            result = (empty if value.attrs['variant'] == 'err' else
                      visit(value.inputs[0], path, frame))
        elif value.op in {'result_is_ok', 'iteration_is_ok', 'variant_is'}:
            result = empty
        elif value.op == 'variant_construct':
            result = (() if path and path[0] != value.attrs['tag'] else
                      visit(value.inputs[0], path[1:] if path else (), frame))
        elif value.op in {'provenance_bind', 'acceptance_guard', 'result_ok', 'result_success',
                          'iteration_success', 'region_argument'}:
            result = visit(value.inputs[0], path, frame)
        elif value.op == 'capture_result':
            if path and path[0] == 'err':
                raise ValueError('delivery from captured failure payload requires an explicit failure frontier')
            body_path = path
            if value.vtype == 'variant' and path and path[0] == 'ok':
                body_path = path if value.attrs.get('body_is_variant_result') else path[1:]
            result = visit(value.attrs['body'], body_path, frame)
        else:
            if any(name.endswith('_block') and attribute for name, attribute in value.attrs.items()):
                raise ValueError('no delivery rule for native region '+value.op)
            result = empty
            support_only = value.attrs.get('support_only_inputs', ())
            for index, dependency in enumerate(value.inputs):
                if index not in support_only:
                    result = union(result, visit(dependency, path, frame))
            for name, attribute in value.attrs.items():
                if (not name.endswith('_block') and
                        not name.startswith(('scientific_', 'realization_'))):
                    for dependency in _refs(attribute):
                        result = union(result, visit(dependency, path, frame))
        memo[key] = result
        return result

    rows = visit(root, tuple(path))
    return rows if with_choices else tuple(refs for _, refs in rows)
