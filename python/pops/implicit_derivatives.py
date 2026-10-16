"""Finite implicit derivatives from the original equations, never from solver iteration traces.

This profile covers full-column-rank finite relations, including native-cardinality arrays. Boundary and gauge equations are
explicit closed GraphFunctions and enter the same residual before differentiation. The caller
supplies a closed tangent algorithm, both Accuracy requests, and a validity/regularity predicate.
Such a predicate is an authored mathematical obligation, not a theorem proved by the engine.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
import hashlib
import json

from pops.graph_functions import GraphFunction, graph
from pops.graph_derivatives import GraphPairings, _assemble, _get, _leaves, jvp, tangent_type
from pops.relations import Accuracy, Algorithm, Approximation, Problem, RelationErrors, solve
from pops.time._authoring import atomic_authoring
from pops.time._program.structured_control import value_signature
from pops.time._program.value_validation import require_owned
from pops.time.variant_types import Bool, Failure, Int64, Product, Result, ValueType, type_of


class ImplicitFailure(IntEnum):
    NOT_DIFFERENTIABLE = 9301
    NOT_REGULAR = 9302


ImplicitErrors = Failure(implicit=Product(code=Int64))
_LINEARIZATION_FACTORY = object()


def _closed(value, where):
    if type(value) is not GraphFunction:
        raise TypeError(where+' requires an exact closed GraphFunction')
    return value


def _members(value_type):
    if value_type.signature['vtype'] != 'product':
        raise TypeError('implicit relation values must be named finite Products')
    return {key: ValueType(child) for key, child in value_type.signature['members']}


def _ok(p, payload, errors=RelationErrors):
    return p.variant(Result(type_of(payload), errors), 'ok', payload)


def _gate(p, payload, predicate, code):
    result_type = Result(type_of(payload), ImplicitErrors)
    return p.require_success(p.branch(predicate, lambda q: q.variant(result_type, 'ok', payload),
        lambda q: q.variant(result_type, 'err', q.variant(ImplicitErrors, 'implicit',
            q.product(code=q.int64(int(code)))))))


@dataclass(frozen=True)
class ImplicitRelation:
    """Original finite equations and explicit domain of their differentiable solution map.

    `problem` retains admissibility, selected root and its Bool boundary check. Additional
    `boundary_equations` and `gauge_equations` are mathematical equations, not predicates;
    supply a closed empty-Product function when either family is absent. All three equations
    use `(unknown,data)`. `validity` has those inputs and returns
    Product(differentiable=Bool, regular=Bool, observations=Product(...)).
    """
    problem: Problem
    boundary_equations: GraphFunction
    gauge_equations: GraphFunction
    validity: GraphFunction

    def __post_init__(self):
        if type(self.problem) is not Problem:
            raise TypeError('ImplicitRelation.problem requires an exact Problem')
        signature = self.problem.residual.inputs
        if set(_members(signature)) != {'unknown', 'data'}:
            raise TypeError('implicit residual must bind exactly unknown and data')
        for name in ('boundary_equations', 'gauge_equations', 'validity'):
            function = _closed(getattr(self, name), name)
            if function.inputs != signature:
                raise TypeError(name+' must preserve the exact residual input types')
        for function in (self.problem.residual, self.boundary_equations, self.gauge_equations):
            tangent_type(function.output)  # closed finite fibre, including dynamic coordinates
        fields = _members(self.validity.output)
        if set(fields) != {'differentiable', 'regular', 'observations'} \
                or fields['differentiable'] != Bool or fields['regular'] != Bool \
                or fields['observations'].signature['vtype'] != 'product':
            raise TypeError('validity must return differentiable/regular Bool and observations Product')


@dataclass(frozen=True, init=False)
class Linearization:
    """Closed mathematical relation, complete residual and its independently checked tangent Problem."""
    relation: ImplicitRelation
    full_residual: GraphFunction
    differential: GraphFunction
    problem: Problem
    unknown_type: ValueType
    data_type: ValueType
    direction_type: ValueType

    def __init__(self, relation, full_residual, differential, problem, unknown_type,
                 data_type, direction_type, *, _factory=None):
        if _factory is not _LINEARIZATION_FACTORY:
            raise TypeError('Linearization must be produced by linearize from the original equations')
        for name, value in zip(self.__dataclass_fields__, (relation, full_residual, differential,
                problem, unknown_type, data_type, direction_type)):
            object.__setattr__(self, name, value)

    @property
    def identity(self):
        return hashlib.sha256(json.dumps(self.to_data(), sort_keys=True,
                                         separators=(',', ':')).encode()).hexdigest()

    def to_data(self):
        original = self.relation.problem
        functions = {'residual': original.residual, 'boundary_equations': self.relation.boundary_equations,
            'gauge_equations': self.relation.gauge_equations, 'validity': self.relation.validity,
            'admissible': original.admissible, 'selection': original.selection,
            'boundary_check': original.boundary, 'base_request_test': original.request_test,
            'full_residual': self.full_residual, 'differential': self.differential,
            'tangent_residual': self.problem.residual, 'tangent_request_test': self.problem.request_test}
        return {'schema_version': 1, 'profile': 'finite_full_column_rank_native_coordinates',
                'functions': {name: function.to_data() for name, function in functions.items()},
                'unknown_type': self.unknown_type.to_data(), 'data_type': self.data_type.to_data(),
                'direction_type': self.direction_type.to_data()}


def linearize(relation, *, request_test, transport_byte_capacity=None):
    """Derive the full tangent equation, including boundary and gauge derivatives.

    The tangent data type is Product(base_unknown=U, base_data=D, direction=T(D)).
    Its residual is D_U Rfull * unknown + D_D Rfull * direction. A supplied tangent algorithm
    may approximate a solution, but `relations.solve` always reevaluates this exact graph.
    `request_test` is mandatory so observation-specific Accuracy requests cannot be ignored.
    Regularity asserts full column rank of the unknown Jacobian on the declared data
    domain. All overdetermined equations remain in every residual check. Incompatible
    ambient directions fail; a VJP never invents an extension off that data domain.
    """
    if type(relation) is not ImplicitRelation:
        raise TypeError('linearize requires an explicit ImplicitRelation')
    _closed(request_test, 'tangent request_test')
    types = _members(relation.problem.residual.inputs)
    unknown, data = types['unknown'], types['data']
    full_type = Product(equations=relation.problem.residual.output,
                        boundary=relation.boundary_equations.output,
                        gauge=relation.gauge_equations.output)
    from pops._finite_differential import static_dimension
    unknown_size, residual_size = static_dimension(unknown.signature), static_dimension(full_type.signature)
    if unknown_size is not None and residual_size is not None and unknown_size > residual_size:
        raise NotImplementedError('underdetermined finite relation requires explicit selection/gauge equations')
    @graph(inputs=relation.problem.residual.inputs, output=full_type)
    def full(p, args):
        return p.product(equations=relation.problem.residual(p, unknown=args['unknown'], data=args['data']),
                         boundary=relation.boundary_equations(p, unknown=args['unknown'], data=args['data']),
                         gauge=relation.gauge_equations(p, unknown=args['unknown'], data=args['data']))
    differential = jvp(full, transport_byte_capacity=transport_byte_capacity)
    tangent_unknown, tangent_data = tangent_type(unknown), tangent_type(data)
    parameters = Product(base_unknown=unknown, base_data=data, direction=tangent_data)
    inputs = Product(unknown=tangent_unknown, data=parameters)
    @graph(inputs=inputs, output=tangent_type(full_type))
    def residual(p, args):
        d = args['data']
        return differential(p, primal=p.product(unknown=d['base_unknown'], data=d['base_data']),
                            tangent=p.product(unknown=args['unknown'], data=d['direction']))['tangent']
    @graph(inputs=inputs, output=Bool)
    def admissible(p, args):
        d = args['data']
        validity = relation.validity(p, unknown=d['base_unknown'], data=d['base_data'])
        return p.branch(validity['differentiable'], lambda q: validity['regular'],
                        lambda q: q.boolean(False))
    true = graph(inputs=inputs, output=Bool)(lambda p, args: p.boolean(True))
    request_inputs = _members(request_test.inputs)
    if set(request_inputs) != {'unknown', 'residual', 'data', 'accuracy', 'evidence'} \
            or request_inputs['unknown'] != tangent_unknown \
            or request_inputs['residual'] != tangent_type(full_type) or request_inputs['data'] != parameters:
        raise TypeError('tangent request_test must bind the exact full tangent residual and data')
    return Linearization(relation, full, differential,
                         Problem(residual, admissible, true, true, request_test),
                         tangent_unknown, parameters, tangent_data, _factory=_LINEARIZATION_FACTORY)


def _checked_base(p, linearization, base, accuracy):
    """Treat a proposed base/evidence as data, then rerun all original acceptance checks."""
    if not isinstance(accuracy, Accuracy):
        raise TypeError('base_accuracy must be an explicit Accuracy')
    require_owned(p, base, 'implicit base', vtype='product')
    fields = dict(value_signature(base)['members'])
    if not {'values', 'data_identity', 'evidence', 'termination'} <= fields.keys():
        raise TypeError('implicit base requires values, data_identity, evidence, termination')
    relation = linearization.relation
    original_request = relation.problem.request_test
    request_types = _members(original_request.inputs)
    request_types['residual'] = linearization.full_residual.output
    @graph(inputs=Product(**request_types), output=original_request.output)
    def base_request(q, args):
        return original_request(q, unknown=args['unknown'], residual=args['residual']['equations'],
                                data=args['data'], accuracy=args['accuracy'], evidence=args['evidence'])
    complete = Problem(linearization.full_residual, relation.problem.admissible,
                       relation.problem.selection, relation.problem.boundary, base_request)
    # Algorithm metadata is an explicit budget payload; it is not a captured caller SSA value.
    base_budget = p.product(evidence=base['evidence'], termination=base['termination'])
    approximation = Approximation(p, values=base['values'], evidence=base['evidence'],
                                  termination=base['termination'])
    @graph(inputs=Product(data=type_of(base['data_identity']), guess=type_of(base['values']),
                          accuracy=type_of(accuracy.parameters), budget=type_of(base_budget)),
           output=Result(type_of(approximation), RelationErrors))
    def proposed(q, args):
        return _ok(q, Approximation(q, values=args['guess'], evidence=args['budget']['evidence'],
                                    termination=args['budget']['termination']))
    checked = p.require_success(solve(p, complete, data=base['data_identity'],
        algorithm=Algorithm(proposed), guess=base['values'], accuracy=accuracy, budget=base_budget))
    validity = relation.validity(p, unknown=checked['values'], data=checked['data_identity'])
    context = p.product(base=checked, validity=validity)
    context = _gate(p, context, validity['differentiable'], ImplicitFailure.NOT_DIFFERENTIABLE)
    context = _gate(p, context, context['validity']['regular'], ImplicitFailure.NOT_REGULAR)
    from pops._finite_differential import coordinate_count
    context = _gate(p, context,
        coordinate_count(p, context['base']['values']) <= coordinate_count(p, context['base']['defects']),
        ImplicitFailure.NOT_REGULAR)
    return context


def _direction_solve(p, linearization, context, direction, algorithm, guess, accuracy, budget):
    if not linearization.direction_type.matches(direction):
        raise TypeError('implicit direction must have the exact tangent data type')
    if not linearization.unknown_type.matches(guess):
        raise TypeError('implicit tangent guess must have the exact tangent unknown type')
    base = context['base']
    data = p.product(base_unknown=base['values'], base_data=base['data_identity'], direction=direction)
    return p.require_success(solve(p, linearization.problem, data=data, algorithm=algorithm,
                                    guess=guess, accuracy=accuracy, budget=budget))


@atomic_authoring
def implicit_jvp(program, linearization, *, base, direction, algorithm, guess,
                 base_accuracy, accuracy, budget, transport_byte_capacity=None):
    """Return Result(values, base, tangent, validity, direction), retaining both solve receipts.

    The base and tangent residual tolerances are distinct explicit requests. A successful small
    tangent residual is not silently promoted into a bound on derivative error or a regularity proof.
    """
    if type(linearization) is not Linearization:
        raise TypeError('implicit_jvp requires a closed Linearization')
    from pops._finite_differential import with_transport
    @with_transport(transport_byte_capacity)
    def body(p):
        context = _checked_base(p, linearization, base, base_accuracy)
        tangent = _direction_solve(p, linearization, context, direction, algorithm, guess, accuracy, budget)
        return p.product(values=tangent['values'], base=context['base'], tangent=tangent,
                         validity=context['validity'], direction=direction)
    return program.capture(body)


@atomic_authoring
def implicit_vjp(program, linearization, *, base, covector, pairings, algorithm, guess,
                 base_accuracy, accuracy, budget, transport_byte_capacity=None):
    """Transpose the finite implicit derivative under explicitly supplied pairings.

    This profile performs one checked tangent solve per real data coordinate. It is not a claim of
    one-adjoint-solve complexity. Every column retains its original tangent residual and evidence.
    """
    if type(linearization) is not Linearization or type(pairings) is not GraphPairings:
        raise TypeError('implicit_vjp requires Linearization and explicit GraphPairings')
    ti, to = linearization.direction_type, linearization.unknown_type
    for mapping, typ in ((pairings.output_to_covector, to), (pairings.input_from_covector, ti)):
        if mapping.inputs != Product(value=typ) or mapping.output != typ:
            raise TypeError('implicit pairings must map Product(value=T) to the same T')
    from pops._finite_differential import (basis_like, check_tangent, coordinate_dot,
        coordinate_extract, coordinate_map, has_dynamic_coordinates, zero_like)
    dynamic = has_dynamic_coordinates(ti.signature) or has_dynamic_coordinates(to.signature)
    inputs, outputs = ((), ()) if dynamic else (_leaves(ti.signature), _leaves(to.signature))
    from pops._finite_differential import with_transport
    @with_transport(transport_byte_capacity)
    def body(p):
        context = _checked_base(p, linearization, base, base_accuracy)
        checked_covector = check_tangent(p, context['base']['values'], covector)
        dual = pairings.output_to_covector(p, value=checked_covector)
        dual = check_tangent(p, context['base']['values'], dual)
        if dynamic:
            data = context['base']['data_identity']
            # One independently checked zero-direction solve supplies the exact receipt type.
            # It remains part of the returned evidence, rather than a metadata-only prototype.
            zero = _direction_solve(p, linearization, context, zero_like(p, data),
                                    algorithm, guess, accuracy, budget)
            zero_receipt = p.product(solution=zero, coordinate=p.float64(0))
            def column(q, path):
                solution = _direction_solve(q, linearization, context, basis_like(q, data, path),
                                           algorithm, guess, accuracy, budget)
                return q.product(solution=solution, coordinate=coordinate_dot(q,solution['values'],dual))
            receipts = coordinate_map(p, data, column, leaf_type=type_of(zero_receipt),leaf_initial=zero_receipt)
            covariant = coordinate_extract(p, data, receipts,lambda q, receipt:receipt['coordinate'])
            representative = pairings.input_from_covector(p, value=covariant)
            representative = check_tangent(p, data, representative)
            return p.product(values=representative, base=context['base'], validity=context['validity'],
                             columns=receipts, zero_column=zero)
        pullback, receipts = {}, {}
        for i, path in enumerate(inputs):
            direction = _assemble(p, ti.signature,
                {other: p.float64(1. if other == path else 0.) for other in inputs})
            solution = _direction_solve(p, linearization, context, direction,
                                        algorithm, guess, accuracy, budget)
            receipts['column_'+str(i)] = solution
            pullback[path] = sum((_get(solution['values'], item)*_get(dual, item)
                                  for item in outputs), p.float64(0.))
        representative = pairings.input_from_covector(p, value=_assemble(p, ti.signature, pullback))
        return p.product(values=representative, base=context['base'], validity=context['validity'],
                         columns=p.product(**receipts))
    return program.capture(body)


__all__ = ['ImplicitFailure', 'ImplicitErrors', 'ImplicitRelation', 'Linearization',
           'linearize', 'implicit_jvp', 'implicit_vjp']
