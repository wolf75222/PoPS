"""stageproviderread@1: actual issued provider observations for one FE evaluation.

This proves stage read authority, not a forward-Euler invariant theorem. The
existing spatial guards and exact convex coefficient proof remain mandatory.
"""
from pops.model import FieldSpace, OperatorHandle, StateSpace
from pops.model.provider_pack import ComponentKey
from pops.time.field_context import FieldContext

CONTRACT = "pops.stageproviderread@1"


def _same_point(left, right):
    from pops.time import StagePoint
    if type(left) is StagePoint:
        left = left.time
    if type(right) is StagePoint:
        right = right.time
    return left == right


def _alias(value):
    from .program_accepted_ssp import successful_guard_input
    while value.op == "acceptance_guard":
        value = successful_guard_input(value)
    return value


def _rate_surface(authority, observation, rate):
    owner = authority.owner_for_block(rate.block)
    module = authority.source_module_for_owner(owner)
    handle = rate.attrs["operator_handle"]
    if not isinstance(handle, OperatorHandle):
        raise ValueError("stage provider read needs an exact typed rate handle")
    declaration = handle.declaration_ref if handle.is_instance else handle
    if not isinstance(declaration, OperatorHandle) or declaration.owner_path.canonical() != owner \
            or handle.is_instance and handle.block_ref != rate.block:
        raise ValueError("stage provider read changes its source rate owner")
    registry = module.operator_registry()
    name = registry.target_for_handle(declaration.name)
    operator = registry.get(name)
    if registry.owner_path.canonical() != owner or handle.registered_operator_name != name \
            or declaration.registered_operator_name != name or handle.kind != operator.kind \
            or handle.signature != operator.signature:
        raise ValueError("stage provider read changes its actual rate target/kind/signature")
    states = tuple(x for x in operator.signature.inputs if isinstance(x, StateSpace))
    fields = tuple(x for x in operator.signature.inputs if isinstance(x, FieldSpace))
    if states != (rate.inputs[0].space,) or fields != (observation.space,) \
            or len(operator.signature.inputs) != 2:
        raise ValueError("stage provider read changes the formal State/FieldSpace")


def _authenticate_linear_equations(plan, solve, program):
    """Compare executed coefficient/load/apply payloads with the registered equations.

    ResolvedProgramFieldPlan already owns the physical problem and solver graph;
    expression re-encoding additionally refuses stale payloads under unchanged IDs.
    """
    from .program_field_plan import _nodes, _reachable, _canonical
    from pops.fields._program_problem import _physical_coefficients, _physical_boundary, validate_field_apply
    from pops.fields._program_expression import encode_field_expression, field_expression_dependencies
    from pops.identity.scalar import scalar_literal
    from pops.time.references import canonical_handle
    problem = plan.operator
    diffusion, reaction = _physical_coefficients(problem)
    expected_boundary = _physical_boundary(problem)
    operations = [v for v in _reachable(solve, _nodes(program)) if v.op in
                  ("field_problem_load", "field_problem_coefficients", "field_problem_apply")]
    for node in operations:
        if node.attrs.get("physical_boundary") != expected_boundary:
            raise ValueError("stage provider operation changed its registered physical boundary")
        if node.op == "field_problem_apply":
            validate_field_apply(node)
            if _canonical(node.attrs["reaction"]) != _canonical(tuple(scalar_literal(x).to_data() for x in reaction)):
                raise ValueError("stage provider apply changed its registered reaction")
            continue
        states = tuple(node.inputs)
        expected = tuple(_canonical(x.canonical_identity()) for x in problem.dependencies())
        actual = tuple(_canonical(canonical_handle(x.state_ref).canonical_identity()) for x in states)
        if actual != expected:
            raise ValueError("stage provider equation changed its complete State binding")
        expressions = tuple(encode_field_expression(eq.rhs, states) for eq in problem.equations) \
            if node.op == "field_problem_load" else tuple(encode_field_expression(eq, states) for eq in diffusion)
        if _canonical(expressions) != _canonical(node.attrs["expressions"]) \
                or _canonical(field_expression_dependencies(expressions, states)) != _canonical(node.attrs["field_dependencies"]):
            raise ValueError("stage provider equation payload changed under its physical identity")


def prove_stage_provider_read(program, observation, rate, reads, authority, issued,
                              frozen_read):
    from .program_models import ProgramModelGraph
    from .program_field_publication import publication_target_space
    from pops.fields._program_publication import (
        validate_field_publication, publication_states, publication_observation,
    )
    from pops.fields._observation_contract import validate_field_observation
    from .program_field_plan import ResolvedProgramFieldPlan, _canonical
    from pops.time.references import canonical_handle
    from .program_accepted_ssp import _independent_effect

    if observation.op != "field_publication":
        return False  # Other reads retain their separate existing closure obligations.
    if type(authority) is not ProgramModelGraph:
        raise ValueError("stage provider read requires its actual resolved ProgramModelGraph")
    if set(observation.attrs) != {"bindings", "consumer_states", "field_problem_identity", "field"}:
        raise ValueError("stage provider read has no closed publication operation contract")
    if issued.get(observation.id) is not observation or issued.get(rate.id) is not rate \
            or not rate.inputs or issued.get(rate.inputs[0].id) is not rate.inputs[0]:
        raise ValueError("stage provider read has detached issued SSA authority")
    if observation.clock != rate.clock or observation.region != rate.region \
            or not _same_point(observation.point, rate.point):
        raise ValueError("stage provider read changes clock/point/region")
    if type(observation.field_context) is not FieldContext or rate.field_context != observation.field_context:
        raise ValueError("stage provider read changes its actual FieldContext")
    _rate_surface(authority, observation, rate)
    rows = validate_field_publication(observation,
        target_space=lambda target: publication_target_space(authority, target))
    states = publication_states(observation,
        target_space=lambda target: publication_target_space(authority, target))
    state = rate.inputs[0]
    if rate.block not in states or _alias(states[rate.block]) is not _alias(state) \
            or states[rate.block].state_ref != state.state_ref:
        raise ValueError("stage provider read does not consume its exact RHS State snapshot")
    pack = authority.resolved_provider_pack_for_block(rate.block)
    # The already-reauthenticated issued plan holds the actual physical output claims.
    _, source_plan, _ = authority._resolved_provider_sources[rate.block.local_id]
    claims = source_plan.provider_evidence.get("program_field_publications", ())
    selected = {}
    for row, source in zip(rows, observation.inputs[:len(rows)], strict=True):
        target = row["target"]
        if target.block_ref != rate.block:
            continue
        key = ComponentKey(str(target.declaration_ref.owner_path.canonical()), "field",
                           target.declaration_ref.local_id, row["component"])
        selected[row["component"]] = key
        entry = pack[key]
        if entry.producer != observation.attrs["field_problem_identity"] or not entry.available:
            raise ValueError("stage provider read differs from its resolved/emitted producer")
        observed = publication_observation(source)
        expected = {"key": key.to_data(), "target": target._resolved().canonical_identity(),
                    "producer": observation.attrs["field_problem_identity"],
                    "unknown": observed.attrs["field_unknown"], "observation": source.op,
                    "source_component": row["source_component"]}
        if not any(_canonical(claim) == _canonical(expected) for claim in claims):
            raise ValueError("stage provider read changes its issued physical output claim")
        width, component, solve = validate_field_observation(observed)
        plan = authority.resolved_program_field_plan(observation.attrs["field"], program)
        plan.validate_program(program)
        if plan.operator.identity.token != observation.attrs["field_problem_identity"] \
                or solve.id not in plan.solve_node_ids:
            raise ValueError("stage provider solve differs from its resolved physical Field plan")
        _authenticate_linear_equations(plan, solve, program)
    if not reads <= set(selected):
        raise ValueError("stage provider read lacks complete constitutive component coverage")
    # Select independently at every RHS: a later stage may legitimately replace
    # storage after an earlier RHS, but an intervening writer invalidates its token.
    order = {v.id: i for i, v in enumerate(program._values)}
    if observation.id not in order or rate.id not in order or order[observation.id] >= order[rate.id]:
        raise ValueError("stage provider publication is not executed before its RHS")
    for value in program._values[order[observation.id]+1:order[rate.id]]:
        if value.op != "field_publication":
            continue
        mutations = validate_field_publication(value,
            target_space=lambda target: publication_target_space(authority, target))
        if any(row["target"].block_ref == rate.block and row["component"] in reads \
               and row["target"].declaration_ref.local_id == observation.space.name for row in mutations):
            raise ValueError("stage provider read was replaced before its RHS evaluation")
    visiting = set()
    def closure(value, bound=None):
        bound = {} if bound is None else bound
        if issued.get(value.id) is not value:
            raise ValueError("stage provider closure has detached SSA authority")
        if value.id in bound:
            if bound[value.id] is not value:
                raise ValueError("stage provider callable binder is detached")
            return True
        if value.vtype == "state":
            if _alias(value) is _alias(state):
                return value.block == rate.block and value.clock == state.clock \
                    and value.region == state.region and _same_point(value.point, state.point) \
                    and canonical_handle(value.state_ref) == canonical_handle(state.state_ref)
            if value.block == rate.block:
                return False
            return value.clock == rate.clock and frozen_read(value, reads, rate.block)
        if value.id in visiting:
            raise ValueError("stage provider mathematical closure contains a cycle")
        visiting.add(value.id)
        try:
            if value.op == "matrix_free_operator":
                left, right = value.attrs["apply_in"], value.attrs["apply_out"]
                body = value.attrs["apply_block"]
                if left.op != "apply_in" or right.op != "apply_out" or left is right \
                        or not any(v is left for v in body) or not any(v is right for v in body):
                    raise ValueError("stage provider callable lost its actual binders")
                local = {left.id:left, right.id:right}
                return all(closure(v, local) for v in body)
            if value.op == "field_problem_apply":
                if len(value.inputs) != 3 or value.inputs[0].id not in bound or value.inputs[1].id not in bound \
                        or value.inputs[0].op != "apply_out" or value.inputs[1].op != "apply_in":
                    return False
                return closure(value.inputs[2], bound)
            return _independent_effect(program, value) and all(closure(v, bound) for v in value.inputs)
        finally:
            visiting.remove(value.id)
    return all(closure(source) for source in observation.inputs[:len(rows)])
