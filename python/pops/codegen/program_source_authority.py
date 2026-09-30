"""Authenticate the physical source inputs consumed by a detached emitter.

The Module is the authority; mutable lowering caches cannot replace its source,
reachable constitutive recipes, state coordinates, or runtime parameter layout.
Unrelated compiler caches and artifacts do not participate in this comparison.
"""
from pops._ir.expr import Var
from pops._ir.values import RuntimeParamRef
from pops._ir.visitors import _children
from pops.model.hash_data import canonical_hash_data


def _source_closure(impl, name):
    body = impl._source_terms[name]
    pending, seen, recipes, parameters = list(body), set(), {}, []
    while pending:
        node = pending.pop()
        if id(node) in seen:
            continue
        seen.add(id(node))
        if isinstance(node, Var) and node.kind == "prim":
            if node.name in impl.prim_defs and node.name not in recipes:
                recipes[node.name] = impl.prim_defs[node.name]
                pending.append(recipes[node.name])
        if isinstance(node, RuntimeParamRef):
            parameters.append(node)
        pending.extend(_children(node))
    return body, recipes, parameters


def _parameter_data(nodes, module):
    result = []
    registry = module.param_registry()
    for node in nodes:
        if node.handle is None or registry.handle(node.handle) is not node.handle:
            raise ValueError("physical global source runtime parameter authority changed")
        # The registry-issued declaration handle may still be authoring-owned in
        # this internal lowering. Compare its authenticated identity directly;
        # it is neither serialized into the proof nor retained after this check.
        result.append((node.name, canonical_hash_data(node.literal), node.dtype, node.handle))
    return result


def require_lowered_source_authority(model, impl, name, evaluation):
    """Validate before provider planning, parameter slot assignment, or C++ emission."""
    from pops.codegen.module_lowering import _module_to_model
    from pops.model.global_quantity import global_references
    from pops.time._program.global_source_plan import require_detached_source_global

    module = getattr(model, "module", None)
    if module is None:
        raise ValueError("detached physical global source requires Module/body authority")
    require_detached_source_global(evaluation, module=module)
    expected = _module_to_model(
        module, state_space=evaluation.state_ref.declaration_ref.local_id)._m
    if tuple(impl.cons_names) != tuple(expected.cons_names):
        raise ValueError("physical global source component binding authority changed")
    body, recipes, parameters = _source_closure(impl, name)
    expected_body, expected_recipes, expected_parameters = _source_closure(expected, name)
    if canonical_hash_data(body) != canonical_hash_data(expected_body):
        raise ValueError("physical global source lowered body changed from its Module authority")
    recipe_order = tuple(key for key in impl.prim_defs if key in recipes)
    expected_order = tuple(key for key in expected.prim_defs if key in expected_recipes)
    if (recipe_order != expected_order
            or canonical_hash_data(recipes) != canonical_hash_data(expected_recipes)):
        raise ValueError("physical global source primitive body authority changed")
    # This branch controls inlining in the existing emitter. Authenticate it too,
    # including a recipe outside the selected source that could alter that branch.
    if bool(global_references(tuple(impl.prim_defs.values()))) != bool(
            global_references(tuple(expected.prim_defs.values()))):
        raise ValueError("physical global source primitive expansion authority changed")
    if _parameter_data(parameters, module) != _parameter_data(expected_parameters, module):
        raise ValueError("physical global source runtime parameter recipe authority changed")
    # Slots are model-wide and sorted by name. A parameter introduced in another
    # lowering expression can silently shift the source's params.get(index).
    if _parameter_data(impl.runtime_param_nodes(), module) != _parameter_data(
            expected.runtime_param_nodes(), module):
        raise ValueError("physical global source runtime parameter slot authority changed")
