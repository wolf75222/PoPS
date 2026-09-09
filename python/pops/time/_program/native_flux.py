"""Capture the authenticated native providers of a selected physical rate."""

from pops._ir.native_call import native_functions
from pops._ir.primitive_expansion import expand_primitive_recipes


def physical_rate_native_functions(op, args):
    roots = [op.body]
    view = op.lowering.get("physical_balance")
    if not args or (view is None and op.kind != "local_rate"):
        return native_functions(roots)
    state = args[0]
    block_model = state.block._instance_registry.spec(state.block.local_id)["model"]
    module = getattr(block_model, "module", block_model)
    if view is None:
        # Module.rate_operator retains exact dependency handles independently of
        # the blackboard physical-balance view. Follow that same public contract.
        try:
            contract = module.rate_contract(module.operator_handle(op.name))
        except ValueError:
            # A direct local_rate expression owns its body instead of a composed
            # rate contract. Its native providers were already included above.
            return native_functions(expand_primitive_recipes(roots, module.primitive_recipes()))
        for dependency in (*tuple(contract["flux"] or ()), *contract["sources"]):
            if dependency.owner_path != module.owner_path:
                raise ValueError("native rate dependency belongs to a different Module")
            roots.append(module.operator_registry().get(dependency.registered_operator_name).body)
        return native_functions(expand_primitive_recipes(roots, module.primitive_recipes()))
    for occurrence in view.occurrences:
        if occurrence.kind not in {"flux", "source"}:
            continue
        payload = occurrence.payload
        registry_name = getattr(payload, "reg_name", None)
        if registry_name is None:
            continue
        if payload.owner_path != module.owner_path:
            raise ValueError("native physical rate provider belongs to a different Module")
        declaration = module.operator_registry().get(registry_name)
        roots.append(declaration.body)
    return native_functions(expand_primitive_recipes(roots, module.primitive_recipes()))
