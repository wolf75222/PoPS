"""Capture the authenticated native providers of a selected physical rate."""

from pops._ir.native_call import native_functions


def physical_rate_native_functions(op, args):
    roots = [op.body]
    view = op.lowering.get("physical_balance")
    if view is None or not args:
        return native_functions(roots)
    state = args[0]
    block_model = state.block._instance_registry.spec(state.block.local_id)["model"]
    module = getattr(block_model, "module", block_model)
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
    return native_functions(roots)
