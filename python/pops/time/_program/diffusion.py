"""Program authoring for an explicitly resolved diffusive rate."""
from pops.model.balance_analysis import diffusion_balance_supported, fitted_balance_supported


def lower_diffusive_rate(program, op, args, name):
    from pops.time.field_context import require_field_read
    from pops.time._program.value_validation import rate_space_for
    view = op.lowering.get("physical_balance")
    fitted=fitted_balance_supported(view)
    if not diffusion_balance_supported(view) and not fitted:
        return None
    state = args[0]
    fields = args[1] if len(args)>1 else None
    field_context = None if fields is None else require_field_read(fields,state,"diffusive_rhs")
    # Capture the exact native closures needed by the selected physical endpoint and its
    # derivative before detaching the Program. Preparation/build staging uses this same set.
    from pops._ir.native_call import native_functions
    from pops._ir.lowering import diff
    from pops._ir.quantity import QuantityRef
    block_model = state.block._instance_registry.spec(state.block.local_id)["model"]
    module = getattr(block_model, "module", block_model)
    roots = []
    for row in view.occurrences:
        if row.kind in {"diffusion", "drift"}:
            roots.extend(row.payload.law.expressions)
            if row.kind == "diffusion" and not fitted:
                for variable, component in zip(row.payload.law.variables, state.space.components, strict=True):
                    target = QuantityRef(view.target, component, space=state.space)
                    roots.append(diff(variable, target, module.primitive_recipes()))
        elif row.kind == "source":
            roots.append(module.operator_registry().get(row.payload.reg_name).body)
    attrs = {"physical_balance": view, "fitted": fitted}
    functions = native_functions(roots)
    if functions:
        attrs["native_functions"] = functions
    return program._new("rhs", "diffusive_rhs", tuple(args), attrs, name,
                        state.block, space=rate_space_for(state.space), field_context=field_context)
