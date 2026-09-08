"""Checked target adapter for retained constitutive laws, never a new physics authority."""
from types import MappingProxyType
from pops.numerics.diffusion import diffusion_balance_supported
from pops.numerics.scharfetter_gummel import fitted_balance_supported


def prepare_diffusion_carrier(emitter, module, *, quantity_handles=()):
    from pops.model.state_symbols import rebind_state_symbols
    from pops._ir.lowering import diff
    from pops._ir.expr import Var
    laws = [op.lowering.get("diffusive_law") for op in module.operator_registry()
            if op.lowering.get("diffusive_law") is not None]
    if not laws:
        return
    if len(module.state_spaces()) != 1:
        raise ValueError("diffusion requires an explicitly selected scalar state route")
    state = next(iter(module.state_spaces().values()))
    impl = getattr(emitter, "_m", emitter)
    native = {}
    axes = laws[0].axes
    for law in laws:
        if law.axes != axes:
            raise ValueError("diffusive physical frames differ inside one state route")
        roots = rebind_state_symbols(law.expressions, state, module.state_spaces().values(),
                                     module=module, quantity_handles=quantity_handles)
        variable = roots[0]
        derivative = diff(variable, Var(state.components[0], "cons"), impl.prim_defs)
        name = next(op.name for op in module.operator_registry()
                    if op.lowering.get("diffusive_law") is law)
        native[name] = MappingProxyType({"physical": law, "variable": variable,
            "diagonal": tuple(roots[1+i*law.dimension+i] for i in range(law.dimension)),
            "derivative": derivative})
    object.__setattr__(impl, "_diffusive_laws", MappingProxyType(native))
    drift={}
    for operator in module.operator_registry():
        law=operator.lowering.get("drift_law")
        if law is None:
            continue
        roots=rebind_state_symbols(law.expressions,state,module.state_spaces().values(),
                                   module=module,quantity_handles=quantity_handles)
        drift[operator.name]=MappingProxyType({"physical":law,"density":roots[0],
                                              "mobility":roots[1],"potential":roots[2]})
    object.__setattr__(impl,"_drift_laws",MappingProxyType(drift))
    if not impl._flux:
        object.__setattr__(impl, "_program_only_storage_axes", axes)


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
    return program._new("rhs", "diffusive_rhs", tuple(args), {"physical_balance": view,"fitted":fitted}, name,
                        state.block, space=rate_space_for(state.space), field_context=field_context)
