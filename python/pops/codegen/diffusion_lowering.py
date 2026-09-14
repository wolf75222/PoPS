"""Checked target adapter for retained constitutive laws, never a new physics authority."""
from types import MappingProxyType


def prepare_diffusion_carrier(emitter, module, *, quantity_handles=()):
    from pops.model.state_symbols import rebind_state_symbols
    from pops._ir.lowering import diff
    from pops._ir.expr import Var
    laws = [op.lowering.get("diffusive_law") for op in module.operator_registry()
            if op.lowering.get("diffusive_law") is not None]
    if not laws:
        return
    if len(module.state_spaces()) != 1:
        raise ValueError("diffusion requires an explicitly selected state route; multiple states require a joint state binding")
    state = next(iter(module.state_spaces().values()))
    impl = getattr(emitter, "_m", emitter)
    native = {}
    axes = laws[0].axes
    for law in laws:
        if law.axes != axes:
            raise ValueError("diffusive physical frames differ inside one state route")
        roots = rebind_state_symbols(law.expressions, state, module.state_spaces().values(),
                                     module=module, quantity_handles=quantity_handles)
        count = len(law.variables)
        variables = roots[:count]
        derivatives = tuple(diff(variable, Var(component, "cons"), impl.prim_defs)
                            for variable, component in zip(variables, state.components, strict=True))
        tensors = tuple(tuple(tuple(roots[count+c*law.dimension**2+i*law.dimension+j]
                                  for j in range(law.dimension)) for i in range(law.dimension))
                        for c in range(count))
        jacobian = tuple(tuple(diff(variable, Var(component, "cons"), impl.prim_defs)
                              for component in state.components) for variable in variables)
        name = next(op.name for op in module.operator_registry()
                    if op.lowering.get("diffusive_law") is law)
        native[name] = MappingProxyType({"physical": law, "variables": variables, "tensors": tensors, "jacobian": jacobian,
            "variable": variables[0],
            "diagonals": tuple(tuple(roots[count+c*law.dimension**2+i*law.dimension+i]
                                     for i in range(law.dimension)) for c in range(count)),
            "diagonal": tuple(roots[count+i*law.dimension+i] for i in range(law.dimension)),
            "derivatives": derivatives, "derivative": derivatives[0]})
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
