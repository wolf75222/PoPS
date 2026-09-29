"""Lower complete authenticated principal groups into the common scalar IR."""
from types import MappingProxyType


def prepare_principal_carrier(emitter, module, numerics):
    groups = () if numerics is None else numerics.principal_groups
    impl = getattr(emitter, "_m", emitter)
    if not groups:
        if any(op.lowering.get("principal_balance") for op in module.operator_registry()):
            raise ValueError("cross-state physical transport requires complete principal sampling")
        return
    from pops._ir import Var
    from pops._ir.application import substitute_quantities
    from pops._ir.primitive_expansion import expand_primitive_recipes
    prepared = []
    for group in groups:
        bindings, offset = {}, 0
        for quantity, count in zip(group.states, group.component_counts, strict=True):
            declaration = quantity.declaration_ref or quantity
            space = module.state_spaces()[declaration.local_id]
            handle = module.state_handle(space)
            for component in range(count):
                bindings[handle, component] = Var("pops_principal_%d" % (offset + component), "cons")
            offset += count

        def native(body):
            return substitute_quantities(expand_primitive_recipes(body, module.primitive_recipes()), bindings)

        flux_rows = []
        axes = None
        for rate in group.rates:
            operator = module.operator_registry().get(rate.registered_operator_name)
            view = operator.lowering.get("physical_balance")
            if view is None or not view.accumulation.is_identity:
                raise ValueError("principal transport requires an authenticated identity accumulation")
            terms = tuple(view.occurrences)
            if len(terms) != 1 or terms[0].kind != "flux" or terms[0].coefficient != -1:
                raise ValueError("principal transport requires one complete -div(flux) row")
            body = module.operator_registry().get(terms[0].payload.reg_name).body
            if axes is None:
                axes = tuple(body)
            if tuple(body) != axes:
                raise ValueError("principal flux rows disagree on their physical ranked axes")
            flux_rows.append(native(body))
        if module._eigenvalues is None:
            raise ValueError("principal group requires its authored common wave-speed bound")
        waves = native(module._eigenvalues)
        if tuple(waves) != axes:
            raise ValueError("principal bound and physical flux axes disagree")
        method = group.methods[0]
        if method.variables.scheme != "conservative":
            raise NotImplementedError("principal primitive reconstruction requires a joint authored variable map")
        prepared.append(MappingProxyType({"group": group, "fluxes": tuple(flux_rows),
            "waves": waves, "axes": axes,
            "cpp_name": "PoPSPrincipal_" + group.identity.token.split(":")[-1][:24]}))
    object.__setattr__(impl, "_principal_groups", tuple(prepared))
    indices = tuple((node.name, index) for index, node in enumerate(impl.assign_runtime_indices()))
    object.__setattr__(impl, "_principal_groups", tuple(
        MappingProxyType({**entry, "parameter_indices": indices}) for entry in prepared))


def principal_for_value(model, value):
    impl = getattr(model, "_m", model)
    handle = value.attrs.get("operator_handle")
    if handle is None:
        raise ValueError("principal evaluation lost its physical operator authority")
    candidates = [entry for entry in getattr(impl, "_principal_groups", ())
                  if any(rate.registered_operator_name == handle.registered_operator_name
                         and rate.block_ref == value.block._resolved()
                         for rate in entry["group"].rates)]
    if len(candidates) != 1:
        raise ValueError("principal evaluation requires exactly one resolved complete group")
    return candidates[0]
