"""Lower complete authenticated principal groups into the common scalar IR."""
from types import MappingProxyType


def principal_row_expressions(entry):
    """The exact formulas read in each row's parameter context, including numerics."""
    from pops._ir import _wrap
    waves = tuple(value for values in entry["waves"].values() for value in values)
    rows = []
    for row, (body, method) in enumerate(zip(entry["fluxes"], entry["group"].methods, strict=True)):
        roots = [_wrap(value) for values in body.values() for value in values]
        roots.extend(_wrap(value) for value in waves)
        if method.reconstruction.scheme == "source_stencil":
            roots.extend(method.reconstruction.expression if isinstance(method.reconstruction.expression, tuple)
                         else (method.reconstruction.expression,))
        if method.riemann.scheme == "source_face":
            roots.extend(method.riemann.expression)
        conversion = entry.get("conversion")
        if conversion is not None:
            roots.extend(conversion["forward"][row])
            roots.extend(conversion["inverse"][row])
            roots.extend(conversion["constraints"][row])
        rows.append(tuple(roots))
    return tuple(rows)


def _authenticate_numerical_captures(module, group):
    from pops._ir.values import RuntimeParamRef
    from pops._ir.visitors import _children
    from pops.numerics.reconstruction.user import authenticated_user_reconstruction
    from pops.numerics.riemann.user import authenticated_user_face
    for state, method in zip(group.states, group.methods, strict=True):
        roots = []
        if method.reconstruction.scheme == "source_stencil":
            descriptor = authenticated_user_reconstruction(method.reconstruction)
            if descriptor.capabilities.get("vector_row"):
                if descriptor.options["state"] != state or any(
                        source not in group.states for source in descriptor.options["sampling"]):
                    raise ValueError("joint reconstruction inputs differ from the exact principal group")
            roots.extend(descriptor.expression if isinstance(descriptor.expression, tuple)
                         else (descriptor.expression,))
        if method.riemann.scheme == "source_face":
            descriptor = authenticated_user_face(method.riemann)
            if descriptor.options["state"] != state:
                raise ValueError("principal face state differs from its exact row owner")
            roots.extend(descriptor.expression)
        seen = set()
        while roots:
            node = roots.pop()
            if id(node) in seen:
                continue
            seen.add(id(node))
            if isinstance(node, RuntimeParamRef):
                handle = node.handle
                if handle.is_instance:
                    if handle.block_ref._resolved() != state.block_ref._resolved():
                        raise ValueError("principal numerical capture belongs to another block instance")
                    handle = handle.declaration_ref
                try:
                    module._param_registry.handle(handle)
                except (KeyError, ValueError) as exc:
                    raise ValueError("principal numerical capture belongs to another model") from exc
            roots.extend(_children(node))


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
        _authenticate_numerical_captures(module, group)
        bindings, offset = {}, 0
        for quantity, count in zip(group.states, group.component_counts, strict=True):
            declaration = quantity.declaration_ref or quantity
            space = module.state_spaces()[declaration.local_id]
            handle = module.state_handle(space)
            for component in range(count):
                bindings[handle, component] = Var("pops_principal_%d" % (offset + component), "cons")
            offset += count

        declarations = tuple(module.state_handle((state.declaration_ref or state).space)
                             for state in group.states)
        matches = [row for row in module.primitive_coordinates()
                   if len(row.states) == len(declarations) and set(row.states) == set(declarations)]
        if len(matches) > 1:
            raise ValueError("principal reconstruction has ambiguous physical coordinate maps")
        coordinates = matches[0] if matches else None
        recipes = dict(module.primitive_recipes())
        if coordinates is not None:
            recipes.update({value.name: body for value,body in zip(coordinates.coordinates,coordinates.forward)
                            if isinstance(value,Var) and value.kind == "prim"})

        def native(body):
            return substitute_quantities(expand_primitive_recipes(body, recipes), bindings)

        flux_rows = []
        flux_names = []
        axes = None
        for rate in group.rates:
            operator = module.operator_registry().get(rate.registered_operator_name)
            view = operator.lowering.get("physical_balance")
            if view is None or not view.accumulation.is_identity:
                raise ValueError("principal transport requires an authenticated identity accumulation")
            terms = tuple(view.occurrences)
            if len(terms) != 1 or terms[0].kind != "flux" or terms[0].coefficient != -1:
                raise ValueError("principal transport requires one complete -div(flux) row")
            flux_names.append(terms[0].payload.reg_name)
            body = module.operator_registry().get(terms[0].payload.reg_name).body
            if axes is None:
                axes = tuple(body)
            if tuple(body) != axes:
                raise ValueError("principal flux rows disagree on their physical ranked axes")
            flux_rows.append(native(body))
        from pops.model.flux_waves import common_flux_waves
        common_waves = common_flux_waves(module, flux_names, global_authority=True)
        if common_waves is None:
            raise ValueError("principal group requires its authored common wave-speed bound")
        waves = native(common_waves)
        if tuple(waves) != axes:
            raise ValueError("principal bound and physical flux axes disagree")
        method = group.methods[0]
        if method.variables.scheme == "primitive" and coordinates is None:
            raise ValueError("principal primitive reconstruction requires explicit joint primitive_state coordinates and inverse")
        conversion = None
        if coordinates is not None:
            from pops._ir.quantity import QuantityRef
            source_offsets, total = {}, 0
            for state in coordinates.states:
                source_offsets[state] = total
                total += len(state.space.components)
            primitive_bindings, expression_bindings, indices = {}, {}, []
            for state,count in zip(declarations,group.component_counts):
                indices.extend(range(source_offsets[state],source_offsets[state]+count))
            for target, source in enumerate(indices):
                coordinate = coordinates.coordinates[source]
                replacement = Var("pops_principal_p%d" % target,"cons")
                if isinstance(coordinate,QuantityRef):
                    primitive_bindings[coordinate.handle,coordinate.index] = replacement
                else:
                    expression_bindings[id(coordinate)] = replacement
            def in_primitive(body):
                return substitute_quantities(body,primitive_bindings,expression_bindings=expression_bindings)
            forward = native(coordinates.forward)
            inverse = in_primitive(coordinates.inverse)
            predicates = {index: in_primitive(body) for index,body in coordinates.constraints}
            conversion = {"forward": [], "inverse": [], "constraints": []}
            for state,count in zip(declarations,group.component_counts):
                selected = range(source_offsets[state],source_offsets[state]+count)
                conversion["forward"].append(tuple(forward[i] for i in selected))
                conversion["inverse"].append(tuple(inverse[i] for i in selected))
                conversion["constraints"].append(tuple(predicates[i] for i in selected if i in predicates))
            conversion = MappingProxyType({key:tuple(rows) for key,rows in conversion.items()})
        entry = {"group": group, "fluxes": tuple(flux_rows),
            "waves": waves, "axes": axes, "conversion": conversion,
            "cpp_name": "PoPSPrincipal_" + group.identity.token.split(":")[-1][:24]}
        entry["row_expressions"] = principal_row_expressions(entry)
        prepared.append(MappingProxyType(entry))
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
