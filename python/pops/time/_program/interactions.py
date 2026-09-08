"""Lower balance occurrences through one application per exact Program context."""
from __future__ import annotations


def lower_joint_balance(program, operator, arguments, name):
    from pops._ir.visitors import _key
    from pops.model.handles import OperatorHandle
    from pops.physics.interactions import joint_balance_supported
    from pops.time.values import _Coeff
    view = operator.lowering["physical_balance"]
    if not joint_balance_supported(view):
        raise ValueError("joint balance requires its supported numerical construction")
    cache = getattr(program, "_joint_application_values", None)
    if cache is None:
        cache = {}
        program._joint_application_values = cache

    def inputs_for(signature):
        values = []
        for space in signature.inputs:
            matches = [value for value in arguments if value.space == space]
            if len(matches) != 1:
                raise ValueError("joint balance requires one exact state/sampling input per signature")
            values.append(matches[0])
        return tuple(values)

    terms = []
    for occurrence in view.occurrences:
        if occurrence.kind == "projection":
            application = occurrence.payload.application
            inputs = inputs_for(application.operator.signature)
            key = (_key(application), tuple(value.id for value in inputs), program._current_region())
            result = cache.get(key)
            if result is None:
                result = program._call(application.operator, *inputs)
                # Preserve the instantiated body, including substitutions and its context.
                coupled = program._canonical_value(next(iter(result.items()))[1].inputs[0])
                attrs = dict(coupled.attrs)
                attrs["joint_application"] = application
                program._replace_value(coupled, attrs=attrs)
                cache[key] = result
            target, = [value for value in arguments if value.space == view.target.space]
            value = result[target.block]
            coefficient = occurrence.coefficient
        else:
            handle = occurrence.payload
            registry = program._operator_registries[handle.owner_path]
            # Public scientific handles carry their exact executable operator binding.
            if not isinstance(handle, OperatorHandle):
                registered = registry.target_for_handle(handle.local_id)
                declaration = registry.get(registered)
                handle = OperatorHandle(handle.local_id, kind=declaration.kind,
                    owner=handle.owner_path, signature=declaration.signature,
                    registered_operator_name=registered)
            value = program._call(handle, *inputs_for(handle.signature))
            coefficient = -occurrence.coefficient if occurrence.kind == "flux" else occurrence.coefficient
        terms.append(_Coeff({0: coefficient}) * value)
    combined = terms[0]
    for term in terms[1:]:
        combined = combined + term
    return program.value(name or operator.name, combined)
