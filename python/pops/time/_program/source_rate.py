"""Lower an authenticated source-only balance into prepared source/algebra operations."""
from __future__ import annotations


def lower_source_rate(program, operator, arguments, name):
    from pops.model.handles import OperatorHandle
    from pops.time.values import _Coeff
    from .value_validation import rate_space_for

    from pops._ir.balance import source_balance_supported

    view = operator.lowering.get("physical_balance")
    if not source_balance_supported(view):
        return None
    values, coefficients = [], []
    for occurrence in view.occurrences:
        handle = occurrence.payload
        if not isinstance(handle, OperatorHandle):
            registry = program._operator_registries[handle.owner_path]
            registered = registry.target_for_handle(handle.local_id)
            declaration = registry.get(registered)
            handle = OperatorHandle(handle.local_id, kind=declaration.kind,
                                    owner=handle.owner_path, signature=declaration.signature,
                                    registered_operator_name=registered)
        inputs = []
        for space in handle.signature.inputs:
            matches = [value for value in arguments if value.space == space]
            if len(matches) != 1:
                raise ValueError("source rate requires one exact input for every source space")
            inputs.append(matches[0])
        values.append(program._call(handle, *inputs))
        coefficients.append(_Coeff({0: occurrence.coefficient}).to_polynomial())
    if not values:
        # An empty selected balance is the zero rate in its original state space.
        # Retain its exact physical view rather than manufacturing a source occurrence.
        values.append(arguments[0])
        coefficients.append(_Coeff({0: 0}).to_polynomial())
    return program._new("rhs", "linear_combine", tuple(values),
                        {"coeffs": coefficients, "physical_balance": view},
                        name or operator.name, arguments[0].block,
                        space=rate_space_for(arguments[0].space), point=arguments[0].point)
