"""Capture one principal spatial evaluation without inventing numerical physics."""


def lower_principal_rate(program, operator, arguments, name, sampling=()):
    from .value_validation import require_owned
    argument_ids = {value.id for value in arguments}
    inputs = tuple(arguments) + tuple(value for value in sampling if value.id not in argument_ids)
    for value in inputs:
        require_owned(program, value, "principal sampling")
        if value.vtype != "state":
            raise TypeError("principal finite-volume sampling requires state values")
    if len({value.point for value in inputs}) != 1:
        raise ValueError("principal inputs must share one exact evaluation point")
    if len({value.clock for value in inputs}) != 1:
        raise ValueError("principal inputs must share one exact clock")
    target_space = operator.signature.output.base_space
    targets = [value for value in arguments if value.space == target_space]
    if len(targets) != 1:
        raise ValueError("principal rate needs one exact target state input")
    target = targets[0]
    return program._new("rhs", "principal_rate", inputs,
        {"physical_balance": operator.lowering.get("physical_balance"),
         "principal_region": program._current_region(),
         "operator": operator.name,
         "target_input": next(i for i, value in enumerate(inputs) if value.id == target.id)},
        name or operator.name, target.block, space=operator.signature.output,
        point=target.point, state_ref=target.state_ref)


def call_with_bindings(handle, target, bindings, **kwargs):
    from collections.abc import Mapping
    from pops.model import Handle
    from pops.time.values import ProgramValue
    if not isinstance(bindings, Mapping):
        raise TypeError("rate bindings require a mapping of exact quantities to Program values")
    if not isinstance(target, ProgramValue) or target.vtype != "state":
        raise TypeError("rate bindings require a target Program state")
    declaration = handle.declaration_ref or handle
    model = declaration._model_ref()
    if model is None:
        raise RuntimeError("principal rate declaring Model is unavailable")
    supplied = {}
    for quantity, value in bindings.items():
        if not isinstance(quantity, Handle) or quantity.kind != "state":
            raise TypeError("rate bindings keys must be typed state quantities")
        physical = quantity.declaration_ref or quantity
        model.declaration_index().authenticate(physical)
        if not isinstance(value, ProgramValue) or value.vtype != "state" or value.space != physical.space:
            raise ValueError("rate binding value does not carry its exact physical StateSpace")
        if quantity.is_instance and quantity.block_ref != value.block:
            raise ValueError("rate binding value belongs to another block instance")
        if physical.space in supplied:
            raise ValueError("rate bindings repeat a physical StateSpace")
        supplied[physical.space] = value
    if supplied.get(target.space) is not target:
        raise ValueError("rate bindings must explicitly contain the exact target Program value")
    try:
        inputs = tuple(supplied[space] for space in handle.signature.inputs)
    except KeyError:
        raise ValueError("rate bindings omit a required physical input") from None
    return target.prog._call(handle, *inputs, _sampling=tuple(supplied.values()), **kwargs)
