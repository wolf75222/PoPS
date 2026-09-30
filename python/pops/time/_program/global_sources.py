"""Explicit physical-source inputs backed by typed temporal captures."""
from collections.abc import Mapping


def physical_source_roots(body, state):
    from pops._ir.primitive_expansion import expand_primitive_recipes
    block_model = state.block._instance_registry.spec(state.block.local_id)["model"]
    module = getattr(block_model, "module", block_model)
    return expand_primitive_recipes(body, module.primitive_recipes())


def source_global_inputs(program, operator, state, supplied):
    from pops.model.global_quantity import GlobalQuantityHandle, global_references
    from pops.time.expressions import ProgramGlobal
    from pops.time.value_support import require_owned_value
    from .integrals import integral_units_bytes
    references = global_references(physical_source_roots(operator.body, state))
    wanted = {ref.handle.local_id: ref for ref in references}
    if supplied is None: supplied = {}
    if not isinstance(supplied, Mapping):
        raise TypeError("source global_inputs must be a mapping of physical ports to captures")
    rows, captures = [], []
    seen = set()
    for port, expression in supplied.items():
        if not isinstance(port, GlobalQuantityHandle) or not port.is_instance:
            raise TypeError("source global input requires the exact block[global_quantity] handle")
        if port.block_ref != state.state_ref.block_ref:
            raise ValueError("physical global source input belongs to another block")
        # Authenticate through the live Case registry; equal-looking metadata is insufficient.
        issued = port.block_ref[port.declaration_ref]
        if issued is not port:
            raise ValueError("physical global source input is not the registry-issued port")
        ref = wanted.get(port.local_id)
        if (ref is None or port.declaration_ref != ref.handle
                or port.units != ref.units or port.local_id in seen):
            raise ValueError("physical global source input declaration/units changed or is unused")
        if not isinstance(expression, ProgramGlobal):
            raise TypeError("physical global source requires a typed IntegralState capture")
        capture = require_owned_value(program, expression.value, "physical global source capture")
        if (capture.op != "integral_candidate" or capture.vtype != "scalar"
                or capture.point != state.point or capture.attrs.get("scope") != "candidate"):
            raise ValueError("physical global source capture requires the same exact point and candidate scope")
        if capture.attrs.get("units") != integral_units_bytes(port.units):
            raise ValueError("physical global source and IntegralState units differ")
        seen.add(port.local_id)
        rows.append({"port": port, "units": capture.attrs["units"],
                     "input": len(captures), "version": 1})
        captures.append(capture)
    if seen != set(wanted):
        raise ValueError("physical global source requires explicit bindings for every declared read")
    return tuple(rows), tuple(captures)
