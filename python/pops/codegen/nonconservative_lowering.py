"""Authenticate a physical product and its selected native numerical path."""
from __future__ import annotations

from types import MappingProxyType


def prepare_path_carrier(emitter, module, resolved_operations, numerics):
    from pops.numerics.nonconservative import (PathConservativeFiniteVolume,
        CoordinatedFiniteVolume, path_balance_supported)
    from pops.model.state_symbols import rebind_state_symbols
    from pops.identity import canonical_bytes
    from pops.identity.digest import make_identity
    from pops.identity.semantic import semantic_value

    declarations = tuple(module.operator_registry())
    laws = tuple(op for op in declarations if op.lowering.get("nonconservative_law") is not None)
    if not laws:
        return emitter
    if numerics is None or resolved_operations is None:
        raise ValueError("nonconservative native emission requires its resolved numerical path")
    methods = tuple(row for row in numerics.rates
                    if type(row.method) in (PathConservativeFiniteVolume, CoordinatedFiniteVolume))
    if len(laws) != 1 or len(methods) != 1 or len(module.state_spaces()) != 1:
        raise ValueError("native path transport requires one complete state, law and path rate")
    selection = methods[0]
    method = selection.method
    law = laws[0].lowering["nonconservative_law"]
    if (canonical_bytes(semantic_value(law.to_data(), where="native path law"))
            != canonical_bytes(semantic_value(method.path.product.law.to_data(),
                                              where="selected path law"))):
        raise ValueError("selected path differs from its retained physical nonconservative law")
    rate = module.operator_registry().get(selection.rate.registered_operator_name)
    view = rate.lowering.get("physical_balance")
    if not path_balance_supported(view):
        raise ValueError("native path requires the exact full signed flux/product balance")
    state = next(iter(module.state_spaces().values()))
    conserved = law.conservative_components
    impl = getattr(emitter, "_m", emitter)

    def native(value):
        return rebind_state_symbols(value, state, module.state_spaces().values(), module=module,
                                    quantity_handles=(law.state,))

    covectors = native(method.path.covectors)
    flux_occurrence = next(row for row in view.occurrences if row.kind == "flux")
    product_occurrence = next(row for row in view.occurrences if row.kind == "nonconservative")
    if (flux_occurrence.payload.reg_name != method.flux.reg_name
            or product_occurrence.payload.reg_name != laws[0].name):
        raise ValueError("native path selection changes the physical rate dependencies")
    flux = module.operator_registry().get(flux_occurrence.payload.reg_name)
    method.path.validate_native(law=law, flux_body=flux.body, native=native)
    kernel = method.path.native_kernel()
    if kernel["kind"] == "coordinated_face":
        from pops._ir.values import RuntimeParamRef
        from pops._ir.visitors import _children
        registry = getattr(emitter, "_param_registry", None)
        registered_parameters = {handle._resolved(): handle for handle in registry.handles()}
        pending, seen = list(kernel["parameter_expressions"]), set()
        while pending:
            node = pending.pop()
            if id(node) in seen:
                continue
            seen.add(id(node))
            if isinstance(node, RuntimeParamRef):
                if node.handle.is_instance and node.handle.block_ref != numerics.block:
                    raise ValueError("coordinated face parameter belongs to another block instance")
                handle = node.handle.declaration_ref if node.handle.is_instance else node.handle
                try:
                    registered = registry.handle(registered_parameters[handle._resolved()])
                except (AttributeError, KeyError, ValueError) as exc:
                    raise ValueError("coordinated face parameter belongs to another model") from exc
                if registered.param_kind != "runtime":
                    raise ValueError("coordinated face capture is not a RuntimeParam")
            pending.extend(_children(node))
    method_data = method.to_data()
    evaluations = tuple(operation for operation in resolved_operations.operations
                        if "program_evaluation" in operation.guarantees
                        and operation.native_route == "program:path_conservative_rhs")
    if not evaluations or any(
            canonical_bytes(semantic_value(operation.guarantees.get("numerical_method"),
                                           where="native path numerical authority"))
            != canonical_bytes(semantic_value(method_data, where="selected path method"))
            for operation in evaluations):
        raise ValueError("every native path evaluation requires the exact resolved method identity")
    if any(operation.native_route == "legacy:rate_operator"
           and "program_evaluation" in operation.guarantees for operation in resolved_operations.operations):
        raise ValueError("a path block cannot also evolve an ordinary default-flux rate")
    identity = make_identity(kernel["identity_namespace"], semantic_value({
        "method": method_data, "physical_law": law.to_data(),
    }, where="complete numerical path operator")).token
    mask = (tuple(face in method.zero_measure_faces for face in method.path.frame.boundaries.all)
            if method.zero_measure_faces else (False,) * (2 * len(law.axes)))
    object.__setattr__(impl, "_path_conservative", MappingProxyType({
        "identity": identity, "covectors": covectors, "zero_measure_faces": mask,
        "method": method_data, "conservative_components": conserved,
        "kernel": kernel,
    }))
    return emitter


def require_path_numerical_authority(operation, module):
    """A retained physical product alone never grants an executable weak-solution path."""
    if operation.native_route != "program:path_conservative_rhs":
        return
    from collections.abc import Mapping
    method = operation.guarantees.get("numerical_method")
    if (not isinstance(method, Mapping)
            or method.get("method") not in ("path_conservative_finite_volume", "coordinated_finite_volume")
            or method.get("formal_order") != 1
            or method.get("nonconservative_interfaces") != "canonical_fine_subface_side_contributions"):
        raise ValueError("native nonconservative transport has no authenticated path method")
    path = method.get("path")
    if method.get("method") == "coordinated_finite_volume":
        if (method.get("interface_contract") != 1 or not isinstance(path, Mapping)
                or path.get("schema_version") != 1 or path.get("kind") != "coordinated_face"
                or path.get("side_sign") != "already_signed_cell_rhs"
                or path.get("publication") != "atomic_shared_flux_two_sides_speed"
                or "body" not in path):
            raise ValueError("native coordinated transport has no complete face identity")
        return
    if (not isinstance(path, Mapping) or path.get("schema_version") != 1
            or not path.get("kind") or "integral" not in path or not path.get("stability")):
        raise ValueError("native nonconservative transport has no complete path identity")
