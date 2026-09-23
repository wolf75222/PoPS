"""Authenticate the complete Fan–Li law and selected path before native emission."""
from __future__ import annotations

from types import MappingProxyType


def prepare_path_carrier(emitter, module, resolved_operations, numerics):
    from pops.numerics.nonconservative import PathConservativeFiniteVolume, path_balance_supported
    from pops.model.state_symbols import rebind_state_symbols
    from pops.model.hash_data import canonical_hash_data
    from pops.moments.fan_li import fan_li15_expressions, FAN_LI15_REGULARIZED_COMPONENTS
    from pops.moments.model_builder import moment_names
    from pops._ir.expr import _wrap
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
                    if type(row.method) is PathConservativeFiniteVolume)
    if len(laws) != 1 or len(methods) != 1 or len(module.state_spaces()) != 1:
        raise ValueError("Fan–Li15 native transport requires one complete state, law and path rate")
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
        raise ValueError("Fan–Li15 native path requires the exact full signed flux/product balance")
    state = next(iter(module.state_spaces().values()))
    if tuple(state.components) != tuple(moment_names(4)):
        raise ValueError("Fan–Li15 native path requires the complete q-outer raw moment storage")
    conserved = tuple(name for slot, name in enumerate(moment_names(4))
                      if slot not in FAN_LI15_REGULARIZED_COMPONENTS)
    if law.conservative_components != conserved:
        raise ValueError("Fan–Li15 native path lost a conservative row")
    impl = getattr(emitter, "_m", emitter)

    def native(value):
        return rebind_state_symbols(value, state, module.state_spaces().values(), module=module,
                                    quantity_handles=(law.state,))

    variables = native(law.variables)
    covectors = native(method.path.covectors)
    physical = fan_li15_expressions(variables)
    expected_b = tuple(tuple(tuple(_wrap(value) for value in row)
                             for row in physical.directional_nonconservative_matrix(g))
                       for g in covectors)
    if canonical_hash_data(native(law.matrices)) != canonical_hash_data(expected_b):
        raise ValueError("native Fan–Li15 matrix differs from its authenticated constitutive law")
    flux_occurrence = next(row for row in view.occurrences if row.kind == "flux")
    product_occurrence = next(row for row in view.occurrences if row.kind == "nonconservative")
    if (flux_occurrence.payload.reg_name != method.flux.reg_name
            or product_occurrence.payload.reg_name != laws[0].name):
        raise ValueError("native path selection changes the physical rate dependencies")
    flux = module.operator_registry().get(flux_occurrence.payload.reg_name)
    expected_f = {axis: physical.directional_flux(g)
                  for axis, g in zip(law.axes, covectors, strict=True)}
    if canonical_hash_data(native(flux.body)) != canonical_hash_data(expected_f):
        raise ValueError("native Fan–Li15 speed proof requires the complete Grad flux plus B")
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
    identity = make_identity("fan-li15.path-operator", semantic_value({
        "method": method_data, "physical_law": law.to_data(),
    }, where="complete Fan-Li15 path operator")).token
    mask = (tuple(face in method.zero_measure_faces for face in method.path.frame.boundaries.all)
            if method.zero_measure_faces else (False,) * 4)
    object.__setattr__(impl, "_path_conservative", MappingProxyType({
        "identity": identity, "covectors": covectors, "zero_measure_faces": mask,
        "method": method_data, "conservative_components": conserved,
    }))
    return emitter


def require_path_numerical_authority(operation, module):
    """A retained physical product alone never grants an executable weak-solution path."""
    if operation.native_route != "program:path_conservative_rhs":
        return
    from collections.abc import Mapping
    method = operation.guarantees.get("numerical_method")
    if (not isinstance(method, Mapping)
            or method.get("method") != "path_conservative_finite_volume"
            or method.get("formal_order") != 1
            or method.get("nonconservative_interfaces") != "canonical_fine_subface_side_contributions"):
        raise ValueError("native nonconservative transport has no authenticated path method")
    path = method.get("path")
    if (not isinstance(path, Mapping)
            or path.get("kind") != "fan_li15_straight_raw_moment_path"
            or path.get("integral") != "density_oriented_polynomial_logarithm_v1"
            or path.get("face_geometry") != "arithmetic_trace_covector"
            or path.get("stability") != "whole_path_raw_second_moment_bound"):
        raise ValueError("native nonconservative transport path identity is unsupported")
