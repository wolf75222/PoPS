"""Authenticate adapter claims against the existing scientific Module declarations."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from fractions import Fraction
from typing import Any

from ._resolved_operation_inputs import _reference, derive_module_operations
from ._resolved_operation_records import ExchangeRecord


def require_operation_authority(plan: Any, operation: Any, module: Any) -> None:
    """Allow numerical refinement without replacing the source law or its obligations."""
    from .component_provider_packs import resolve_component_provider_packs
    from .resolved_operations import _reject

    def reject(message: str) -> None:
        _reject(operation.identity, "resolved_operation_authority_mismatch", message)

    requests, operations = derive_module_operations(
        module, resolve_component_provider_packs(module))
    declaration = operation.guarantees.get("declaration_operation", operation.identity)
    originals = {item.identity: item for item in operations}
    if declaration not in originals:
        reject("native construction has no current scientific declaration")
    original = originals[declaration]
    definition = next(item for item in module.operator_registry()
                      if "operation:%s" % _reference(module.operator_handle(item.name)) == declaration)
    original_request = next(item for item in requests if item.identity == original.evaluation)
    request = next(item for item in plan.evaluations if item.identity == operation.evaluation)
    actual_terms = tuple(item for item in request.occurrences if item.identity in operation.consumes)
    context = operation.guarantees.get("program_evaluation")
    if context is not None and not isinstance(context, Mapping):
        reject("native Program evaluation context must retain its structured identity")
    selection = None if context is None else context.get("selection")
    centered = False

    if selection is not None:
        # The retained Program.rhs adapter selects one grid/source contribution per
        # clone. Its sign comes from that operation kind, never from wire metadata.
        if context.get("operation") != "rhs" or definition.kind not in {
                "grid_operator", "local_source"} or len(original_request.occurrences) != 1:
            reject("native RHS contribution is not a supported scientific declaration")
        coefficient = -1 if definition.kind == "grid_operator" else 1
        if not isinstance(selection, Mapping) \
                or dict(selection) != {"name": definition.name, "coefficient": coefficient} \
                or len(actual_terms) != 1:
            reject("native RHS selection differs from its scientific contribution")
        prototype = original_request.occurrences[0]
        expected_terms = (replace(prototype, identity=actual_terms[0].identity,
                                  kind="transport" if coefficient < 0 else "source",
                                  coefficient=Fraction(coefficient)),)
        expected_exchanges = ((ExchangeRecord(actual_terms[0].identity, prototype.target),)
                              if coefficient < 0 else ())
        centered = coefficient < 0 and operation.sampling == "cell_centered_divergence"
    else:
        expected_terms = original_request.occurrences
        expected_exchanges = original.exchanges
        if operation.consumes != original.consumes:
            reject("native construction changes scientific occurrence coverage")
        if context is None and operation.identity != declaration:
            reject("native construction lacks an authenticated declaration or Program evaluation")

    if actual_terms != expected_terms:
        reject("native construction changes a scientific occurrence, coefficient, or target")
    if operation.exchanges != expected_exchanges:
        reject("native construction changes a scientific exchange obligation")
    if operation.outputs != original.outputs:
        reject("native construction changes its scientific output representation")
    if not set(original.effects) <= set(operation.effects):
        reject("native construction discards a required scientific effect")
    if centered:
        method = operation.guarantees.get("numerical_method")
        if operation.stencil_radius != 1 or method is None \
                or method.get("method") != "native_named_centered_divergence":
            reject("native centered-divergence sampling lacks its checked numerical method")
    elif operation.sampling != original.sampling:
        reject("native construction changes sampling without a supported adapter realization")

    for required in original.inputs:
        sampling = "cell" if centered and "face_trace" in required.sampling else required.sampling
        candidates = tuple(read for read in operation.inputs
                           if read.reference == required.reference and read.kind == required.kind
                           and read.representation == required.representation
                           and read.sampling == sampling and read.physical_map == required.physical_map)
        complete = any(read.complete for read in candidates)
        components = {component for read in candidates for component in read.components}
        if not candidates or (not complete and not set(required.components) <= components) \
                or (required.complete and not required.components and not complete):
            reject("native construction discards or changes a required scientific input")
