"""Exact physical frame admission for Program-owned state storage."""
from collections.abc import Mapping

from pops._cartesian_axes import canonical_axis_mapping


def prepare_source_storage_carrier(emitter, module, *, state_space=None):
    impl = getattr(emitter, "_m", emitter)
    if impl._flux:
        return
    states = module.state_spaces()
    if state_space is None:
        if len(states) != 1:
            return
        state = next(iter(states.values()))
    else:
        state = state_space
    operators = tuple(module.operator_registry())
    if any(op.kind == "grid_operator" and state in op.signature.inputs for op in operators):
        return
    frames = set()
    for operator in operators:
        view = operator.lowering.get("physical_balance")
        if (view is None or not view.accumulation.is_identity or not view.occurrences
                or any(row.kind != "source" for row in view.occurrences)
                or operator.signature.inputs[0] != state):
            continue
        axes = tuple(operator.capabilities.get("storage_axes", ()))
        frame = operator.capabilities.get("storage_frame")
        if not axes or frame != state.frame:
            raise ValueError("source-only state storage requires its exact authored Cartesian frame")
        axes = tuple(canonical_axis_mapping(dict.fromkeys(axes), where="source-only storage frame"))
        frames.add((axes, frame))
    if not frames:
        return
    if len(frames) != 1:
        raise ValueError("source-only state storage has conflicting physical frames")
    axes, _ = next(iter(frames))
    previous = getattr(impl, "_program_only_storage_axes", axes)
    if previous != axes:
        raise ValueError("source-only state storage differs from the selected physical frame")
    object.__setattr__(impl, "_program_only_storage_axes", axes)


def prepare_named_flux_storage_carrier(emitter, module, resolved_operations):
    """Select state-only storage for an entirely Program-owned named-flux route."""
    del module
    if resolved_operations is None:
        return
    transport = tuple(
        operation
        for operation in resolved_operations.operations
        if operation.exchanges and "program_evaluation" in operation.guarantees
    )
    if not transport:
        return
    methods = tuple(operation.guarantees.get("numerical_method") for operation in transport)
    if any(
        operation.sampling != "cell_centered_divergence"
        or not isinstance(method, Mapping)
        or method.get("method") != "native_named_centered_divergence"
        for operation, method in zip(transport, methods, strict=True)
    ):
        return

    flux_packs = tuple(method.get("physical_fluxes") for method in methods)
    if any(
        not isinstance(pack, list)
        or not pack
        or any(not isinstance(name, str) or not name for name in pack)
        for pack in flux_packs
    ):
        raise ValueError(
            "named centered-divergence storage requires exact named physical-flux identities"
        )

    impl = getattr(emitter, "_m", emitter)
    names = tuple(dict.fromkeys(
        name
        for pack in flux_packs
        for name in pack
    ))
    if any(name not in impl._flux_terms for name in names):
        raise ValueError("named centered-divergence storage lost an authored flux expression")
    axes = tuple(impl._flux_terms[names[0]])
    if not axes or any(tuple(impl._flux_terms[name]) != axes for name in names[1:]):
        raise ValueError("named centered-divergence fluxes require one exact Cartesian axis set")
    axes = tuple(canonical_axis_mapping(dict.fromkeys(axes), where="named flux storage frame"))
    if impl._flux and tuple(impl._flux) != axes:
        raise ValueError("named centered-divergence storage differs from the model flux frame")
    previous = getattr(impl, "_program_only_storage_axes", axes)
    if previous != axes:
        raise ValueError("named centered-divergence storage differs from another storage authority")
    object.__setattr__(impl, "_program_only_storage_axes", axes)
    # The exact resolved Program owns every evolved transport evaluation.  Retain named formulas
    # for Program codegen, while removing the unused legacy default-flux and eigenvalue routes.
    object.__setattr__(impl, "_flux", {})
    object.__setattr__(impl, "_eig", {})
