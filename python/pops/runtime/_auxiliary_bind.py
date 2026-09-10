"""Authenticate bind inputs against the same resolved packs used by native registrars."""
from __future__ import annotations

from typing import Any

from pops.model.provider_pack import ComponentKey, ProviderPack


def validate_auxiliary_bind_inputs(artifact: Any, values: Any) -> None:
    """Require exactly the declared InputAux keys, with no name or field-output alias."""
    from pops.codegen._compiled_artifact import CompiledSimulationArtifact
    from pops.codegen._plans import _component_mapping

    if type(artifact) is not CompiledSimulationArtifact:
        raise TypeError("auxiliary bind validation requires an exact CompiledSimulationArtifact")
    supplied = _component_mapping(values, where="pops.bind aux")
    declared: dict[ComponentKey, Any] = {}
    for block in artifact.plan.blocks:
        operations = block.resolved_operations
        if operations is None:
            # Older component-free phase records carry no operation/provider authority.  They
            # cannot authorize any upload; their existing required-input gate remains intact.
            if supplied:
                raise ValueError("pops.bind aux requires resolved ProviderPack authority")
            continue
        data = operations.to_data()["provider_evidence"]["auxiliary"]
        pack = ProviderPack.from_data(data)
        for key in pack:
            entry = pack.declared_entry(key)
            previous = declared.get(key)
            if previous is not None and previous != entry:
                raise ValueError("pops.bind auxiliary providers conflict for %r" % (key,))
            declared[key] = entry
    expected = {
        key for key, entry in declared.items()
        if entry.producer == "runtime_input" and entry.availability and entry.slot is not None
    }
    invalid = set(supplied) - expected
    if invalid:
        raise ValueError(
            "pops.bind aux accepts only declared exact InputAux components; "
            "foreign, DerivedAux and field-output keys are invalid: %r" % sorted(invalid)
        )
    missing = expected - set(supplied)
    if missing:
        raise ValueError("pops.bind is missing declared InputAux components: %r" % sorted(missing))


def auxiliary_array_evidence(values: Any) -> dict[str, Any]:
    """Serialize exact keys alongside array evidence without lossy string conversion."""
    from pops.codegen._plans import _component_mapping
    from pops.runtime._bound_snapshot import _array_evidence

    values = _component_mapping(values, where="aux")
    if not values:
        return {}  # Preserve the established provider-empty snapshot identity.
    return {"components": [
        {"key": key.to_data(), "array": _array_evidence(value, where="aux[%r]" % (key,))}
        for key, value in sorted(values.items())
    ]}
