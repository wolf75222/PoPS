"""Authenticated field-nullspace provider contracts."""
from __future__ import annotations

import pytest

from pops.fields import ConstantNullspace, MeanValueGauge
from pops.fields._prepared_field_nullspace_registry import (
    PreparedFieldNullspaceFacts,
    PreparedFieldNullspaceProvider,
    PreparedFieldNullspaceResolution,
    prepared_field_nullspace_binding,
    prepared_field_nullspace_binding_from_data,
    register_prepared_field_nullspace_provider,
)


def _facts(components: int) -> PreparedFieldNullspaceFacts:
    return PreparedFieldNullspaceFacts(
        topology_identity="example.topology@1",
        kernel_components=components,
        operator={"identity": "example.operator@1"},
    )


def test_mean_value_is_provider_owned_and_preserved_in_native_identity() -> None:
    binding = prepared_field_nullspace_binding(
        ConstantNullspace(), MeanValueGauge(3.25),
        facts=_facts(1), where="example nullspace",
    )

    contract = binding.resolution.native_contract
    assert contract["options"] == {"gauge.value": 3.25}
    restored = prepared_field_nullspace_binding_from_data(binding.to_data())
    assert restored.identity == binding.identity


def test_inferred_nonsingular_provider_uses_same_opaque_native_protocol() -> None:
    binding = prepared_field_nullspace_binding(
        None, None, facts=_facts(0), where="example nullspace",
    )
    contract = binding.resolution.native_contract
    assert contract["provider_route"] == "pops.field-nullspace.operator-topology-derived"
    assert contract["options"] == {"gauge.value": 0.0}
    assert binding.resolution.singular is False


def test_registering_an_extension_cannot_change_the_versioned_default_policy() -> None:
    before = prepared_field_nullspace_binding(
        None, None, facts=_facts(0), where="default before extension",
    )

    def unused_author(options, gauge, facts, where):
        raise AssertionError("an explicitly unselected extension must not author a binding")

    extension = register_prepared_field_nullspace_provider(
        PreparedFieldNullspaceProvider(
            provider_id="tests.field-nullspace.overlapping-extension",
            version=1,
            resolver_id="tests.field-nullspace.overlapping-extension.resolve@1",
            resolution_validator_id=(
                "tests.field-nullspace.overlapping-extension.validate-resolution@1"
            ),
            installer_id="tests.field-nullspace.overlapping-extension.install@1",
            capabilities={"kernel_components": 0, "gauge": "none"},
            author=unused_author,
            resolution_validator=lambda binding, where: None,
            native_installer=lambda context, binding: None,
        )
    )
    assert extension.provider_id == "tests.field-nullspace.overlapping-extension"

    after = prepared_field_nullspace_binding(
        None, None, facts=_facts(0), where="default after extension",
    )
    assert after.provider == before.provider
    assert after.identity == before.identity


def test_provider_resolution_validator_rejects_reidentified_native_forgery() -> None:
    binding = prepared_field_nullspace_binding(
        ConstantNullspace(), MeanValueGauge(2.0),
        facts=_facts(1), where="example nullspace",
    )
    provider = binding.provider
    from pops.fields._prepared_field_nullspace_registry import (
        prepared_field_nullspace_provider_from_identity,
    )

    authority = prepared_field_nullspace_provider_from_identity(provider)
    forged = type(binding).create(
        provider=authority,
        options=binding.options,
        facts=binding.facts,
        resolution=PreparedFieldNullspaceResolution(
            {"provider_route": "forged", "schema_identity": "forged@1", "options": {}},
            True,
        ),
    )
    with pytest.raises(ValueError, match="provider route changed"):
        prepared_field_nullspace_binding_from_data(forged.to_data())
