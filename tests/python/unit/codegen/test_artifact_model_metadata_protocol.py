"""Compiled reports obtain exact model facts through a structural provider."""
from __future__ import annotations

import pytest

from pops.codegen._artifact_models import _metadata


class ExternalMetadataProvider:
    def __pops_artifact_model_metadata__(self):
        return {
            "schema_version": 1,
            "state_spaces": ("electrons",),
            "cons_names": ("density", "momentum"),
            "n_vars": 2,
            "params": {},
            "aux_names": ("electric_field",),
            "n_aux": 3,
            "capabilities": {"cpu": True, "amr": False},
        }


def test_external_metadata_provider_is_consumed_without_concrete_class_dispatch():
    provider = ExternalMetadataProvider()

    row = _metadata(
        "plasma", provider, expected_state_spaces=("electrons",))

    assert row.model is provider
    assert row.block_name == "plasma"
    assert row.state_space == "electrons"
    assert row.cons_names == ("density", "momentum")
    assert row.capabilities == {"cpu": True, "amr": False}


def test_metadata_provider_refuses_state_route_drift_and_fabricated_counts():
    provider = ExternalMetadataProvider()
    with pytest.raises(ValueError, match="state-space route"):
        _metadata("plasma", provider, expected_state_spaces=("ions",))

    data = provider.__pops_artifact_model_metadata__()
    data["n_vars"] = 3

    class Invalid:
        def __pops_artifact_model_metadata__(self):
            return data

    with pytest.raises(ValueError, match="exactly match"):
        _metadata("plasma", Invalid(), expected_state_spaces=("electrons",))
