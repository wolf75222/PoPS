"""Compiled reports obtain exact model facts through a structural provider."""
from __future__ import annotations

import pytest

from pops.codegen._artifact_models import _metadata, component_model_metadata
from pops.codegen.loader import CompiledProblem


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


def _compiled_problem(*, routes):
    compiled = object.__new__(CompiledProblem)
    compiled.model = ExternalMetadataProvider()
    compiled.program_name = "program_label_must_not_name_the_model"
    compiled.program_block_routes = routes
    return compiled


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


def test_component_metadata_uses_the_unique_program_block_route():
    compiled = _compiled_problem(routes=((0, "plasma"),))

    row, = component_model_metadata(compiled)

    assert row.block_name == "plasma"
    assert row.block_name != compiled.program_name


@pytest.mark.parametrize("routes", [(), ((0, "plasma"), (1, "ions"))])
def test_component_metadata_refuses_missing_or_multiple_program_block_routes(routes):
    compiled = _compiled_problem(routes=routes)

    with pytest.raises(ValueError, match="exactly one program block route"):
        component_model_metadata(compiled)


@pytest.mark.parametrize("routes", [((0,),), ((0, ""),), (("0", "plasma"),)])
def test_component_metadata_refuses_ambiguous_program_block_routes(routes):
    compiled = _compiled_problem(routes=routes)

    with pytest.raises(ValueError, match="unambiguous"):
        component_model_metadata(compiled)
