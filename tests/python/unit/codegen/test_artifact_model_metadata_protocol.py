"""Compiled reports obtain exact model facts through a structural provider."""
from __future__ import annotations

import pytest

from pops.codegen._artifact_models import _metadata, component_model_metadata
from pops.codegen.loader import CompiledProblem


class ExternalMetadataProvider:
    def __pops_artifact_model_metadata__(self):
        return {
            "schema_version": 3,
            "state_spaces": ("electrons",),
            "cons_names": ("density", "momentum"),
            "cons_roles": ("density", "momentum:0"),
            "n_vars": 2,
            "params": {},
            "provider_components": ("electric_field",),
            "n_aux": 3,
            "native_dimension": 2,
            "capabilities": {"cpu": True, "amr": False},
            "wave_speed_provider": "jacobian",
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
    assert row.cons_roles == ("density", "momentum:0")
    assert row.native_dimension == 2
    assert row.capabilities == {"cpu": True, "amr": False}
    assert row.wave_speed_provider == "jacobian"


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


def test_metadata_provider_refuses_unknown_wave_speed_provenance():
    data = ExternalMetadataProvider().__pops_artifact_model_metadata__()
    data["wave_speed_provider"] = "inferred_somehow"

    class Invalid:
        def __pops_artifact_model_metadata__(self):
            return data

    with pytest.raises(ValueError, match="wave_speed_provider"):
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


@pytest.mark.parametrize("target", ("system", "amr_system"))
def test_compiled_auxiliary_metadata_uses_the_resolved_provider_pack(tmp_path, monkeypatch, target):
    """Exercise the metadata producer with only native compilation replaced by inert bytes."""
    from dataclasses import replace

    from pops.codegen.component_provider_packs import resolve_emitter_provider_packs
    from pops.codegen.inspect_compiled import _build_aux_arguments
    from pops.fields import AuxiliaryBoundary, DerivedAux
    from pops.math import ValueExpr
    from pops.physics._facade import Model

    model = Model("compiled_auxiliary_inventory")
    (q,) = model.conservative_vars("q")
    imposed = model.aux("imposed")
    model.flux(x=[0.0 * q], y=[0.0 * q])
    model.projection([imposed])
    module = model.module
    derived = module.aux_field("derived")
    module.aux_provider(DerivedAux(
        module.aux_handle(derived),
        2.0 * ValueExpr(module.field_handle(module.field_spaces()["fields"])),
        boundary=AuxiliaryBoundary(width=1, kind="foextrap")))
    packs = resolve_emitter_provider_packs(model, module)
    model.__pops_bind_component_provider_packs__(packs)
    assert model._m._provider_components == ["imposed"]
    assert {key.component for key in packs.auxiliary} == {"derived", "imposed"}

    def inert_compile(path, include, **options):
        assert options["target"] == target
        from pathlib import Path
        Path(path).write_bytes(b"source-only metadata producer control; not a native library")
        return str(path)

    monkeypatch.setattr(model._m, "compile", inert_compile)
    monkeypatch.setattr("pops.codegen.abi._abi_key_python", lambda *_: "source-only-metadata-abi")
    compiled = model.compile(
        str(tmp_path / "source-only.so"), include=str(tmp_path), target=target)

    for provider in (model, compiled):
        row = _metadata("scalar", provider, expected_state_spaces=("U",))
        assert row.provider_components == tuple(key.component for key in packs.auxiliary)
        assert row.n_aux == 2
        assert _build_aux_arguments((row,), {}, {"scalar": packs.auxiliary}) == {
            "imposed": {"layout": "cell", "required": True}}
        for components in (("imposed",), ("derived", "imposed", "foreign")):
            with pytest.raises(ValueError, match="metadata differs from its resolved ProviderPack"):
                _build_aux_arguments((replace(row, provider_components=components),), {},
                                     {"scalar": packs.auxiliary})
