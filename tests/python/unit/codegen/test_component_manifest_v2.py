"""ADC-679 complete ComponentManifest schema and fingerprint contract."""
from __future__ import annotations

from copy import deepcopy

import pytest

from pops.model import (
    ComponentExtensionSchema,
    ComponentManifest,
    ComponentManifestError,
)
from pops.identity import canonical_bytes


def _values():
    return {
        "uri": "pops://external.test/components/central-upwind",
        "component_type": "spatial_operator",
        "version": "1.2.3",
        "facets": ("lowering", "stencil"),
        "signature": {"inputs": ["state"], "outputs": ["rate"]},
        "reads": ({"resource": "state:u"},),
        "writes": ({"resource": "rate:u"},),
        "parameters": ({"name": "theta", "kind": "runtime"},),
        "interfaces": ({"uri": "pops://interfaces/spatial-operator", "version": 1},),
        "requirements": ({"capability": "halo", "depth": 2},),
        "capabilities": ({"capability": "formal_order", "value": 2},),
        "effects": ({"kind": "state_read", "resource": "state:u"},),
        "layouts": ({"space": "cell", "layout": "structured"},),
        "clocks": ({"clock": "solution", "access": "stage"},),
        "target": {
            "variants": [{
                "dimension": 2, "scalar": "float64", "device": "cpu",
                "features": ["kokkos"],
            }],
        },
        "determinism": {"classification": "reproducible", "scope": ["rank_count"]},
        "restart": {
            "mode": "stateful", "schema_uri": "pops://schemas/restart/central-upwind",
            "schema_version": 1,
        },
        "precision": {
            "inputs": ["float64"], "accumulation": "float64", "outputs": ["float64"],
        },
        "conservation": ({"quantity": "mass", "scope": "closed_domain"},),
        "entry_points": {"native": "pops::CentralUpwind"},
    }


def test_complete_manifest_round_trips_in_canonical_form():
    manifest = ComponentManifest(**_values())
    reopened = ComponentManifest.from_data(manifest.to_data())
    assert reopened == manifest
    assert reopened.to_bytes() == manifest.to_bytes()
    assert reopened.semantic_digest == manifest.semantic_digest
    assert reopened.component_id.endswith("@1.2.3")


@pytest.mark.parametrize("field,replacement", [
    ("uri", "pops://external.test/components/other"),
    ("component_type", "numerical_flux"),
    ("version", "1.2.4"),
    ("facets", ("lowering",)),
    ("signature", {"inputs": ["state"], "outputs": ["flux"]}),
    ("reads", ({"resource": "state:v"},)),
    ("writes", ({"resource": "rate:v"},)),
    ("parameters", ({"name": "epsilon", "kind": "runtime"},)),
    ("interfaces", ({"uri": "pops://interfaces/flux", "version": 1},)),
    ("requirements", ({"capability": "halo", "depth": 3},)),
    ("capabilities", ({"capability": "formal_order", "value": 3},)),
    ("effects", ({"kind": "state_write", "resource": "state:u"},)),
    ("layouts", ({"space": "face_x", "layout": "structured"},)),
    ("clocks", ({"clock": "fast", "access": "stage"},)),
    ("target", {"variants": [{
        "dimension": 3, "scalar": "float64", "device": "cpu", "features": ["kokkos"],
    }]}),
    ("determinism", {"classification": "bitwise", "scope": ["rank_count"]}),
    ("restart", {"mode": "stateless", "schema_uri": "", "schema_version": 0}),
    ("precision", {"inputs": ["float32"], "accumulation": "float64",
                   "outputs": ["float32"]}),
    ("conservation", ({"quantity": "energy", "scope": "closed_domain"},)),
    ("entry_points", {"native": "pops::Other"}),
])
def test_every_semantic_field_changes_the_semantic_fingerprint(field, replacement):
    baseline = _values()
    changed = deepcopy(baseline)
    changed[field] = replacement
    assert ComponentManifest(**baseline).semantic_digest != ComponentManifest(**changed).semantic_digest


def test_documentary_extension_does_not_change_semantics_but_changes_full_manifest():
    baseline = ComponentManifest(**_values())
    values = _values()
    values["extensions"] = {
        "https://example.test/extensions/docs": {
            "kind": "documentary", "data": {"summary": "Central upwind reference"},
        }
    }
    documented = ComponentManifest(**values)
    assert documented.semantic_digest == baseline.semantic_digest
    assert documented.manifest_digest != baseline.manifest_digest


def test_semantic_extension_requires_a_versioned_schema_and_enters_semantic_identity():
    values = _values()
    values["extensions"] = {
        "https://example.test/extensions/model": {
            "kind": "semantic",
            "schema_uri": "https://example.test/schemas/model-extension",
            "schema_version": 1,
            "data": {"closure": "isothermal"},
        }
    }
    with pytest.raises(ComponentManifestError) as error:
        ComponentManifest(**values)
    assert error.value.code == "unknown_semantic_extension_schema"

    schema = ComponentExtensionSchema(
        "https://example.test/schemas/model-extension", 1, required_fields=("closure",))
    first = ComponentManifest(**values, extension_schemas={schema.key: schema})
    changed = deepcopy(values)
    changed["extensions"]["https://example.test/extensions/model"]["data"]["closure"] = "energy"
    second = ComponentManifest(**changed, extension_schemas={schema.key: schema})
    assert first.semantic_digest != second.semantic_digest


def test_unknown_semantic_top_level_field_is_a_structured_refusal():
    data = ComponentManifest(**_values()).to_data()
    data["surprise"] = True
    with pytest.raises(ComponentManifestError) as error:
        ComponentManifest.from_data(data)
    assert error.value.code == "semantic_field_mismatch"
    assert error.value.evidence["unknown"] == ["surprise"]


def test_component_uri_requires_a_real_namespace_authority():
    values = _values()
    values["uri"] = "pops:/local-only"
    with pytest.raises(ComponentManifestError) as error:
        ComponentManifest(**values)
    assert error.value.code == "invalid_component_uri"
    assert error.value.path == "uri"


def test_target_capability_refusal_contains_requested_and_supported_evidence():
    manifest = ComponentManifest(**_values())
    manifest.require_target({
        "dimension": 2, "scalar": "float64", "device": "cpu", "features": ["kokkos"],
    })
    with pytest.raises(ComponentManifestError) as error:
        manifest.require_target({
            "dimension": 3, "scalar": "float64", "device": "cpu", "features": ["kokkos"],
        })
    assert error.value.code == "unsupported_target_combination"
    assert error.value.evidence["requested"]["dimension"] == 3
    assert error.value.evidence["supported"][0]["dimension"] == 2


def test_target_variants_do_not_invent_unsupported_axis_cross_products():
    values = _values()
    values["target"] = {"variants": [
        {"dimension": 2, "scalar": "float64", "device": "cpu", "features": []},
        {"dimension": 2, "scalar": "float32", "device": "cuda", "features": ["sm80"]},
    ]}
    manifest = ComponentManifest(**values)
    manifest.require_target({
        "dimension": 2, "scalar": "float32", "device": "cuda", "features": ["sm80"],
    })
    with pytest.raises(ComponentManifestError) as error:
        manifest.require_target({
            "dimension": 2, "scalar": "float64", "device": "cuda", "features": ["sm80"],
        })
    assert error.value.code == "unsupported_target_combination"


def test_native_parser_normalizer_matches_python_canonical_bytes():
    from pops import _pops

    manifest = ComponentManifest(**_values())
    assert _pops._component_manifest_canonical_bytes(manifest.to_data()) == canonical_bytes(
        manifest.to_data())
    assert _pops._component_manifest_semantic_bytes(manifest.to_data()) == manifest.semantic_bytes


def test_native_parser_preserves_documentary_and_versioned_semantic_extensions():
    from pops import _pops

    schema = ComponentExtensionSchema(
        "https://example.test/schemas/model-extension", 1, required_fields=("closure",))
    values = _values()
    values["extensions"] = {
        "https://example.test/extensions/docs": {
            "kind": "documentary", "data": {"summary": "reference implementation"},
        },
        "https://example.test/extensions/model": {
            "kind": "semantic",
            "schema_uri": schema.uri,
            "schema_version": schema.version,
            "data": {"closure": "isothermal"},
        },
    }
    manifest = ComponentManifest(**values, extension_schemas={schema.key: schema})
    assert _pops._component_manifest_canonical_bytes(manifest.to_data()) == canonical_bytes(
        manifest.to_data())
    assert _pops._component_manifest_semantic_bytes(manifest.to_data()) == manifest.semantic_bytes


def test_native_parser_raises_the_same_structured_manifest_error_type():
    from pops import _pops

    data = ComponentManifest(**_values()).to_data()
    data["surprise"] = True
    with pytest.raises(ComponentManifestError) as error:
        _pops._component_manifest_canonical_bytes(data)
    assert error.value.code == "unknown_semantic_field"
    assert error.value.path == "ComponentManifest.surprise"
