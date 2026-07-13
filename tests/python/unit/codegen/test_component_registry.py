"""ADC-658 component registry trust-boundary contracts."""
from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from pops.model import ComponentManifest, ComponentRegistry


class ExternalStencil:
    def __init__(self, manifest):
        self.component_manifest = manifest

    def stencil(self):
        return {"width": 1}

    def lower(self, context):
        return (context, "external")


def _manifest(**changes):
    values = {
        "uri": "pops://external.test/central-upwind",
        "component_type": "reconstruction",
        "version": "1.0.0",
        "facets": ("lowering", "stencil"),
        "signature": {"kind": "reconstruction", "order": 2},
        "target": {
            "variants": [{
                "dimension": 2, "scalar": "float64", "device": "cpu", "features": [],
            }],
        },
    }
    values.update(changes)
    return ComponentManifest(**values)


def test_manifest_is_immutable_and_has_canonical_content_digest():
    left = _manifest()
    right = _manifest(signature={"order": 2, "kind": "reconstruction"})

    assert left.manifest_bytes == right.manifest_bytes
    assert left.manifest_digest == right.manifest_digest
    assert left.digest.startswith("pops.component-manifest.v2:sha256:")
    assert ComponentManifest.from_data(left.to_data()) == left
    with pytest.raises((FrozenInstanceError, AttributeError)):
        left.version = 2
    with pytest.raises(TypeError):
        left.signature["order"] = 3


def test_external_structural_component_registers_and_resolves_without_allowlist():
    registry = ComponentRegistry()
    component = ExternalStencil(_manifest())

    assert registry.register(component) is component
    assert registry.resolve(component.component_manifest.component_id) is component
    assert registry.resolve(component.component_manifest.component_id).lower("ctx") == (
        "ctx", "external")
    assert registry.revision == 1


def test_semantically_identical_registration_is_idempotent_but_collision_is_atomic():
    registry = ComponentRegistry()
    original = ExternalStencil(_manifest())
    registry.register(original)

    duplicate = ExternalStencil(_manifest())
    assert registry.register(duplicate) is original
    assert registry.revision == 1
    assert len(registry) == 1

    documented = ExternalStencil(_manifest(extensions={
        "https://example.test/extensions/docs": {
            "kind": "documentary", "data": {"summary": "same implementation"},
        },
    }))
    assert registry.register(documented) is original
    assert registry.revision == 1

    collision = ExternalStencil(_manifest(signature={"kind": "reconstruction", "order": 3}))
    with pytest.raises(ValueError, match="identity collision"):
        registry.register(collision)
    assert registry.revision == 1
    assert registry.resolve(_manifest().component_id) is original


def test_advertised_facet_conformance_is_checked_before_mutation():
    class Malformed:
        component_manifest = _manifest()

        def stencil(self):
            return {"width": 1}

    registry = ComponentRegistry()
    with pytest.raises(TypeError, match="lowering"):
        registry.register(Malformed())
    assert registry.revision == 0
    assert len(registry) == 0


def test_wrong_facet_signature_is_rejected_without_executing_component_code():
    class MalformedSignature:
        component_manifest = _manifest()

        def stencil(self):
            return {"width": 1}

        def lower(self):
            raise AssertionError("registration must not execute lowering")

    registry = ComponentRegistry()
    with pytest.raises(TypeError, match="lowering"):
        registry.register(MalformedSignature())
    assert registry.revision == 0


def test_snapshot_is_revision_stable_and_freeze_refuses_mutation():
    registry = ComponentRegistry()
    first = ExternalStencil(_manifest())
    registry.register(first)
    second = ExternalStencil(_manifest(
        uri="pops://external.test/second", signature={"kind": "reconstruction", "order": 4}
    ))
    registry.register(second)
    with pytest.raises(RuntimeError, match="frozen before snapshot"):
        registry.snapshot()

    assert registry.freeze() is registry
    frozen = registry.snapshot()
    assert registry.frozen and frozen.frozen and frozen.revision == 2 and len(frozen) == 2
    assert frozen.resolve(first.component_manifest.component_id) is first
    with pytest.raises(RuntimeError, match="frozen"):
        registry.register(ExternalStencil(_manifest(uri="pops://external.test/third")))
    assert registry.revision == 2
