"""Strict round-trip proof for the compiled-artifact manifest schema."""
from __future__ import annotations

import copy

import pytest

from pops.external.artifact_manifest import CompiledArtifactManifest


def _manifest() -> CompiledArtifactManifest:
    return CompiledArtifactManifest(
        model_name="transport",
        abi_key="pops-abi-v1",
        abi_version=1,
        required_headers_sig="headers-v1",
        blocks=("tracer",),
        variables=("u",),
        roles=("scalar",),
        dimension=2,
        amr_refinement_ratio=2,
        precision="float64",
        real_bytes=8,
        communicator="serial",
        supports_uniform=True,
        supports_amr=True,
        supports_mpi=False,
        supports_gpu=False,
        supports_stride=True,
        supports_partial_imex_mask=False,
        supports_named_fields=True,
        supports_custom_communicator=False,
        native_entrypoints=("pops_install",),
    )


def test_artifact_manifest_round_trip_from_dict():
    manifest = _manifest()
    encoded = manifest.to_dict()
    rebuilt = CompiledArtifactManifest.from_dict(encoded)

    assert rebuilt == manifest
    assert rebuilt.to_dict() == encoded
    assert rebuilt.dimension == 2
    assert rebuilt.real_bytes == 8
    assert rebuilt.supports_custom_communicator is False


def test_artifact_manifest_refuses_unknown_or_missing_semantic_fields():
    encoded = _manifest().to_dict()
    unknown = copy.deepcopy(encoded)
    unknown["payload"]["invented_physics"] = "silent-default"
    with pytest.raises(ValueError, match="unknown field"):
        CompiledArtifactManifest.from_dict(unknown)

    missing = copy.deepcopy(encoded)
    del missing["payload"]["precision"]
    with pytest.raises(ValueError, match="missing required field"):
        CompiledArtifactManifest.from_dict(missing)
