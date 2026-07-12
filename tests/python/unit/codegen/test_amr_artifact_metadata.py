"""An AMR compile result carries every block directly on the exact artifact."""
from __future__ import annotations

import pytest

pytest.importorskip("pops")

from pops.codegen.compiled_artifact import CompiledSimulationArtifact  # noqa: E402

from _typed_artifact_fixture import artifact_fixture  # noqa: E402


def test_amr_artifact_without_whole_system_program_is_explicit_and_multiblock():
    artifact = artifact_fixture(target="amr_system", block_names=("ions", "electrons"))

    assert type(artifact) is CompiledSimulationArtifact
    assert artifact.program is None
    assert artifact.target == "amr_system"
    assert artifact.layout["kind"] == "amr"
    assert tuple(block.name for block in artifact.blocks) == ("ions", "electrons")
    assert tuple(block.model.name for block in artifact.blocks) == ("ions", "electrons")
    artifact.verify()


def test_amr_artifact_reports_aggregate_every_declared_block():
    artifact = artifact_fixture(target="amr_system", block_names=("ions", "electrons"))

    assert artifact.so_path == "/tmp/ions.so"
    assert {row["name"] for row in artifact.inspect().blocks} == {"ions", "electrons"}
    assert artifact.requirements().constraints["layout"] == "amr"
    assert set(artifact.manifest().blocks) == {"ions", "electrons"}
    assert set(artifact.arguments().instances) == {"ions", "electrons"}
    manifest = artifact.manifest()
    assert manifest.supports_uniform is True and manifest.supports_amr is True
    assert artifact.capability_matrix().to_dict() == manifest.capability_matrix().to_dict()


def test_system_artifact_cannot_omit_the_compiled_program():
    artifact = artifact_fixture()
    with pytest.raises(ValueError, match="requires a compiled program"):
        CompiledSimulationArtifact(plan=artifact.plan, program=None, blocks=artifact.blocks)
