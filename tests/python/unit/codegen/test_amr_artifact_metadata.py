"""An AMR compile result carries every block directly on the exact artifact."""
from __future__ import annotations

import pytest

pops = pytest.importorskip("pops")

from pops.codegen._compiled_artifact import CompiledSimulationArtifact  # noqa: E402

from _typed_artifact_fixture import artifact_fixture  # noqa: E402


def test_amr_artifact_without_whole_system_program_is_explicit_and_multiblock():
    artifact = artifact_fixture(target="amr_system", block_names=("ions", "electrons"))

    assert type(artifact) is CompiledSimulationArtifact
    assert artifact.program is None
    assert artifact.target == "amr_system"
    assert pops.inspect(artifact.layout)["amr_report"]["layout"] == "amr"
    assert tuple(block.name for block in artifact.blocks) == ("ions", "electrons")
    assert tuple(block.model.name for block in artifact.blocks) == ("ions", "electrons")
    artifact.verify()


def test_amr_artifact_reports_aggregate_every_declared_block():
    artifact = artifact_fixture(target="amr_system", block_names=("ions", "electrons"))

    with pytest.raises(ValueError, match="aggregate artifact has no scalar so_path"):
        _ = artifact.so_path
    assert {block.name: block.model.so_path for block in artifact.blocks} == {
        "ions": "/tmp/ions.so",
        "electrons": "/tmp/electrons.so",
    }
    report = artifact.inspect()
    assert {row["name"] for row in report.blocks} == {"ions", "electrons"}
    assert report.artifacts["so_path"] is None
    assert report.artifacts["so_paths"] == {
        "block:ions": "/tmp/ions.so",
        "block:electrons": "/tmp/electrons.so",
    }
    assert artifact.requirements().constraints["layout"] == "amr"
    assert set(artifact.manifest().blocks) == {"ions", "electrons"}
    assert set(artifact.arguments().instances) == {"ions", "electrons"}
    manifest = artifact.manifest()
    assert manifest.supports_uniform is True and manifest.supports_amr is True
    assert not hasattr(artifact, "capability_matrix")
    report_rows = report.capabilities["routes"]
    manifest_rows = [row.to_dict() for row in manifest.capability_matrix().rows]
    assert report_rows == manifest_rows


def test_system_artifact_cannot_omit_the_compiled_program():
    artifact = artifact_fixture()
    with pytest.raises(
        ValueError,
        match="layout_programs must cover every and only per-layout system target",
    ):
        CompiledSimulationArtifact(plan=artifact.plan, program=None, blocks=artifact.blocks)
