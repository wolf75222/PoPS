"""An AMR compile result carries every block directly on the exact artifact."""
from __future__ import annotations

import pytest

pops = pytest.importorskip("pops")

from pops.codegen._compiled_artifact import CompiledSimulationArtifact  # noqa: E402

from _typed_artifact_fixture import artifact_fixture  # noqa: E402


def test_amr_artifact_carries_one_whole_system_program_and_is_multiblock():
    artifact = artifact_fixture(target="amr_system", block_names=("ions", "electrons"))

    assert type(artifact) is CompiledSimulationArtifact
    assert artifact.program is artifact.layout_programs[0].program
    assert artifact.layout_programs[0].block_names == ("ions", "electrons")
    assert artifact.target == "amr_system"
    assert pops.inspect(artifact.layout)["amr_report"]["layout"] == "amr"
    assert tuple(block.name for block in artifact.blocks) == ("ions", "electrons")
    assert tuple(block.model.name for block in artifact.blocks) == ("ions", "electrons")
    artifact.verify()


def test_amr_artifact_reports_program_and_every_declared_block():
    artifact = artifact_fixture(target="amr_system", block_names=("ions", "electrons"))

    assert artifact.so_path == "/tmp/program.so"
    assert {block.name: block.model.so_path for block in artifact.blocks} == {
        "ions": "/tmp/ions.so",
        "electrons": "/tmp/electrons.so",
    }
    report = artifact.inspect()
    assert {row["name"] for row in report.blocks} == {"ions", "electrons"}
    assert report.artifacts["so_path"] == "/tmp/program.so"
    assert tuple(report.artifacts["so_paths"].values()) == ("/tmp/program.so",)
    assert artifact.requirements().constraints["layout"] == "amr"
    assert set(artifact.manifest().blocks) == {"ions", "electrons"}
    assert set(artifact.arguments().instances) == {"ions", "electrons"}
    manifest = artifact.manifest()
    assert manifest.supports_uniform is True and manifest.supports_amr is True
    assert not hasattr(artifact, "capability_matrix")
    report_rows = report.capabilities["routes"]
    manifest_rows = [row.to_dict() for row in manifest.capability_matrix().rows]
    assert report_rows == manifest_rows


@pytest.mark.parametrize("target", ["system", "amr_system"])
def test_single_layout_artifact_cannot_omit_the_compiled_program(target):
    artifact = artifact_fixture(target=target)
    with pytest.raises(
        ValueError,
        match="layout_programs must cover every resolved layout exactly once",
    ):
        CompiledSimulationArtifact(plan=artifact.plan, program=None, blocks=artifact.blocks)
