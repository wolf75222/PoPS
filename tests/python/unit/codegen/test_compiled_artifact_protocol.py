"""The public compile result is one exact concrete artifact, not a structural protocol."""
from __future__ import annotations

import pytest

pytest.importorskip("pops")

from pops.codegen.compiled_artifact import CompiledSimulationArtifact  # noqa: E402

from _typed_artifact_fixture import artifact_fixture  # noqa: E402


def test_exact_artifact_names_the_complete_inspection_surface():
    for method in ("inspect", "requirements", "manifest", "arguments", "capability_matrix"):
        assert callable(getattr(CompiledSimulationArtifact, method, None))


def test_compiled_phase_has_one_exact_result_type():
    artifact = artifact_fixture()
    assert type(artifact) is CompiledSimulationArtifact
    assert artifact.so_path == "/tmp/program.so"
    assert [row["name"] for row in artifact.inspect().blocks] == ["fluid"]
    artifact.verify()


def test_top_level_reexports_only_the_exact_artifact_type():
    import pops

    assert pops.CompiledSimulationArtifact is CompiledSimulationArtifact
    assert not hasattr(pops, "Compiled" + "Artifact")
