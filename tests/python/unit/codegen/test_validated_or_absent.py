"""A compile result is authenticated on construction and has no post-compile check step."""
from __future__ import annotations

import pytest

pytest.importorskip("pops")

from pops.codegen.compiled_artifact import CompiledSimulationArtifact  # noqa: E402

from _typed_artifact_fixture import artifact_fixture  # noqa: E402


def test_exact_artifact_has_no_public_check_or_revalidation_alias():
    artifact = artifact_fixture()
    assert not hasattr(artifact, "check")
    assert not hasattr(artifact, "_assert_invariants")
    assert callable(artifact.verify)


def test_artifact_exists_only_after_exact_constructor_validation():
    artifact = artifact_fixture()
    artifact.verify()

    with pytest.raises(TypeError, match="exact ResolvedSimulationPlan"):
        CompiledSimulationArtifact(plan=object(), program=object(), blocks=())


def test_inspection_is_available_without_a_second_validity_phase():
    artifact = artifact_fixture()
    assert artifact.inspect().status.startswith("compiled")
    assert artifact.requirements().constraints["backend"] == "production"
    assert artifact.manifest().model_name == "program"
    assert artifact.target == "system"
