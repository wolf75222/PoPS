"""The public compile result is one exact concrete artifact, not a structural protocol."""
from __future__ import annotations

from dataclasses import replace

import pytest

pytest.importorskip("pops")

from pops.codegen._compiled_artifact import CompiledSimulationArtifact  # noqa: E402

from _typed_artifact_fixture import artifact_fixture  # noqa: E402


def test_exact_artifact_names_the_complete_inspection_surface():
    # Capabilities are part of ``inspect()``/``manifest()``; the artifact deliberately does not
    # grow a second capability-inspection facade.
    for method in ("inspect", "requirements", "manifest", "arguments"):
        assert callable(getattr(CompiledSimulationArtifact, method, None))


def test_compiled_phase_has_one_exact_result_type():
    artifact = artifact_fixture()
    assert type(artifact) is CompiledSimulationArtifact
    assert artifact.so_path == "/tmp/program.so"
    assert [row["name"] for row in artifact.inspect().blocks] == ["fluid"]
    artifact.verify()


def test_compiled_artifact_stays_at_its_codegen_owner():
    import pops

    assert CompiledSimulationArtifact.__module__ == "pops.codegen._compiled_artifact"
    assert not hasattr(pops, "Compiled" + "SimulationArtifact")


def test_amr_compiled_record_rejects_missing_real_authorities():
    record = artifact_fixture(target="amr_system").plan

    # Start from one genuinely resolved public AMR plan and tamper only with authority presence.
    # No structural stand-in can accidentally make this invariant test pass.
    with pytest.raises(ValueError, match="partial AMR authority set"):
        replace(record, resolved_hierarchy=None)
    with pytest.raises(ValueError, match="AMR target has no complete AMR authority set"):
        replace(
            record,
            resolved_hierarchy=None,
            amr_transfer=None,
            bootstrap_plan=None,
            amr_execution=None,
        )
