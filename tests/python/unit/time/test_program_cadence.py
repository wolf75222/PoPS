"""Public immutable Program cadence and its bind-time transport."""
from __future__ import annotations

import pytest

from pops.identity.semantic import semantic_identity_of
from pops.runtime._program_cadence_install import install_program_cadence
from pops.time import Program
from pops.time._cadence import ProgramCadence
from pops.time._program.detach import detach_compiled_program


def test_program_cadence_is_single_declaration_exact_positive_and_identity_bearing():
    baseline = Program("baseline")
    configured = Program("configured")

    assert "cadence" not in baseline._serialize(include_provenance=False)
    assert "cadence" not in baseline.to_graph().to_data()
    assert configured.cadence(substeps=2, stride=3) is configured
    assert configured.cadence_contract().to_data() == {
        "schema_version": 1,
        "substeps": 2,
        "stride": 3,
    }
    assert configured._serialize(include_provenance=False)["cadence"] == {
        "schema_version": 1,
        "substeps": 2,
        "stride": 3,
    }
    assert configured.to_graph().to_data()["cadence"] == {
        "schema_version": 1,
        "substeps": 2,
        "stride": 3,
    }
    assert configured._ir_hash() != baseline._ir_hash()
    assert configured.to_graph().graph_hash != baseline.to_graph().graph_hash
    assert semantic_identity_of(program=configured) != semantic_identity_of(program=baseline)

    with pytest.raises(ValueError, match="only once"):
        configured.cadence(stride=4)

    for name, kwargs in (
        ("bool substeps", {"substeps": True, "stride": 1}),
        ("bool stride", {"substeps": 1, "stride": False}),
        ("float stride", {"substeps": 1, "stride": 2.0}),
    ):
        candidate = Program(name)
        with pytest.raises(TypeError, match="exact int"):
            candidate.cadence(**kwargs)
    for name, kwargs in (
        ("zero substeps", {"substeps": 0, "stride": 1}),
        ("zero stride", {"substeps": 1, "stride": 0}),
    ):
        candidate = Program(name)
        with pytest.raises(ValueError, match=">= 1"):
            candidate.cadence(**kwargs)
    with pytest.raises(TypeError, match="exact v1 schema"):
        ProgramCadence.from_data(
            type("CadenceDict", (dict,), {})(
                schema_version=1, substeps=1, stride=2
            )
        )
    with pytest.raises(ValueError, match="schema_version"):
        ProgramCadence.from_data(
            {"schema_version": True, "substeps": 1, "stride": 2}
        )


def test_compiled_detachment_preserves_cadence_and_freeze_refuses_mutation():
    authored = Program("detached-cadence").cadence(stride=3)
    detached = detach_compiled_program(authored)

    assert detached is not authored
    assert detached.cadence_contract() == authored.cadence_contract()
    assert detached._ir_hash() == authored._ir_hash()
    with pytest.raises(RuntimeError, match="frozen"):
        detached.cadence(stride=4)


class _CadenceEngine:
    def __init__(self, *, lie: bool = False) -> None:
        self.calls = []
        self.substeps = 1
        self.stride = 1
        self.lie = lie

    def set_program_cadence(self, substeps, stride):
        self.calls.append((substeps, stride))
        self.substeps = substeps
        self.stride = stride

    def program_substeps(self):
        return self.substeps

    def program_stride(self):
        return self.stride + int(self.lie)


def test_bind_installs_and_authenticates_only_the_frozen_compiled_cadence():
    authored = Program("install-cadence").cadence(substeps=2, stride=3)
    detached = detach_compiled_program(authored)
    engine = _CadenceEngine()

    install_program_cadence(engine, detached)

    assert engine.calls == [(2, 3)]
    with pytest.raises(TypeError, match="frozen compiled Program"):
        install_program_cadence(_CadenceEngine(), authored)
    with pytest.raises(RuntimeError, match="differs from the compiled contract"):
        install_program_cadence(_CadenceEngine(lie=True), detached)
