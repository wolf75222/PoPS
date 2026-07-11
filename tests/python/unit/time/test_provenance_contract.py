from __future__ import annotations

import copy

import pytest

from pops.model.ownership import OwnerPath
from pops.provenance import ProvenanceRecord, SourceSpan


def test_source_span_and_provenance_are_exact_immutable_content_addressed_data():
    span = SourceSpan(__file__, 12, 3, 12, 9)
    record = ProvenanceRecord(
        primary=span,
        owner=OwnerPath.consumer("time-step"),
        authoring_api="pops.time.Program.rhs",
        origins=(span,),
        transformation="direct",
    )

    assert SourceSpan.from_data(span.to_data()).to_data() == span.to_data()
    assert ProvenanceRecord.from_data(record.to_data()).to_data() == record.to_data()
    assert record.id.startswith("sha256:") and len(record.id) == 71
    with pytest.raises(AttributeError, match="immutable"):
        record.phase = "lowering"
    with pytest.raises(TypeError):
        record.owner_data["nodes"][0]["name"] = "changed"

    extra = copy.deepcopy(record.to_data())
    extra["legacy"] = True
    with pytest.raises(TypeError, match="exactly"):
        ProvenanceRecord.from_data(extra)
    forged = copy.deepcopy(record.to_data())
    forged["id"] = "sha256:" + "0" * 64
    with pytest.raises(ValueError, match="canonical"):
        ProvenanceRecord.from_data(forged)


def test_derive_preserves_ordered_origins_and_parents():
    owner = OwnerPath.consumer("time-step")
    first_span = SourceSpan(__file__, 31)
    second_span = SourceSpan(__file__, 32)
    first = ProvenanceRecord(
        primary=first_span, owner=owner, authoring_api="manual.first",
    )
    second = ProvenanceRecord(
        primary=second_span, owner=owner, authoring_api="manual.second",
    )

    derived = ProvenanceRecord.derive((first, second), transformation="cse")
    assert derived.primary == first_span
    assert derived.origins == (first_span, second_span)
    assert derived.parents == (first.id, second.id)
    assert derived.phase == "transform"
    assert derived.transformation == "cse"


def test_operator_source_rejects_strings_and_manifest_boundary_requires_provenance():
    from pops.model import Operator, Rate, Signature, StateSpace

    state = StateSpace("U", ("rho",))
    with pytest.raises(TypeError, match="ProvenanceRecord"):
        Operator("rhs", "local_rate", Signature((state,), Rate(state)), source="module")

    operator = Operator("rhs", "local_rate", Signature((state,), Rate(state)))
    with pytest.raises(TypeError, match="ProvenanceRecord"):
        operator.freeze()


def test_operator_manifest_round_trips_documentary_provenance():
    from pops.model import Operator, OperatorHandle, OperatorManifestEntry, Rate, Signature, StateSpace

    owner = OwnerPath.model("transport")
    state = StateSpace("U", ("rho",))
    signature = Signature((state,), Rate(state))
    provenance = ProvenanceRecord(
        primary=SourceSpan(__file__, 70), owner=owner,
        authoring_api="pops.model.Module.operator",
    )
    operator = Operator("rhs", "local_rate", signature, source=provenance)
    handle = OperatorHandle(
        "rhs", kind="local_rate", owner=owner, signature=signature,
        registered_operator_name="rhs",
    )
    entry = OperatorManifestEntry(operator, 0, handle)

    data = entry.to_dict()
    assert data["provenance"] == provenance.to_data()
    assert OperatorManifestEntry.from_dict(data, owner=owner).to_dict() == data


def test_program_serializes_provenance_but_excludes_it_from_ir_hash():
    from pops.identity.semantic import program_semantic_data
    from pops.time import Program

    first = Program("identity")
    first._scalar_binop(1, 2, "add")
    second = Program("identity")
    second._scalar_binop(1, 2, "add")

    assert first._serialize()["nodes"][0]["provenance"] != second._serialize()["nodes"][0]["provenance"]
    assert "provenance" not in first._serialize(include_provenance=False)["nodes"][0]
    assert first._ir_hash() == second._ir_hash()
    assert program_semantic_data(first) == program_semantic_data(second)
    assert first._values[0].provenance.transformation == "direct"


def test_cse_merges_duplicate_lineage_and_records_transform():
    from pops.time import Program

    program = Program("cse")
    program._scalar_binop(1, 2, "add")
    program._scalar_binop(1, 2, "add")
    original_ids = tuple(value.provenance.id for value in program._values)

    optimized = program.eliminate_common_subexpressions()
    assert len(optimized._values) == 1
    provenance = optimized._values[0].provenance
    assert provenance.transformation == "cse"
    assert provenance.parents == original_ids
    assert optimized._ir_hash() != program._ir_hash()


def test_lowering_projection_appends_lower_without_mutating_program():
    from pops.codegen.compile_provenance import lowering_provenance_data
    from pops.time import Program

    program = Program("lower")
    value = program._scalar_binop(1, 2, "add")
    original = value.provenance

    [row] = lowering_provenance_data(program)
    lowered = ProvenanceRecord.from_data(row["provenance"])
    assert row["node_id"] == value.id
    assert lowered.transformation == "lower"
    assert lowered.phase == "lowering"
    assert lowered.parents == (original.id,)
    assert value.provenance is original
