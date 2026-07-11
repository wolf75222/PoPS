"""ADC-654: full Problem identity is distinct from compile-artifact identity."""
from __future__ import annotations

import pytest

pops = pytest.importorskip("pops", exc_type=ImportError)

from pops.math import Integer, Real  # noqa: E402
from pops.model import Module  # noqa: E402
from pops.params import (  # noqa: E402
    ConstParam,
    ParamProvenance,
    Positive,
    RuntimeParam,
)


def _case_snapshot(declaration):
    problem = pops.Problem(name="artifact-parameter-case")
    authored_handle = problem.param(declaration)
    return problem.freeze(), authored_handle


def _model_snapshot(declaration):
    model = Module("artifact-parameter-model")
    model.param(declaration)
    problem = pops.Problem(name="artifact-model-case").block("fluid", physics=model)
    return problem.freeze()


def test_runtime_default_value_or_absence_changes_full_hash_not_artifact_hash():
    one, _ = _case_snapshot(RuntimeParam("alpha", default=1.0))
    two, _ = _case_snapshot(RuntimeParam("alpha", default=2.0))
    required, _ = _case_snapshot(RuntimeParam("alpha"))

    assert len({one.hash, two.hash, required.hash}) == 3
    assert one.artifact_hash == two.artifact_hash == required.artifact_hash


def test_report_only_parameter_provenance_changes_full_hash_not_artifact_hash():
    first, _ = _case_snapshot(RuntimeParam(
        "alpha", default=1.0,
        provenance=ParamProvenance("input-a", metadata={"line": 10})))
    second, _ = _case_snapshot(RuntimeParam(
        "alpha", default=1.0,
        provenance=ParamProvenance("input-b", metadata={"line": 20})))

    assert first.hash != second.hash
    assert first.artifact_hash == second.artifact_hash


def test_const_value_remains_in_artifact_identity():
    first, _ = _case_snapshot(ConstParam("gamma", 1.4))
    second, _ = _case_snapshot(ConstParam("gamma", 1.6))

    assert first.hash != second.hash
    assert first.artifact_hash != second.artifact_hash


def test_model_manifest_uses_the_same_parameter_artifact_projection():
    runtime_one = _model_snapshot(RuntimeParam("alpha", default=1.0))
    runtime_two = _model_snapshot(RuntimeParam("alpha", default=2.0))
    const_one = _model_snapshot(ConstParam("alpha", 1.0))
    const_two = _model_snapshot(ConstParam("alpha", 2.0))

    assert runtime_one.hash != runtime_two.hash
    assert runtime_one.artifact_hash == runtime_two.artifact_hash
    assert const_one.artifact_hash != const_two.artifact_hash


def test_runtime_abi_type_domain_storage_and_qualified_handle_are_in_projection():
    real, authored = _case_snapshot(RuntimeParam(
        "count", dtype=Real, default=1.0, domain=Positive(), unit="1/s"))
    integer, _ = _case_snapshot(RuntimeParam(
        "count", dtype=Integer, default=1, domain=Positive(), unit="1/s"))

    assert real.artifact_hash != integer.artifact_hash
    artifact = real.artifact_to_dict()["payload"]
    row = artifact["params"]["count"]
    assert row["kind"] == "runtime"
    assert row["dtype"] == "Real"
    assert row["unit"] == "1/s"
    assert row["storage"] == "runtime_slot"
    assert row["domain"]["kind"] == "positive"
    assert "default" not in row
    assert "provenance" not in row
    assert row["handle"]["kind"] == "parameter"
    assert row["handle"]["local_id"] == "count"
    assert row["qid"] != authored.qualified_id  # the authoring capability never enters snapshots
    assert artifact["handles"]["params"][0]["$handle"]["qualified_id"] == row["qid"]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
