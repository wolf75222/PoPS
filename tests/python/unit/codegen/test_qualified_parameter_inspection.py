"""Independent block parameters remain distinct through reports and manifest creation."""

from dataclasses import replace
from types import SimpleNamespace

import pops
import pytest

from pops.codegen import _artifact_models, inspect_compiled
from pops.codegen._compiler_lowering import require_compiler_lowering
from pops.model.bind_schema import BindSchema
from tests.python.support.local_product_operator_case import make_case
from tests.python.unit.codegen._typed_artifact_fixture import artifact_fixture


@pytest.mark.parametrize("reverse", (False, True))
def test_distinct_homonymous_defaults_survive_arguments_manifest_and_memory_metadata(
        monkeypatch, reverse):
    case, _, _ = make_case(reverse=reverse)
    pops.validate(case)
    schema = BindSchema.from_problem(case)
    names = tuple(case.blocks())
    specifications = dict(case._blocks.items())
    artifact = artifact_fixture(block_names=names, bind_schema=schema)
    original_rows = _artifact_models.artifact_model_metadata(artifact)
    metadata = tuple(replace(row, params=dict(require_compiler_lowering(
        specifications[row.block_name]["model"]).emit_model.params)) for row in original_rows)
    assert metadata[0].params["gain"] != metadata[1].params["gain"]
    monkeypatch.setattr(inspect_compiled, "_artifact_model_metadata", lambda _: metadata)
    monkeypatch.setattr(_artifact_models, "artifact_model_metadata", lambda _: metadata)

    arguments = artifact.arguments()
    expected = {slot.qid: slot.to_dict()["declaration"]["default"] for slot in schema.slots}
    assert len(expected) == 2
    assert {key: value["default"] for key, value in arguments.params.items()} == expected
    assert "gain" not in arguments.params
    assert set(artifact.manifest().params_runtime) == set(expected)
    aggregate_params = _artifact_models.aggregate_model_metadata(artifact)[2]
    assert {key: value["default"] for key, value in aggregate_params.items()} == expected


def test_low_level_unqualified_conflict_is_still_refused():
    rows = (SimpleNamespace(block_name="left", params={"gain": 1}),
            SimpleNamespace(block_name="right", params={"gain": 2}))
    with pytest.raises(ValueError, match="conflicting parameter metadata"):
        inspect_compiled._parameter_arguments(SimpleNamespace(bind_schema=None), rows)
