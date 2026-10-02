"""SOURCE_ONLY exact authoring/identity refusals; no Native readiness claim."""
import sys
import pytest
import pops
from pops.amr import AMRExecution, AcceptedHaloPreparation
from pops.identity import make_identity
from tests.python.support.tag_selection_case import build


def test_default_execution_retains_exact_legacy_identity_shape():
    assert AMRExecution.synchronous().to_data() == {
        "schema_version": 2, "authority_type": "amr_execution", "mode": "synchronous", "relations": []}


@pytest.mark.parametrize("width", [True, 0, -1, (), (1, True), (1, -1), [1, 1]])
def test_invalid_halo_extent_is_refused_at_authoring(width):
    with pytest.raises(ValueError): AcceptedHaloPreparation(cells=width)


@pytest.mark.parametrize("mode", ["synchronous", "subcycled"])
def test_opted_effect_and_ranked_extent_enter_execution_identity(mode):
    plain = AMRExecution(mode)
    prepared = AMRExecution(mode, accepted_halo=AcceptedHaloPreparation(cells=(1, 2)))
    data = prepared.to_data()
    assert data["schema_version"] == 3
    assert data["accepted_halo"] == {"schema_version": 1, "effect": "prepare_accepted_halo",
        "point_authority": "candidate_accepted_clock", "cells": (1, 2), "components": "all_state_components"}
    assert make_identity("amr-execution", plain.to_data()) != make_identity("amr-execution", data)
    with pytest.raises(TypeError): AMRExecution(mode, accepted_halo={"cells": 1})


@pytest.mark.parametrize("shape", [(8, 8), (8, 12)])
@pytest.mark.parametrize("buffer", [0, 1])
@pytest.mark.parametrize("transfer", ["linear", "injection"])
def test_actual_public_stationary_case_resolves_with_explicit_numerical_effect(shape, buffer, transfer):
    case, layout = build(shape, buffer, transfer)
    resolved = pops.resolve(pops.validate(case), layout=layout)
    assert resolved is not None
    assert layout.runtime_layout_data()["execution"]["accepted_halo"]["effect"] == "prepare_accepted_halo"
    assert not any(name == "_pops" or name.endswith("._pops") for name in sys.modules)


def test_actual_program_exports_its_typed_primary_clock_for_topology_relay():
    import json
    from pops.codegen.program_emit_amr import _emit_checkpoint_shape_metadata
    case, _ = build((8, 12), 1, "injection")
    program = case._time
    manifest = program.temporal_manifest()
    source = _emit_checkpoint_shape_metadata(program)
    expected = ('extern "C" const char* pops_program_checkpoint_primary_clock_identity() {\n'
                '  return %s;\n}\n' % json.dumps(manifest["primary_clock"]))
    assert expected in source
    assert manifest["primary_clock"] == program.clock.qualified_id
    assert not any(name == "_pops" or name.endswith("._pops") for name in sys.modules)
