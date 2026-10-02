"""Independent detached bind metadata attacks; Source only."""
import copy
import pytest
from pops.mesh.boundaries.compiled_plan import CompiledBoundaryPlan
from tests.review.test_sol61_public_inflow_expression_route import compiled_public_ghost_plan
@pytest.mark.parametrize("mutation",["duplicate_binding","ambiguous_target_payload","wrong_state","duplicate_region","wrong_side"])
def test_detached_delegation_requires_unique_binding_and_exact_region(mutation):
    d=compiled_public_ghost_plan().canonical_identity()["compile_data"]
    if mutation=="duplicate_binding":d["component_bindings"].append(copy.deepcopy(d["component_bindings"][0]))
    if mutation=="ambiguous_target_payload":
        other=copy.deepcopy(d["component_bindings"][0]);other["component_manifest_identity"]="different-manifest";d["component_bindings"].append(other)
    if mutation=="wrong_state":d["component_region_templates"][0]["state_identity"]="foreign-state"
    if mutation=="duplicate_region":d["component_region_templates"].append(copy.deepcopy(d["component_region_templates"][0]))
    if mutation=="wrong_side":d["component_region_templates"][0]["region"]["sides"]=[1]
    with pytest.raises(ValueError,match="component value delegation"):
        CompiledBoundaryPlan(d).runtime_boundary_data({})

def test_imported_production_is_exact_current_source():
    import pops.mesh.boundaries.compiled_plan as module
    from pathlib import Path
    assert Path(module.__file__).resolve()==Path(__file__).resolve().parents[2]/"python/pops/mesh/boundaries/compiled_plan.py"

def test_existing_external_path_without_new_value_protocol_stays_available():
    d=compiled_public_ghost_plan().canonical_identity()["compile_data"]
    face=d["faces"][0];del face["value_protocol"];del face["value_delegate"]
    # Legacy explicit external metadata does not use the additive value protocol.
    bound=CompiledBoundaryPlan(d).runtime_boundary_data({})
    assert bound["faces"][0]["type"]=="external" and bound["faces"][0]["values"]==[0.,0.]
