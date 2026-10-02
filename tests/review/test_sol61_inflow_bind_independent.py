"""Independent detached bind metadata attacks; Source only."""
import copy
import pytest
from pops.mesh.boundaries.compiled_plan import CompiledBoundaryPlan
from tests.review.test_sol61_public_inflow_expression_route import compiled_public_ghost_plan
@pytest.mark.parametrize("mutation",["duplicate_binding","wrong_state","duplicate_region","wrong_side"])
def test_detached_delegation_requires_unique_binding_and_exact_region(mutation):
    d=compiled_public_ghost_plan().canonical_identity()["compile_data"]
    if mutation=="duplicate_binding":d["component_bindings"].append(copy.deepcopy(d["component_bindings"][0]))
    if mutation=="wrong_state":d["component_region_templates"][0]["state_identity"]="foreign-state"
    if mutation=="duplicate_region":d["component_region_templates"].append(copy.deepcopy(d["component_region_templates"][0]))
    if mutation=="wrong_side":d["component_region_templates"][0]["region"]["sides"]=[1]
    with pytest.raises(ValueError,match="component value delegation"):
        CompiledBoundaryPlan(d).runtime_boundary_data({})
