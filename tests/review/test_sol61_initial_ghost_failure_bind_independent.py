"""Non-author bind-metadata gate; run with the exact intended Core Source selected."""
import pops
from pops.mesh.boundaries.compiled_plan import CompiledBoundaryPlan
from tests.python.support.initial_ghost_failure_component import load_component,InitialFailureBoundary
from tests.python.support.initial_field_ghost_native_case import build


def test_genuine_initial_failure_package_reaches_detached_bind_metadata(tmp_path):
    component=load_component(tmp_path/"component")
    case,layout=build(boundary_composer=lambda base:InitialFailureBoundary(base,component))
    plan=pops.resolve(pops.validate(case),layout=layout,components=(component,))
    compiled=CompiledBoundaryPlan.from_resolved(plan.blocks[0].numerics.boundaries[0])
    data=compiled.runtime_boundary_data({})
    assert data["faces"][0]["type"]=="external"
    assert len(data["component_regions"])==1
