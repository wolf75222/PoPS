"""Independent Source routing check; never evaluates the execution tail."""
from pathlib import Path
import pops
from pops.mesh.boundaries.compiled_plan import CompiledBoundaryPlan


def test_public_identity_renaming_preserves_inferred_dependency_and_delegate():
    root = Path(__file__).resolve().parents[2]
    source = (root / "docs/tutorials/initial_field_ghost/01_public_initial_field_ghost.py").read_text().split("# Execution begins:", 1)[0]
    identities = []
    for renamed in (False, True):
        text = source
        if renamed:
            for before, after in (("initial field ghost", "renamed domain"), ("growth with screened field", "renamed physics"), ("accepted initial Field/Ghost scientific board", "renamed case"), ("'marker'", "'another block'"), ("'phi'", "'another_potential'")):
                text = text.replace(before, after)
        ns = {"__name__": "independent_source_preparation"}
        exec(compile(text, "independent_example_preamble", "exec"), ns)
        plan = pops.resolve(pops.validate(ns["case"]), layout=ns["layout"])
        component, = plan.component_inputs
        signature = component.component_manifest.signature["inferred_boundary_expression"]
        assert signature["schema"] == "inferred-boundary-expression-component@1"
        assert len(signature["expressions"]) == 2
        assert len(signature["dependencies"]["fields"]) == 1
        assert not signature["dependencies"]["states"]
        data = CompiledBoundaryPlan.from_resolved(plan.blocks[0].numerics.boundaries[0]).runtime_boundary_data({})
        assert data["faces"][0]["type"] == "external"
        assert len(data["component_regions"]) == 1
        identities.append(component.component_manifest.manifest_digest.token)
    assert identities[0] != identities[1]
