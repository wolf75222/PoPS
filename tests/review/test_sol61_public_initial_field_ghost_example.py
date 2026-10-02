"""Pure Source example checks: execution tail is deliberately never evaluated."""
import ast
from pathlib import Path
import pops
from pops.mesh.boundaries.compiled_plan import CompiledBoundaryPlan

EXAMPLE=Path(__file__).resolve().parents[2]/"docs/tutorials/initial_field_ghost/01_public_initial_field_ghost.py"


def test_linear_example_public_model_and_real_bind_metadata_source():
    source=EXAMPLE.read_text()
    tree=ast.parse(source)
    assert not any(isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)) for node in tree.body)
    preamble=source.split("# Execution begins:",1)[0]
    namespace={"__name__":"source_example_preparation"}
    exec(compile(preamble,str(EXAMPLE),"exec"),namespace)
    assert namespace["DT"]==1/64 and namespace["ALPHA"]==1/8
    plan=pops.resolve(pops.validate(namespace["case"]),layout=namespace["layout"])
    component,=plan.component_inputs
    signature=component.component_manifest.signature["inferred_boundary_expression"]
    assert len(signature["expressions"])==2
    assert len(signature["dependencies"]["fields"])==1
    assert not signature["dependencies"]["states"]
    compiled=CompiledBoundaryPlan.from_resolved(plan.blocks[0].numerics.boundaries[0])
    data=compiled.runtime_boundary_data({})
    assert data["faces"][0]["type"]=="external"
    assert len(data["component_regions"])==1


def test_example_execution_identity_and_capture_are_not_source_receipts():
    source=EXAMPLE.read_text()
    tail=source.split("# Execution begins:",1)[1]
    assert "select_native_dimension(2)" in tail
    assert "is_relative_to(prefix)" in tail and "__abi_version__ != 8" in tail
    assert "pops.compile(resolved)" in tail and "pops.bind(artifact" in tail
    assert "resumed.restart(accepted_checkpoint)" in tail
    assert tail.count(".checkpoint(")==3
    assert "exist_ok=False" in tail
    assert "not full-grown scientific reception" in tail
