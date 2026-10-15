"""The public pipeline retains distinct State/Field scientific storage schemas."""
import pops
import numpy as np

from examples.api_v02.continuation_regrid_auxiliary import (
    build_continuation_regrid_auxiliary, initial_auxiliary_values,
)
from pops.codegen.program_codegen import emit_cpp_program
from pops.codegen.program_models import ProgramModelGraph


def test_auxiliary_history_and_explicit_target_allocation_resolve_and_emit():
    spec, providers = build_continuation_regrid_auxiliary()
    resolved = pops.resolve(pops.validate(spec.case), layout=spec.layout, layout_providers=providers)
    inputs = initial_auxiliary_values(spec, resolved)
    key, scalar = next(iter(inputs.items()))
    assert (key.space_name, key.component) == ("previous", "q_previous")
    initial = next(iter(spec.initial_values.values()))
    np.testing.assert_array_equal(scalar, initial[0])
    assert scalar.shape == (8, 16)
    metadata = next(value.attrs["continuation_forest"] for value in resolved.time._values
                    if value.op == "post_synchronization")
    members = dict(metadata["level_schema"]["read"]["members"])
    assert len(members["state"]["space"].components) == 2
    assert members["previous"]["vtype"] == "scalar_field"
    assert len(members["previous"]["space"].components) == 1
    assert tuple(metadata["level_schema"]["retained_fields"]) == (("previous",),)
    graph = ProgramModelGraph.from_resolved_blocks(resolved.blocks, scalar_layout_plan=resolved.layout_plan)
    source = emit_cpp_program(resolved.time, model_graph=graph, target="amr_system")
    assert source.count("ctx.allocate_field_storage(") == 1
    assert source.count("ctx.import_field_bundle(") == 2
