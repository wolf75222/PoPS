"""History retains the native origin while generated Q consumes its numeric value."""
import pytest
import pops
from pops.codegen.module_lowering import lower_and_validate
from pops.codegen.program_codegen import emit_cpp_program
from pops.codegen.program_emit_transform_obligation import emit_physical_input_guards
from pops.time._program.traversal import all_ops
from tests.python.support.nonlinear_representation_history import build


@pytest.mark.parametrize("reject", [False, True])
def test_history_keeps_the_coordinate_and_executes_its_q(reject):
    case, layout, _, _ = build(reject_after_first=reject)
    plan = pops.resolve(pops.validate(case), layout=layout)
    emitter, _ = lower_and_validate(plan.blocks[0].model)
    source = emit_cpp_program(plan.time, model=emitter)
    assert source.count("ctx.own_field_representation(") == 1
    # The residual also emits Q; the two top-level Q evaluations consume current
    # and retained coordinates through the same generated discharge path.
    assert source.count("ctx.own_discharged_field_representation(") == 3
    assert '"retained.physical.u"' in source
    assert '"retained.physical.v"' in source
    assert "coordinates" in source and "store_history" in source
    store = next(value for value in all_ops(plan.time) if value.op == "store_history")
    # Retention transports the coordinate obligation; it does not publish a
    # physical state. Its native History producer enforces the metadata budget.
    assert emit_physical_input_guards(store, {store.inputs[0].id: "raw_coordinate"}) == []
    if reject:
        assert "reject_after_history_write" in source
