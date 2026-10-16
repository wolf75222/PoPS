"""Host structural projections must become primitive device captures."""

import pytest
from pops.time import Program
from pops.codegen.program_emit_expressions import emit_pointwise_kernel
from typed_program_support import typed_state


@pytest.mark.parametrize(
    "binding",
    [
        "std::any_cast<const pops::Real&>(record.values.at(1))",
        "(typed_capability->members()[0])->real()",
        "already_primitive",
    ],
)
def test_scalar_projection_is_materialized_before_device_kernel(binding):
    p = Program("device_scalar_capture")
    q = typed_state(p, "material", state_name="U")
    scalar = p.product(factor=p.float64(2.0))["factor"]
    value = p.value("scale", scalar * q.n)
    variables = {q.n.id: "source_field", scalar.id: binding}
    source = "\n".join(
        emit_pointwise_kernel(value, variables, "output", block_index=0, status="status")
    )
    host, device = source.split("pops::for_each_cell", 1)
    expected = "const pops::Real expression_scalar_%d_%d = %s;" % (value.id, scalar.id, binding)
    assert expected in host
    assert binding not in device
    assert "expression_scalar_%d_%d" % (value.id, scalar.id) in device
    assert "Kokkos::isfinite" in device
    assert "std::any_cast" not in device and "->members()" not in device
