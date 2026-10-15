"""Target allocation authoring contracts; native regrid reception is separate."""
from types import SimpleNamespace

import pytest

from pops.codegen.program_emit_field_representation import emit_field_representation
from pops.fields.spatial_interaction import CellMidpoint
from pops.model.spaces import StateSpace
from pops.time import Program
from pops.time._program.detach import detach_compiled_program
from pops.time.field_representation import (
    validate_field_representation,
    validate_field_representation_program,
)
from pops.time.value_signatures import same_signature, value_signature
from typed_program_support import typed_state


def fixture():
    p = Program("target_field_storage")
    state = typed_state(p, "owner", space=StateSpace("q", ("q",), sampling="cell_average"))
    field = p.cell_field(state, quadrature=CellMidpoint())
    # This source-only fixture enables the closed regrid authoring region.
    # It cannot issue a native FieldTopology capability.
    p.__dict__["_continuation_regrid_authoring"] = {"blocks": (field.block,), "dimension": 2}
    target = p.regrid_topology(field, level=p.int64(1), side="target", dimension=2)
    return p, field, target


def test_allocation_retains_scientific_type_and_detaches_without_distributed_policy():
    p, field, target = fixture()
    initial = p.float64(2.5)
    result = p.scalar_field(like=field, topology=target, initial=initial)
    assert result.op == "field_storage_allocate"
    assert result.inputs == (field, target, initial)
    assert same_signature(value_signature(result), value_signature(field))
    assert result.attrs["ncomp"] == len(field.space.components)
    assert "coeffs" not in result.attrs
    assert "storage_bytes" not in result.attrs
    validate_field_representation_program(p)
    detached = detach_compiled_program(p)
    validate_field_representation_program(detached)
    assert detached._ir_hash() == p._ir_hash()


def test_allocation_lowering_owns_the_actual_native_target_storage():
    p, field, target = fixture()
    initial = p.float64(-3.5)
    result = p.scalar_field(like=field, topology=target, initial=initial)
    variables = {field.id: "old_field", target.id: "target_view", initial.id: "initial"}
    lines = []
    emit_field_representation(p, result, variables, None, lines, [], {field.block: 4}, {}, "amr_system")
    code = "\n".join(lines)
    assert "ctx.allocate_field_storage(old_field, target_view, initial)" in code
    assert "target_view.program_block() != 4" in code
    assert "target_view.components() != 1" in code
    assert "static_assert(pops::kNativeDimension == 2" in code
    assert variables[result.id].startswith("(*field_representation_")
    assert "distributed_registry" not in code
    assert "bind_field_provenance" not in code
    assert "scalar_scratch" not in code
    with pytest.raises(ValueError, match="AMR regrid"):
        emit_field_representation(p, result, variables, None, [], [], {field.block: 4}, {}, "system")


def test_allocation_requires_exact_target_and_component_width_atomically():
    p, field, target = fixture()
    initial = p.float64(0)
    source = p.regrid_topology(field, level=p.int64(0), side="source", dimension=2)
    count = len(p._values)
    with pytest.raises(TypeError, match="target topology"):
        p.scalar_field(like=field, topology=source, initial=initial)
    assert len(p._values) == count
    for width in (0, 2, True, 1.0):
        with pytest.raises(ValueError, match="component shape"):
            p.scalar_field(like=field, topology=target, initial=initial, ncomp=width)
        assert len(p._values) == count


def test_allocation_requires_owned_float64_and_explicit_target():
    p, field, target = fixture()
    foreign = Program("foreign").float64(0)
    for initial in (None, 0., p.int64(0), foreign):
        with pytest.raises((ValueError, TypeError)):
            p.scalar_field(like=field, topology=target, initial=initial)
    with pytest.raises(ValueError, match="target topology"):
        p.scalar_field(like=field, initial=p.float64(0))
    # Existing scratch allocation remains a separate, unchanged route.
    assert p.scalar_field(like=field).op == "scalar_field"


def test_allocation_rejects_other_scientific_owner_and_detached_width_tampering():
    p, field, target = fixture()
    other = p.cell_field(typed_state(p, "other", space=StateSpace("q", ("q",),
                         sampling="cell_average")), quadrature=CellMidpoint())
    initial = p.float64(0)
    with pytest.raises(TypeError, match="exact topology field"):
        p.scalar_field(like=other, topology=target, initial=initial)
    result = p.scalar_field(like=field, topology=target, initial=initial)
    forged = SimpleNamespace(op=result.op, inputs=result.inputs,
        attrs={**result.attrs, "ncomp": 2}, vtype=result.vtype, space=result.space,
        block=result.block, state_ref=result.state_ref, field_context=result.field_context,
        point=result.point)
    with pytest.raises(TypeError, match="component storage"):
        validate_field_representation(forged)
    forged.attrs = {**result.attrs, "scalar_storage_component": ("grafted", "component")}
    with pytest.raises(TypeError, match="storage authority"):
        validate_field_representation(forged)
