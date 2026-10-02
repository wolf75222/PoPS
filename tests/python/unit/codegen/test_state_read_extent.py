"""Compiler-owned cells certify effects; extension traversal is not certification."""
from pops.physics._facade import Model
from pops.codegen.module_lowering import lower_and_validate
from pops.codegen.state_read_extent import cell_state_read_extent
from pops._ir.expr import Expr, Add, Const, Var
from pops._ir.control_expr import Where, Rounded


def model():
    result = Model("state_reach")
    (rho,) = result.conservative_vars("rho")
    result.primitive_vars(rho)
    result.conservative_from([rho])
    result.flux(x=[rho], y=[0 * rho])
    result.eigenvalues(x=[1 + 0 * rho], y=[0 * rho])
    result.elliptic_field("charge", 2 * rho, aux=["potential"])
    emitter, _ = lower_and_validate(result, facade=result)
    return emitter._m


def test_actual_named_brick_certifies_valid_cell_rhs():
    m = model()
    receipt = cell_state_read_extent(m, m._elliptic_fields["charge"]["rhs"])
    assert receipt == {"schema": 1, "unit": "cells", "cells": (0, 0),
                       "authority": "compiler.cell-state-ast@1"}
    emitted = m.emit_cpp_elliptic_field("charge", "Charge")
    assert "prepared_field_rhs_read_contract_version = 1" in emitted
    assert "prepared_field_rhs_state_read_cells = 0" in emitted


def test_extension_with_empty_dependencies_is_unknown_even_inside_control():
    class Opaque(Expr):
        def __pops_ir_children__(self):
            return ()
        def deps(self):
            return set()
    m = model()
    assert cell_state_read_extent(m, Opaque()) is None
    assert cell_state_read_extent(m, Add(Const(1), Opaque())) is None
    assert cell_state_read_extent(m, Where(Const(1) > Const(0), Rounded(Const(1)), Opaque())) is None


def test_unknown_variable_and_field_reads_are_not_certified():
    m = model()
    assert cell_state_read_extent(m, Var("unbound", "cons")) is None
    assert cell_state_read_extent(m, Var("potential", "aux")) is None


def test_actual_amr_loader_carries_certified_effect_with_exact_rhs_binding():
    m = model()
    source = m.emit_cpp_native_loader(
        name="StateReach", target="amr_system", native_field_roles=({
            "kind": "rhs", "field": "tests.field", "block": "material",
            "binding_ordinal": 0, "binding_identity": "tests.binding",
            "provider_key": "charge", "coefficient": 1.0,
        },),
    )
    assert 'attachment.binding_identity = "tests.binding";' in source
    assert "attachment.rhs_read_contract_version = pops::poisson_rhs_read_contract_version" in source
    assert 'attachment.rhs_read_authority = "compiler.cell-state-ast@1";' in source
    assert "attachment.rhs_state_read_cells.fill(pops::poisson_rhs_state_read_cells" in source
