#!/usr/bin/env python3
"""ADC-556: ``m.solve_field(equation=, outputs=, solver=)`` is the single field-solve facade, and
its result is a typed :class:`pops.physics.FieldsHandle` -- an ``OperatorHandle`` of kind
``"field_operator"`` with STRUCTURED outputs (``fields.outputs.E``) and a callable that lowers to a
per-stage FieldContext value.

This pins:
  - solve_field returns a FieldsHandle that IS an OperatorHandle (kind field_operator);
  - outputs are reachable by attribute AND item, and iterate like the dict (backward compatible);
  - an unknown output raises a structured error naming the known handles;
  - calling the handle with a Program State value lowers to solve_fields (FieldContext-tagged);
  - a bare call without a Program value is refused;
  - solve_field UNIFIES with field_problem: it also records the inspectable PoissonProblem.

Pure Python: only pops.physics / pops.math / pops.fields / pops.model / pops.time are needed.
"""
import sys

import pytest

from pops.model.handles import OperatorHandle
from pops.physics.board_handles import FieldOutputs, FieldsHandle


def _board_with_field():
    physics = pytest.importorskip("pops.physics")
    from pops.math import grad, laplacian
    m = physics.Model("plasma")
    U = m.state("U", components=["rho", "mx", "my"],
                roles={"rho": "density", "mx": "momentum_x", "my": "momentum_y"})
    rho, mx, my = U
    phi = m.field("phi")
    return m, rho, phi, laplacian, grad


def test_field_outputs_attribute_and_item_access():
    outs = FieldOutputs({"E": "grad_phi", "phi": "phi"})
    assert outs.E == "grad_phi"
    assert outs["phi"] == "phi"
    assert "E" in outs and len(outs) == 2
    assert dict(outs.items()) == {"E": "grad_phi", "phi": "phi"}


def test_field_outputs_unknown_raises_naming_known():
    outs = FieldOutputs({"E": "grad_phi"})
    with pytest.raises(AttributeError) as exc:
        _ = outs.B
    assert "B" in str(exc.value) and "E" in str(exc.value)
    with pytest.raises(KeyError):
        _ = outs["B"]


def test_fields_handle_is_a_typed_operator_handle():
    h = FieldsHandle("phi_solve", outputs={"E": "grad"}, solver=None)
    assert isinstance(h, OperatorHandle)
    assert h.kind == "field_operator"
    assert h.name == "phi_solve"
    assert isinstance(h.outputs, FieldOutputs)
    assert h.outputs.E == "grad"


def test_fields_handle_call_lowers_to_solve_fields():
    from pops.time.program import Program
    P = Program("p")
    U = P.state("plasma")
    h = FieldsHandle("phi_solve", outputs={"E": "grad"}, solver=None)
    v = h(U)
    assert v.vtype == "fields" and v.op == "solve_fields"
    # ADC-588 tag rides through: the value carries a FieldContext.
    assert v.field_context.block == "plasma"


def test_fields_handle_bare_call_refused():
    h = FieldsHandle("phi_solve", outputs={}, solver=None)
    with pytest.raises(ValueError) as exc:
        h(42)
    assert "field operator" in str(exc.value)


def test_solve_field_returns_typed_handle_and_unifies_with_field_problem():
    fields = pytest.importorskip("pops.fields")
    m, rho, phi, laplacian, grad = _board_with_field()
    h = m.solve_field("poisson", equation=(-laplacian(phi) == rho),
                      outputs={"phi": phi, "grad_x": grad(phi).x}, solver="geometric_mg")
    assert isinstance(h, FieldsHandle) and h.kind == "field_operator"
    assert h.outputs.phi is phi
    # Unified with field_problem: the inspectable descriptor is recorded on the SAME call.
    assert "poisson" in m._field_problems
    assert isinstance(m._field_problems["poisson"], fields.PoissonProblem)
    assert m._field_solvers["poisson"] == "geometric_mg"


def test_solve_field_rejects_non_equation_and_non_laplacian():
    m, rho, phi, laplacian, grad = _board_with_field()
    with pytest.raises(TypeError):
        m.solve_field("bad", equation="not an equation")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
