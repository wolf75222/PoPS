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
from pops.problem import Case


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
    m, rho, phi, laplacian, grad = _board_with_field()
    h = m.solve_field("phi_solve", equation=(-laplacian(phi) == rho),
                      outputs={"E": grad(phi).x}, solver=None)
    assert isinstance(h, OperatorHandle)
    assert h.kind == "field_operator"
    assert h.name == "phi_solve"
    assert isinstance(h.outputs, FieldOutputs)
    assert h.outputs.E.field is phi


def test_fields_handle_call_lowers_to_solve_fields():
    from pops.time.program import Program
    m, rho, phi, laplacian, grad = _board_with_field()
    h = m.solve_field("phi_solve", equation=(-laplacian(phi) == rho),
                      outputs={"E": grad(phi).x}, solver=None)
    P = Program("p").bind_operators(m.module)
    plasma = Case(name="field_solve").block("plasma", m)
    state = m.module.state_handle(m.module.state_spaces()["U"])
    U = P.state(plasma, state).n
    v = h(U)
    assert v.vtype == "fields" and v.op == "solve_fields"
    # ADC-588 tag rides through: the value carries a FieldContext.
    assert v.field_context.stage_sources == ((plasma, U.id),)


def test_fields_handle_alias_keeps_exact_declaring_owner():
    """The readable alias maps explicitly to fields_from_state without becoming name-only."""
    from pops.time.program import Program

    first, rho1, phi1, laplacian1, grad1 = _board_with_field()
    first_handle = first.solve_field(
        "phi_solve", equation=(-laplacian1(phi1) == rho1),
        outputs={"E": grad1(phi1).x}, solver=None)
    second, rho2, phi2, laplacian2, grad2 = _board_with_field()
    foreign_handle = second.solve_field(
        "phi_solve", equation=(-laplacian2(phi2) == rho2),
        outputs={"E": grad2(phi2).x}, solver=None)
    program = Program("p").bind_operators(first.module)
    plasma = Case(name="field_alias").block("plasma", first)
    declaration = first.module.state_handle(first.module.state_spaces()["U"])
    state = program.state(plasma, declaration).n

    assert first_handle(state).op == "solve_fields"
    with pytest.raises(ValueError, match="no operator registry is bound for owner"):
        foreign_handle(state)


def test_fields_handle_bare_call_refused():
    m, rho, phi, laplacian, grad = _board_with_field()
    h = m.solve_field("phi_solve", equation=(-laplacian(phi) == rho),
                      outputs={}, solver=None)
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
