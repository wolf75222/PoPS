from __future__ import annotations

import pytest

from pops.physics import Model


def _scalar_model(*, guarded: bool) -> Model:
    model = Model("scalar")
    (conserved,) = model.conservative_vars("q")
    (primitive,) = model.primitive_vars(q=conserved)
    model.conservative_from([primitive])
    model.flux(x=[conserved], y=[conserved])
    model.eigenvalues(x=[1], y=[1])
    if guarded:
        model.recovery_admissibility(q=primitive > 0)
    return model


def test_recovery_admissibility_is_emitted_and_hashed() -> None:
    guarded = _scalar_model(guarded=True)
    plain = _scalar_model(guarded=False)

    generated = guarded._m.emit_cpp_brick(name="GuardedScalar")
    assert "bool recovery_admissible(const Prim& P, int* failing_component_)" in generated
    assert "const pops::Real q = P[0];" in generated
    assert "*failing_component_ = 0;" in generated
    assert "return false;" in generated
    assert "recovery_admissible" not in plain._m.emit_cpp_brick(name="PlainScalar")
    assert guarded._model_hash() != plain._model_hash()


def test_recovery_admissibility_rejects_ambiguous_authoring() -> None:
    model = Model("guarded_scalar")
    (conserved,) = model.conservative_vars("q")

    with pytest.raises(ValueError, match=r"primitive_vars\(\.\.\.\) first"):
        model.recovery_admissibility(q=conserved > 0)

    (primitive,) = model.primitive_vars(q=conserved)
    with pytest.raises(ValueError, match="unknown primitive components"):
        model.recovery_admissibility(density=primitive > 0)
    with pytest.raises(TypeError, match="typed symbolic Boolean expression"):
        model.recovery_admissibility(q=1)

    model.recovery_admissibility(q=primitive > 0)
    with pytest.raises(ValueError, match="policy already declared"):
        model.recovery_admissibility(q=primitive >= 0)
