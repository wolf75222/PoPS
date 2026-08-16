from __future__ import annotations

import pytest

from pops.codegen.module_lowering import lower_and_validate
from pops.physics import Model
from tests.python.support.physics_roles import FRAME, X_AXIS, Y_AXIS


def _lowered(model: Model):
    """Resolve the public Model through its canonical Module/provider-pack route."""
    emit_model, source_module = lower_and_validate(model, facade=model)
    assert source_module is model.module
    assert type(emit_model._auxiliary_provider_pack).__name__ == "ProviderPack"
    return emit_model


def _scalar_model(*, guarded: bool) -> Model:
    model = Model("scalar", frame=FRAME)
    state = model.state("U", components=("q",))
    (primitive,) = state
    model.flux(
        "transport",
        frame=FRAME,
        state=state,
        components={X_AXIS: (primitive,), Y_AXIS: (primitive,)},
        waves={X_AXIS: (1,), Y_AXIS: (1,)},
    )
    if guarded:
        model.recovery_admissibility(q=primitive > 0)
    return model


def test_recovery_admissibility_is_emitted_and_hashed() -> None:
    guarded = _scalar_model(guarded=True)
    plain = _scalar_model(guarded=False)

    generated = _lowered(guarded)._m.emit_cpp_brick(name="GuardedScalar")
    assert "bool recovery_admissible(const Prim& P, int* failing_component_)" in generated
    assert "const pops::Real q = P[0];" in generated
    assert "*failing_component_ = 0;" in generated
    assert "return false;" in generated
    assert "recovery_admissible" not in _lowered(plain)._m.emit_cpp_brick(name="PlainScalar")
    assert _lowered(guarded)._model_hash() != _lowered(plain)._model_hash()


def test_recovery_admissibility_rejects_ambiguous_authoring() -> None:
    model = Model("guarded_scalar", frame=FRAME)

    with pytest.raises(ValueError, match=r"primitive_vars\(\.\.\.\) first"):
        model.recovery_admissibility(q=1)

    state = model.state("U", components=("q",))
    (primitive,) = state
    with pytest.raises(ValueError, match="unknown primitive components"):
        model.recovery_admissibility(density=primitive > 0)
    with pytest.raises(TypeError, match="typed symbolic Boolean expression"):
        model.recovery_admissibility(q=1)

    model.recovery_admissibility(q=primitive > 0)
    with pytest.raises(ValueError, match="policy already declared"):
        model.recovery_admissibility(q=primitive >= 0)
