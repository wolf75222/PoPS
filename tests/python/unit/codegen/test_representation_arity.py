from __future__ import annotations

import pytest

from pops.physics._model import HyperbolicModel


def test_generated_model_refuses_unequal_primitive_and_conservative_arities() -> None:
    model = HyperbolicModel("unequal_representation_arities")
    first, second = model.conservative_vars("first", "second")
    model.set_flux([first, second], [first, second])
    model.set_eigenvalues([0.0], [0.0])
    model.set_primitive_state(first)
    model.set_conservative_from([first, first])

    with pytest.raises(
        ValueError,
        match=(
            r"primitive and conservative states must have equal arity "
            r"\(got 1 primitive and 2 conservative components\)"
        ),
    ):
        model.emit_cpp_brick()
