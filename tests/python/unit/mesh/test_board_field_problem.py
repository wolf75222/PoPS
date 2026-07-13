"""The final Model field surface has no transitional solve/problem/vector aliases."""
from __future__ import annotations

import pytest

from pops.physics import Model


@pytest.mark.parametrize("legacy_name", ("solve_field", "field_problem", "vector_field"))
def test_legacy_model_field_verbs_are_absent(legacy_name: str) -> None:
    model = Model("final-field-surface")

    assert not hasattr(model, legacy_name)
    with pytest.raises(AttributeError):
        getattr(model, legacy_name)


def test_model_keeps_only_the_explicit_field_authoring_families() -> None:
    model = Model("final-field-families")

    assert callable(model.field)
    assert callable(model.field_operator)
    assert callable(model.vector)
