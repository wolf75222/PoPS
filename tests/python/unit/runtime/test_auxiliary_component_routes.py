"""Uniform and AMR auxiliary uploads are ComponentKey-only."""
from __future__ import annotations

import numpy as np
import pytest

from pops.model.provider_pack import ComponentKey
from pops.runtime._system_aux_state import _SystemAuxState
from pops.runtime._amr_system_aux_state import _AmrSystemAuxState


class _Native:
    def __init__(self) -> None:
        self.staged = None

    def stage_auxiliary_input(self, *values):
        self.staged = values

    def auxiliary_component(self, *values):
        return values


class _Runtime(_SystemAuxState):
    def __init__(self) -> None:
        self._s = _Native()


class _AmrRuntime(_AmrSystemAuxState):
    def __init__(self) -> None:
        self._s = _Native()


def test_stage_auxiliary_input_routes_only_an_exact_component_key() -> None:
    runtime = _Runtime()
    key = ComponentKey("model/electron", "aux", "material", "temperature")
    runtime.stage_auxiliary_input(key, np.array([[1.0, 2.0]]))

    assert runtime._s.staged[:4] == (
        "model/electron", "aux", "material", "temperature",
    )
    assert runtime._s.staged[4].tolist() == [1.0, 2.0]
    assert runtime.auxiliary_component(key) == runtime._s.staged[:4]


def test_auxiliary_runtime_refuses_a_bare_name_or_incomplete_mapping() -> None:
    runtime = _Runtime()
    with pytest.raises(TypeError, match="ComponentKey"):
        runtime.stage_auxiliary_input("temperature", [1.0])
    with pytest.raises(TypeError, match="owner_qid"):
        runtime.stage_auxiliary_input({"component": "temperature"}, [1.0])


def test_amr_auxiliary_runtime_uses_the_same_exact_component_key_contract() -> None:
    runtime = _AmrRuntime()
    key = ComponentKey("model/electron", "aux", "material", "temperature")
    values = np.array([[1.0, 2.0]])

    runtime.stage_auxiliary_input(key, values)

    assert runtime._s.staged[:4] == (
        "model/electron", "aux", "material", "temperature",
    )
    assert runtime._s.staged[4].shape == values.shape
    with pytest.raises(TypeError, match="ComponentKey"):
        runtime.stage_auxiliary_input("legacy-component", values)
