"""The AMR Python boundary authenticates Cartesian array orientation before flattening."""
from __future__ import annotations

import numpy as np
import pytest

from pops.model.provider_pack import ComponentKey

try:
    from pops.runtime._system import AmrSystem
except ImportError as exc:  # native extension is installed by the Python integration gate
    pytest.skip("PoPS native extension unavailable: %s" % exc, allow_module_level=True)


def _runtime() -> AmrSystem:
    return AmrSystem(
        shape=(6, 4),
        lower=(0.0, 0.0),
        upper=(3.0, 1.0),
        periodicity=(True, True),
    )


_INPUT_AUX = ComponentKey("test/orientation", "aux", "material", "coefficient")


@pytest.mark.parametrize(
    "operation",
    [
        lambda runtime, value: runtime.set_density("rho", value),
        lambda runtime, value: runtime.stage_auxiliary_input(_INPUT_AUX, value),
    ],
    ids=("density", "input-aux"),
)
def test_amr_cell_arrays_reject_transposed_rectangular_shape(operation) -> None:
    with pytest.raises(
        ValueError,
        match=r"cell-array shape differs from the exact native spatial shape",
    ):
        operation(_runtime(), np.zeros((6, 4), dtype=np.float64))


@pytest.mark.parametrize(
    "operation",
    [
        lambda runtime, value: runtime.set_density("rho", value),
        lambda runtime, value: runtime.stage_auxiliary_input(_INPUT_AUX, value),
    ],
    ids=("density", "input-aux"),
)
def test_amr_cell_arrays_reject_flat_python_input(operation) -> None:
    with pytest.raises(
        ValueError,
        match=r"cell-array rank differs from the native spatial dimension",
    ):
        operation(_runtime(), np.zeros(24, dtype=np.float64))
