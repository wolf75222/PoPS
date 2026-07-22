"""The AMR Python boundary authenticates Cartesian array orientation before flattening."""
from __future__ import annotations

import numpy as np
import pytest

try:
    from pops.runtime._system import AmrSystem
except ImportError as exc:  # native extension is installed by the Python integration gate
    pytest.skip("PoPS native extension unavailable: %s" % exc, allow_module_level=True)


def _runtime() -> AmrSystem:
    return AmrSystem(n=6, ny=4, L=3.0, Ly=1.0, periodicity=(True, True))


@pytest.mark.parametrize(
    "operation",
    [
        lambda runtime, value: runtime.set_density("rho", value),
        lambda runtime, value: runtime.set_magnetic_field(value),
        lambda runtime, value: runtime.set_aux_field_component(5, value),
    ],
    ids=("density", "magnetic-field", "named-aux"),
)
def test_amr_cell_arrays_reject_transposed_rectangular_shape(operation) -> None:
    with pytest.raises(
        ValueError,
        match=r"expected Cartesian cell shape \(ny, nx\) = \(4, 6\); got \(6, 4\)",
    ):
        operation(_runtime(), np.zeros((6, 4), dtype=np.float64))


@pytest.mark.parametrize(
    "operation",
    [
        lambda runtime, value: runtime.set_density("rho", value),
        lambda runtime, value: runtime.set_magnetic_field(value),
        lambda runtime, value: runtime.set_aux_field_component(5, value),
    ],
    ids=("density", "magnetic-field", "named-aux"),
)
def test_amr_cell_arrays_reject_flat_python_input(operation) -> None:
    with pytest.raises(
        ValueError,
        match=r"expected one 2D Cartesian cell array of shape \(ny, nx\) = \(4, 6\); got ndim=1",
    ):
        operation(_runtime(), np.zeros(24, dtype=np.float64))
