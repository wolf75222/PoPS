"""Pure oracle checks for the actual M01 script; no native compilation/import."""
import ast
from pathlib import Path

import numpy as np
import pytest

EXAMPLE = Path(__file__).resolve().parents[4] / "examples/migration/scientific/api040_m01_cell_averages.py"


def _oracle():
    tree = ast.parse(EXAMPLE.read_text())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                    and node.name == "exact_cell_averages")
    namespace = {"np": np}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(EXAMPLE), "exec"), namespace)
    return namespace["exact_cell_averages"]


@pytest.mark.parametrize("n", (40, 80, 160, 320))
@pytest.mark.parametrize("time_value", (0., .125))
def test_m01_oracle_is_an_integral_not_a_center_sample(n, time_value):
    exact = _oracle()(n, time_value)
    edges = np.arange(n + 1, dtype=float) / n - time_value
    independent_integral = 1 + .1 * np.diff(np.sin(2 * np.pi * edges)) * n / (2 * np.pi)
    centers = (np.arange(n, dtype=float) + .5) / n - time_value
    point_samples = 1 + .1 * np.cos(2 * np.pi * centers)
    np.testing.assert_allclose(exact, independent_integral, rtol=0., atol=1e-14)
    assert np.max(np.abs(exact - point_samples)) > 1e-8
    assert abs(np.mean(exact) - 1.) < 1e-15
    full_grid = np.broadcast_to(exact, (1, n, n))
    assert full_grid.size == n * n
    np.testing.assert_array_equal(full_grid[0, 0], full_grid[0, -1])
