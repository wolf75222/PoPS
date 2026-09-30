"""Initial storage uses the authenticated reference geometry of composed layouts."""
from types import SimpleNamespace

import numpy as np
import pytest

from pops.mesh._layout_plan_contracts import NormalizedGeometry
from pops.runtime._bind_validation import validate_bound_initial_values


class ProjectedLayout:
    def __init__(self, cells):
        rank = len(cells)
        self.geometry = NormalizedGeometry(
            coordinate_system="pops://coordinates/cartesian-test@1",
            cell_measure="pops://measure/reference-cell-test@1",
            axis_names=tuple("xyz"[:rank]), lower=(0.0,) * rank,
            upper=(1.0,) * rank, cells=cells,
        )

    def normalized_geometry(self):
        return self.geometry


def validate(layout, values):
    subject = SimpleNamespace
    # Handles are already registry-checked upstream; this isolated gate consumes
    # their exact assigned block id and refuses storage shape/dtype before loading.
    class BoundSubject:
        block_ref = subject(local_id="fluid")
    return validate_bound_initial_values(
        SimpleNamespace(precision="double"),
        SimpleNamespace(instances={"fluid": {"components": 3}}),
        layout, {BoundSubject(): values},
    )


@pytest.mark.parametrize("cells", ((12,), (12, 7), (12, 7, 5)))
def test_projected_reference_counts_keep_native_axis_order(cells):
    layout = ProjectedLayout(cells)
    assert validate(layout, np.zeros((3, *reversed(cells)), dtype=np.float64)) == []
    assert "complete state" in validate(
        layout, np.zeros(tuple(reversed(cells)), dtype=np.float64))[0]
    if len(cells) > 1:
        assert "complete state" in validate(
            layout, np.zeros((3, *cells), dtype=np.float64))[0]
    assert "declared precision" in validate(
        layout, np.zeros((3, *reversed(cells)), dtype=np.float32))[0]


def test_invalid_projection_cannot_fall_back_to_mesh_metadata():
    layout = SimpleNamespace(
        mesh=SimpleNamespace(cells=(12,)),
        normalized_geometry=lambda: SimpleNamespace(cells=(12,)),
    )
    assert "reference-grid cells" in validate(layout, np.zeros((3, 12)))[0]


def test_actual_declared_moving_layout_admits_complete_initial_state():
    from tests.python.unit.codegen.test_moving_interval_codegen import declared_case

    _case, layout = declared_case(components=("a", "b", "c"), cells=17)
    assert validate(layout, np.zeros((3, 17))) == []
    assert "complete state" in validate(layout, np.zeros((3, 16)))[0]

