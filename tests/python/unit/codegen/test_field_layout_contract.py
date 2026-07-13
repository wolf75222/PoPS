from __future__ import annotations

import pytest

from pops.codegen.field_boundary_lowering import (
    field_layout_contract,
    topology_recipe,
)
from pops.descriptors_report import CapabilitySet
from pops.domain import Rectangle
from pops.frames import Cartesian2D
from pops.mesh import CartesianGrid


class ExtensionAMR:
    """An AMR extension protocol implementation that inherits no PoPS layout class."""

    def __init__(self) -> None:
        frame = Rectangle("extension", (0.0, 0.0), (1.0, 1.0)).frame(Cartesian2D())
        self.grid = CartesianGrid(frame=frame, cells=(8, 8))

    def capabilities(self) -> CapabilitySet:
        return CapabilitySet({
            "layout": "amr",
            "max_levels": 4,
            "transition_ratios": [2, 4, 2],
        })


def test_field_layout_contract_accepts_an_open_amr_extension_protocol() -> None:
    layout = ExtensionAMR()
    contract = field_layout_contract(layout)
    recipe = topology_recipe(layout)

    assert contract.kind == "amr"
    assert contract.mesh is layout.grid
    assert contract.levels == 4
    assert contract.transition_ratios == (2, 4, 2)
    assert contract.level_refinements == (1, 2, 8, 16)
    assert recipe["connectivity"]["graph"] == "amr-composite-cell-graph"
    assert recipe["levels"] == 4
    assert recipe["transition_ratios"] == [2, 4, 2]
    assert recipe["level_refinements"] == [1, 2, 8, 16]


def test_field_layout_contract_refuses_missing_hierarchy_evidence() -> None:
    layout = ExtensionAMR()
    layout.capabilities = lambda: CapabilitySet({
        "layout": "amr",
        "transition_ratios": [2],
    })

    with pytest.raises(TypeError, match="max_levels must be an integer"):
        field_layout_contract(layout)
