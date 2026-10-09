"""Ranked, current-epoch AMR ownership projection without a native build."""
from __future__ import annotations

import pytest

from pops.runtime.amr._view import AmrRuntimeView


class _Facade:
    _shape = (8, 12)

    def __init__(self, rows):
        self.rows = rows

    def coarse_local_box_bounds(self):
        return self.rows


def test_local_boxes_are_exact_ranked_half_open_snapshot():
    facade = _Facade((((0, 0), (4, 6)), ((4, 6), (8, 12))))
    view = AmrRuntimeView(facade)
    assert view.coarse_local_box_bounds() == facade.rows
    facade.rows = (((1, 2), (3, 5)),)
    assert view.coarse_local_box_bounds() == facade.rows


@pytest.mark.parametrize("rows,error", [
    ((((0, 0), (0, 4)),), ValueError),
    ((((0, 0), (4,)),), TypeError),
    ((((0, False), (4, 4)),), TypeError),
])
def test_malformed_native_box_fails_closed(rows, error):
    with pytest.raises(error):
        AmrRuntimeView(_Facade(rows)).coarse_local_box_bounds()
