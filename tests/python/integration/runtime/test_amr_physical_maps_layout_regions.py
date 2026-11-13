"""Each adaptive layout of the physical maps refines its own region (review A1 of N12 lot 3b).

This test was in ``test_amr_physical_maps.py``. It runs nine minutes, so it has its own file and
its own weight in ``tests/python/test_durations.json``: in the same file it put the file out of the
shard that the catalog planned for it (CI batch 43421613714, shard 86).
"""
import pytest

from tests.python.integration.runtime.test_amr_physical_maps import (
    MAPS,
    PROFILE,
    resolve_amr_maps,
)


@pytest.mark.compiler
@pytest.mark.native_loader
def test_native_amr_maps_each_layout_refines_its_own_region_as_the_tag_graph(tmp_path):
    """Review A1 of lot 3b: from a non-uniform state, with one threshold per layout inside its
    range, each layout refines a different part of its domain; the regions generated into each
    Program slice give the boxes that the tag graph gave, recorded before the VM was removed."""
    import numpy as np
    from tests.python.support.amr_graph_oracles import assert_observations, layouts_trajectory

    # Every block starts from 1 + position, position in (0, 1) along its 8-cell axis.
    plan = resolve_amr_maps(tmp_path, initial=1., profile=PROFILE)
    observations, margins, _ = layouts_trajectory(plan, MAPS, PROFILE)
    assert_observations("physical_maps/a1_profile", observations, margins)
    boxes = {name: [row[1:] for row in observations[0][name]["boxes"] if row[0] == 1]
             for name in MAPS}
    # The recorded boxes of review A1, written out: each layout refines its own part.
    assert boxes == {"distribution": [[[0, 4], [7, 15]]], "moment": [[[8, 0], [15, 0]]],
                     "extension": [[[0, 12], [7, 15]]]}
    cells = {name: sum(int(np.prod(np.subtract(upper, lower) + 1)) for lower, upper in rows)
             for name, rows in boxes.items()}
    full = {"distribution": 4 * 8 * 4, "moment": 8 * 1 * 2, "extension": 4 * 8 * 4}
    assert all(0 < cells[name] < full[name] for name in cells), cells
