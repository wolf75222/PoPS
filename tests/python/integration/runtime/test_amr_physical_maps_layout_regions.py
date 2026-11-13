"""Each adaptive layout of the physical maps refines its own region (review A1 of N12 lot 3b).

This test was in ``test_amr_physical_maps.py``. It runs nine minutes, so it has its own file and
its own weight in ``tests/python/test_durations.json``: in the same file it put the file out of the
shard that the catalog planned for it (CI batch 43421613714, shard 86).
"""
import pops
import pytest

from tests.python.integration.runtime.test_amr_physical_maps import (
    _hierarchy_image,
    resolve_amr_maps,
)


def _fine_boxes(instance):
    return {name: sorted((tuple(lower), tuple(upper))
                         for level, lower, upper in
                         instance._executor.executor_for_block(name)._s.patch_boxes() if level == 1)
            for name in ("distribution", "moment", "extension")}


@pytest.mark.compiler
@pytest.mark.native_loader
def test_native_amr_maps_each_layout_refines_its_own_region_as_the_tag_graph(tmp_path):
    """Review A1 of lot 3b: from a non-uniform state, with one threshold per layout inside its
    range, each layout refines a different part of its domain; the regions generated into each
    Program slice give the same boxes and states as the tag graph."""
    import numpy as np
    from tests.python.support.native_execution_context import artifact_execution_context

    # Every block starts from 1 + position, position in (0, 1) along its 8-cell axis.
    profile = {"distribution": 1.25, "moment": 1.5, "extension": 1.75}
    boxes, images = {}, {}
    for graph in (True, False):
        plan = resolve_amr_maps(tmp_path / ("graph" if graph else "regions"), graph=graph,
                                initial=1., profile=profile)
        artifact = pops.compile(plan)
        instance = pops.bind(
            artifact, resources={"execution_context": artifact_execution_context(artifact)})
        boxes[graph] = _fine_boxes(instance)
        pops.run(instance, t_end=0.01, max_steps=1)
        images[graph] = _hierarchy_image(instance)
    assert boxes[True] == boxes[False]
    # Every layout refines part of its domain, and no two layouts refine the same part.
    cells = {name: sum(int(np.prod(np.subtract(upper, lower) + 1)) for lower, upper in rows)
             for name, rows in boxes[False].items()}
    full = {"distribution": 4 * 8 * 4, "moment": 8 * 1 * 2, "extension": 4 * 8 * 4}
    assert all(0 < cells[name] < full[name] for name in cells), cells
    assert boxes[False]["distribution"] != boxes[False]["extension"]
    assert images[True].keys() == images[False].keys()
    for key, value in images[True].items():
        np.testing.assert_array_equal(images[False][key], value)
