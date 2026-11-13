"""A refinement criterion written as regions of cell decides as the tag-graph VM did (N12).

Each case runs the physics of scalar-advection tutorial 09 with the criterion written as functions
of ``cell`` and generated into the compiled Program (spec 0.3, DM13b). At bind and after every
accepted step, the levels, the patch boxes, the counts of the level-0 requests that the native
tagging plan derives (after its equality and conflict policies) and the regrid count must be those
that the bytecode tag graph gave, recorded in ``tests/python/data/amr_graph_oracles.json`` before
the VM was removed in lot 3c-2b (``tests/python/support/amr_graph_oracles.py`` says how). The run
must also keep every compared value at least 1e-9 (relative) away from its threshold, off the
cells that sit on it exactly, so that the recorded marks do not depend on the last bit of a build.
The ``negation`` case cannot (its band sits on the background value, and cells sit one ulp from
it): its requests are checked at every observation against a NumPy evaluation of the criterion
on the run's own state, which matched the tag graph at every observation when it was recorded.

``value`` is the criterion of tutorials 05 and 06, ``gradient`` that of tutorial 09, and
``composite`` joins value and gradient leaves of both comparisons with ``any_of`` and ``all_of``
under a hysteresis of one cycle. ``negation`` writes ``not``, the last logical opcode of the VM, in
a band ``(v > a) & ~(v > b)`` and in a negated coarsen ``~(v < b)``, with ``b`` the background value
that many cells hold exactly: there the VM kept the negated comparison unknown, and the region
must decide nothing either.

Every case declares the thresholds of all four criteria and has the same name, so the cases differ
only in their criterion: the module shares one native cache, and the model is compiled once.
"""
from __future__ import annotations

import numpy as np
import pops
import pytest
from pops.amr import Coarsen, EqualityPolicy, Hysteresis, Tag
from pops.lib.amr import gradient_norm
from tests.python.support.amr_graph_oracles import (
    advection_requests,
    advection_trajectory,
    advection_values,
    assert_observations,
    oracle,
)
from tests.python.support.refinement_criterion_cases import CELLS, T_END, advection_case

pytestmark = [pytest.mark.kokkos, pytest.mark.regression]

# refine and coarsen are the thresholds of tutorial 09, value_* those of tutorials 05 and 06.
THRESHOLDS = {
    "refine": 0.10, "coarsen": 0.04,
    "value_refine": 0.30, "value_coarsen": 0.20,
    "composite_refine": 0.60, "composite_coarsen": 0.20, "composite_gradient": 2.0,
    "composite_level": 0.15, "composite_flat": 0.50,
    "negation_refine": 0.30, "band_lower": 0.04, "band_upper": 0.05,
}
HYSTERESIS = {"composite": Hysteresis(1, EqualityPolicy.HOLD)}


def _region_rules(kind):
    def rules(names):
        axes = names.frame.axes

        def magnitude(cell):
            return gradient_norm(cell, names.state, axes=axes, stencil=names.gradient)

        def value(cell):
            return cell.value(names.state)

        if kind == "value":
            return (Tag(lambda cell: value(cell) > names.value("value_refine")),
                    Coarsen(lambda cell: value(cell) < names.value("value_coarsen")))
        if kind == "gradient":
            return (Tag(lambda cell: magnitude(cell) > names.value("refine")),
                    Coarsen(lambda cell: magnitude(cell) < names.value("coarsen")))
        if kind == "negation":
            return (Tag(lambda cell: value(cell) > names.value("negation_refine")),
                    Tag(lambda cell: (value(cell) > names.value("band_lower"))
                        & ~(value(cell) > names.value("band_upper"))),
                    Coarsen(lambda cell: ~(value(cell) < names.value("band_upper"))))
        return (
            Tag(lambda cell: value(cell) > names.value("composite_refine")),
            Tag(lambda cell: (magnitude(cell) > names.value("composite_gradient"))
                & (value(cell) > names.value("composite_level"))),
            Coarsen(lambda cell: (value(cell) < names.value("composite_coarsen"))
                    & (magnitude(cell) < names.value("composite_flat"))),
        )
    return rules


@pytest.fixture(scope="module")
def criterion_native_cache(tmp_path_factory):
    cache = tmp_path_factory.mktemp("refinement-criterion-native-cache")
    environment = pytest.MonkeyPatch()
    environment.setenv("POPS_CACHE_DIR", str(cache))
    environment.setenv("POPS_NATIVE_CACHE_DIR", str(cache))
    try:
        yield cache
    finally:
        environment.undo()


def _trajectory(rules, kind):
    case, layout, _ = advection_case(
        rules, thresholds=THRESHOLDS, hysteresis=HYSTERESIS.get(kind),
        name="refinement_criterion_parity")
    runtime = pops.bind(pops.compile(pops.resolve(pops.validate(case), layout=layout)))
    return advection_trajectory(runtime, T_END, kind, THRESHOLDS, CELLS)


@pytest.mark.compiler
@pytest.mark.native_loader
@pytest.mark.parametrize("kind", ("value", "gradient", "composite", "negation"))
def test_region_criterion_decides_as_the_recorded_tag_graph(kind, criterion_native_cache) -> None:
    del criterion_native_cache
    observations, margins, samples = _trajectory(_region_rules(kind), kind)

    if kind == "negation":
        # The band sits on the background value, and some cells sit one ulp from it, so marks
        # recorded on one build could differ on another (review A3): the oracle is the NumPy
        # evaluation of the criterion on the run's own states, which matched the tag graph at
        # every observation when the oracles were recorded.
        expected = oracle("advection/negation")
        assert expected["form"] == "numpy_evaluation"
        assert len(observations) == expected["observations"]
        for step, (state, requests) in enumerate(samples):
            values = advection_values(state, CELLS)
            assert np.array_equal(np.frombuffer(requests, dtype=np.uint8),
                                  advection_requests(kind, values, THRESHOLDS)), step
    else:
        assert_observations("advection/%s" % kind, observations, margins)
    assert len(observations) > 2
    assert all(row["levels"] == 2 for row in observations)
    # The criterion is exercised: both requests occur and the hierarchy regrids along the way.
    assert sum(row["marks"][0][1] for row in observations) > 0
    assert sum(row["marks"][0][2] for row in observations) > 0
    assert observations[-1]["regrids"] > observations[0]["regrids"]
    assert len({str(row["boxes"]) for row in observations}) > 1
    if kind == "negation":
        # Cells sit exactly on the threshold of both negations. Under Kleene's logic a comparison
        # there decides nothing, so the region keeps them, as the VM did. A two-valued reading of
        # ~, the one the surface of §2.6.2 would take without an equality policy (DM13b, open with
        # N15), would have both rules act there, and refinement would win.
        state, requests = samples[0]
        on_threshold = state == THRESHOLDS["band_upper"]
        assert np.count_nonzero(on_threshold) > 0
        assert margins["ties"] > 0
        requests = np.frombuffer(requests, dtype=np.uint8)
        assert requests.shape == state.shape
        assert not requests[on_threshold].any()


@pytest.mark.compiler
@pytest.mark.native_loader
def test_a_checkpoint_of_the_tag_graph_route_is_refused_by_identity(
        tmp_path, criterion_native_cache) -> None:
    """O3 of the lot 3c-2b note: the ``value`` case, written on the graph route and checkpointed
    at the commit before the graph went (``tests/python/data/amr_graph_checkpoint.json`` names it),
    is refused by the same case written as regions, by identity, before any state changes. Its own
    checkpoint restarts, so the refusal is the identity's."""
    import hashlib
    import json
    import shutil
    from pathlib import Path

    del criterion_native_cache
    data = Path(__file__).resolve().parents[2] / "data"
    provenance = json.loads((data / "amr_graph_checkpoint.json").read_text(encoding="utf-8"))
    fixture = data / provenance["file"]
    assert hashlib.sha256(fixture.read_bytes()).hexdigest() == provenance["sha256"]
    graph_checkpoint = tmp_path / "graph" / provenance["file"]
    graph_checkpoint.parent.mkdir()
    shutil.copyfile(fixture, graph_checkpoint)

    case, layout, _ = advection_case(_region_rules("value"), thresholds=THRESHOLDS,
                                     name="refinement_criterion_parity")
    runtime = pops.bind(pops.compile(pops.resolve(pops.validate(case), layout=layout)))
    pops.step(runtime, t_bound=T_END)
    own = runtime.checkpoint(tmp_path / "region.npz")
    time, step = runtime.time(), runtime.macro_step()
    with pytest.raises(ValueError, match="checkpoint semantic_identity does not match"):
        runtime.restart(graph_checkpoint)
    assert (runtime.time(), runtime.macro_step()) == (time, step)
    runtime.restart(own)
    assert (runtime.time(), runtime.macro_step()) == (time, step)
