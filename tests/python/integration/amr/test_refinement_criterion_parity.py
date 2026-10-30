"""Native parity of a refinement criterion written as regions of cell with the tag-graph VM (N12).

Each case runs the physics of scalar-advection tutorial 09 twice: once with the bytecode tag graph,
once with the same criterion written as functions of ``cell`` and generated into the compiled
Program (spec 0.3, DM13b). At bootstrap and after every accepted step, the level-0 requests that
the native tagging plan derives (after its equality and conflict policies) must be the same bytes,
and the patch boxes and the coarse state must be the same: the generated criterion decides as the
VM does, cell for cell, through regrids.

``value`` is the criterion of tutorials 05 and 06, ``gradient`` that of tutorial 09, and
``composite`` joins value and gradient leaves of both comparisons with ``any_of`` and ``all_of``
under a hysteresis of one cycle. ``negation`` writes ``not``, the last logical opcode of the VM, in
a band ``(v > a) & ~(v > b)`` and in a negated coarsen ``~(v < b)``, with ``b`` the background value
that many cells hold exactly: there the VM keeps the negated comparison unknown, and the region
must decide nothing either. The magnitude leaf has no public authoring route.

Every case declares the thresholds of all four criteria and has the same name, so the cases of a
route differ only in their criterion: the module shares one native cache, and the model and the
tag-graph Programs are compiled once.
"""
from __future__ import annotations

import numpy as np
import pops
import pytest
from pops.amr import Coarsen, EqualityPolicy, Hysteresis, Tag
from pops.lib.amr import gradient_norm
from pops.math import ValueExpr, grad, norm
from tests.python.support.refinement_criterion_cases import T_END, advection_case

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


def _graph_rules(kind):
    def rules(names):
        value = ValueExpr(names.state)
        magnitude = norm(grad(value))
        if kind == "value":
            return (Tag(value > names.value("value_refine")),
                    Coarsen(value < names.value("value_coarsen")))
        if kind == "gradient":
            return (Tag(magnitude > names.value("refine")),
                    Coarsen(magnitude < names.value("coarsen")))
        if kind == "negation":
            return (Tag(value > names.value("negation_refine")),
                    Tag((value > names.value("band_lower")) & ~(value > names.value("band_upper"))),
                    Coarsen(~(value < names.value("band_upper"))))
        return (
            Tag(value > names.value("composite_refine")),
            Tag((magnitude > names.value("composite_gradient"))
                & (value > names.value("composite_level"))),
            Coarsen((value < names.value("composite_coarsen"))
                    & (magnitude < names.value("composite_flat"))),
        )
    return rules


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
    rows = []
    while True:
        rows.append({
            "time": float(runtime.time()),
            "requests": bytes(runtime._executor._tagging_requests(0)),
            "boxes": tuple(runtime.patch_boxes()),
            "levels": int(runtime.n_levels()),
            "coarse": np.asarray(runtime.block_level_state_global("tracer", 0)).tobytes(),
            "regrids": runtime.amr.explain_regrid().regrid_count,
        })
        if float(runtime.time()) >= T_END:
            return rows
        pops.step(runtime, t_bound=T_END)


@pytest.mark.compiler
@pytest.mark.native_loader
@pytest.mark.parametrize("kind", ("value", "gradient", "composite", "negation"))
def test_region_criterion_decides_as_the_tag_graph(kind, criterion_native_cache) -> None:
    del criterion_native_cache
    graph = _trajectory(_graph_rules(kind), kind)
    regions = _trajectory(_region_rules(kind), kind)

    assert len(graph) == len(regions) > 2
    for step, (expected, actual) in enumerate(zip(graph, regions, strict=True)):
        assert actual["time"] == expected["time"], step
        assert actual["requests"] == expected["requests"], step
        assert actual["boxes"] == expected["boxes"], step
        assert actual["levels"] == expected["levels"] == 2, step
        assert actual["coarse"] == expected["coarse"], step
        assert actual["regrids"] == expected["regrids"], step
    # The criterion is exercised: both requests occur and the hierarchy regrids along the way.
    requests = b"".join(row["requests"] for row in regions)
    assert requests.count(1) > 0 and requests.count(2) > 0
    assert regions[-1]["regrids"] > regions[0]["regrids"]
    assert len({row["boxes"] for row in regions}) > 1
    if kind == "negation":
        # Cells sit exactly on the threshold of both negations, so the parity covers equality.
        coarse = np.frombuffer(regions[0]["coarse"], dtype=np.float64)
        assert np.count_nonzero(coarse == THRESHOLDS["band_upper"]) > 0
