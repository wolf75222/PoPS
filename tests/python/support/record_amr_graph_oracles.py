"""Record ``tests/python/data/amr_graph_oracles.json`` from the tag-graph route (N12 lot 3c-2b).

This script runs at the last commit that has the tag-graph VM. For each parity case it runs the
graph route and the region route, checks that they give the same observations and the same arrays
(the parity that the tests asserted until then), checks the level-0 requests against a NumPy
evaluation of the criterion where no hysteresis holds a request, and records the graph route's
observations, margin and ties. It leaves with the VM; the file keeps the commit and this command.

    python3.12 -m tests.python.support.record_amr_graph_oracles <output.json> <commit>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pops

from tests.python.support.amr_graph_oracles import (
    MARGIN,
    advection_requests,
    advection_trajectory,
    advection_values,
    layouts_trajectory,
)

FORMS = {
    "recorded_marks": "the observations of the graph route; a run must give them and keep a "
                      "relative margin of at least margin_bound off ties, with the recorded tie "
                      "count",
    "numpy_evaluation": "observations holds the number of observations only; at each one the "
                        "level-0 requests must equal the NumPy evaluation of the criterion on the "
                        "run's own state (tests/python/support/amr_graph_oracles.py), which "
                        "equalled the graph route's requests at every observation when recorded; "
                        "the recorded margin is below margin_bound by construction",
}
COMMAND = "python3.12 -m tests.python.support.record_amr_graph_oracles <output.json> <commit>"


def _advection_case(kind: str) -> dict[str, Any]:
    from tests.python.integration.amr import test_refinement_criterion_parity as parity
    from tests.python.support.refinement_criterion_cases import CELLS, T_END, advection_case

    routes = {}
    for route, rules in (("graph", parity._graph_rules(kind)),
                         ("region", parity._region_rules(kind))):
        case, layout, _ = advection_case(
            rules, thresholds=parity.THRESHOLDS, hysteresis=parity.HYSTERESIS.get(kind),
            name="refinement_criterion_parity")
        runtime = pops.bind(pops.compile(pops.resolve(pops.validate(case), layout=layout)))
        routes[route] = advection_trajectory(runtime, T_END, kind, parity.THRESHOLDS, CELLS)
    (graph, margins, samples), (region, region_margins, region_samples) = (
        routes["graph"], routes["region"])
    assert graph == region and margins == region_margins, kind
    for (state, requests), (other_state, other_requests) in zip(samples, region_samples,
                                                                strict=True):
        assert state.tobytes() == other_state.tobytes() and requests == other_requests, kind
    if kind != "composite":
        for step, (state, requests) in enumerate(samples):
            expected = advection_requests(kind, advection_values(state, CELLS), parity.THRESHOLDS)
            assert np.array_equal(np.frombuffer(requests, dtype=np.uint8), expected), (kind, step)
    if kind == "negation":
        # The band sits on the background value on purpose, and some cells are one ulp away
        # from it: their marks may differ on another build. The oracle is the NumPy evaluation
        # of the criterion on the run's own states, checked above against the graph at every
        # observation (review A3, second form).
        return {"form": "numpy_evaluation", "observations": len(graph), **margins}
    return {"form": "recorded_marks", "observations": graph, **margins}


def _maps_case(directory: Path, build: Any, names: tuple[str, ...],
               thresholds: dict[str, float]) -> dict[str, Any]:
    runs = {}
    for graph in (True, False):
        runs[graph] = layouts_trajectory(
            build(directory / ("graph" if graph else "regions"), graph), names, thresholds)
    assert runs[True] == runs[False]
    observations, margins, _ = runs[True]
    return {"form": "recorded_marks", "observations": observations, **margins}


def record(output: Path, commit: str, directory: Path) -> dict[str, Any]:
    from tests.python.integration.runtime.test_amr_physical_maps import resolve_amr_maps
    from tests.python.integration.runtime.test_interstage_physical_maps import (
        resolve_interstage_maps,
    )

    maps = ("distribution", "moment", "extension")
    stages = ("population", "integral", "extended")
    cases: dict[str, Any] = {}
    for kind in ("value", "gradient", "composite", "negation"):
        cases["advection/%s" % kind] = _advection_case(kind)
    for refine in (-100.0, 100.0):
        cases["physical_maps/refine=%g" % refine] = _maps_case(
            directory / ("maps%g" % refine),
            lambda path, graph, refine=refine: resolve_amr_maps(path, graph=graph, refine=refine),
            maps, dict.fromkeys(maps, refine))
    profile = {"distribution": 1.25, "moment": 1.5, "extension": 1.75}
    cases["physical_maps/a1_profile"] = _maps_case(
        directory / "a1",
        lambda path, graph: resolve_amr_maps(path, graph=graph, initial=1.0, profile=profile),
        maps, profile)
    cases["interstage/adaptive"] = _maps_case(
        directory / "interstage",
        lambda path, graph: resolve_interstage_maps(path, adaptive=True, graph=graph),
        stages, dict.fromkeys(stages, -100.0))
    for name, row in cases.items():
        assert row["form"] == "numpy_evaluation" or row["margin"] >= MARGIN, (name, row["margin"])
    document = {
        "schema": "pops.amr-graph-oracles.v1",
        "about": "Observations of the AMR tag-graph route, recorded before the VM was removed "
                 "(N12 lot 3c-2b): per observation the levels, the boxes (level, lower, upper) "
                 "and per parent level the keep, refine and coarsen request counts; per case the "
                 "smallest relative margin of its comparisons off ties, and its tie count.",
        "commit": commit,
        "command": COMMAND,
        "build": "gcc 13.3, Kokkos 4.4.01 Serial, Dim2, Release",
        "margin_bound": MARGIN,
        "forms": FORMS,
        "cases": cases,
    }
    output.write_text(json.dumps(document, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return document


if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as scratch:
        result = record(Path(sys.argv[1]), sys.argv[2], Path(scratch))
    print(json.dumps({name: {"form": row["form"], "margin": row["margin"], "ties": row["ties"]}
                      for name, row in result["cases"].items()}, indent=1))
