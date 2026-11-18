"""Recorded oracles of the AMR tag-graph route (N12 lot 3c-2b, design note D4, review A3).

The tag-graph VM was the reference of the parity tests of regions of ``cell``. Before it was
removed, each parity case ran on the graph route, at the commit and with the command that
``tests/python/data/amr_graph_oracles.json`` names, and the integers that the tagging route acts
through were recorded: for each observation, the number of levels, the boxes of each level, and per
tagged level the counts of keep, refine and coarsen requests. The solvers and transfers that make
the arrays do not depend on the route, so equal boxes and requests give equal arrays.

A recorded mark compares an evolved value with a threshold. To keep a last-bit difference between
the recording build and another build from flipping one, each case records the smallest relative
margin ``|value - threshold| / max(|threshold|, 1)`` over the cells that its comparisons read, and
the tests assert that their own run keeps a margin of at least ``MARGIN``. A cell exactly on a
threshold (a tie) decides nothing under ``EqualityPolicy.HOLD``; ties are counted apart, and a case
records how many it has, since only a value that is exact on every build (the background of the
advection cases) may sit there.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

ORACLES = Path(__file__).resolve().parents[1] / "data" / "amr_graph_oracles.json"
MARGIN = 1.0e-9


@lru_cache(maxsize=1)
def _oracles() -> dict[str, Any]:
    return json.loads(ORACLES.read_text(encoding="utf-8"))


def oracle(case: str) -> dict[str, Any]:
    """The recorded oracle of one case."""
    cases = _oracles()["cases"]
    if case not in cases:
        raise KeyError("no recorded tag-graph oracle for case %r" % case)
    return cases[case]


def mark_counts(requests: bytes) -> list[int]:
    """The keep, refine and coarsen counts of one level's requests (0 keep, 1 refine, 2 coarsen)."""
    values = np.frombuffer(bytes(requests), dtype=np.uint8)
    if values.size and int(values.max()) > 2:
        raise ValueError("a tagging request is 0, 1 or 2")
    return [int(np.count_nonzero(values == code)) for code in (0, 1, 2)]


def _boxes(rows: Any) -> list[list[Any]]:
    return sorted([int(level), [int(v) for v in lower], [int(v) for v in upper]]
                  for level, lower, upper in rows)


def observe_system(system: Any, *, levels: int, boxes: Any) -> dict[str, Any]:
    """One observation of one adaptive layout: its levels, boxes and per-parent-level marks."""
    return {
        "levels": int(levels),
        "boxes": _boxes(boxes),
        "marks": [mark_counts(system._tagging_requests(level)) for level in range(int(levels) - 1)],
    }


def relative_margin(values: Any, threshold: float) -> tuple[float, int]:
    """The smallest ``|value - threshold| / max(|threshold|, 1)`` off ties, and the tie count."""
    values = np.asarray(values, dtype=np.float64).ravel()
    ties = values == threshold
    rest = values[~ties]
    if not rest.size:
        return float("inf"), int(np.count_nonzero(ties))
    scale = max(abs(float(threshold)), 1.0)
    return float(np.min(np.abs(rest - threshold)) / scale), int(np.count_nonzero(ties))


def central_gradient_norm(field: Any, widths: tuple[float, ...], ghosts: Any) -> Any:
    """``|grad f|`` with the stencil (-1/2, +1/2) along each axis, on the valid cells of ``field``.

    ``ghosts(field)`` returns ``field`` padded by one cell on each side of each axis, as the
    boundary conditions of the case fill the ghost cells that the stencil reads.
    """
    padded = np.asarray(ghosts(np.asarray(field, dtype=np.float64)))
    squared = None
    for axis, width in enumerate(widths):
        lower = [slice(1, -1)] * padded.ndim
        upper = [slice(1, -1)] * padded.ndim
        lower[axis] = slice(0, -2)
        upper[axis] = slice(2, None)
        derivative = (-0.5 * padded[tuple(lower)] + 0.5 * padded[tuple(upper)]) / width
        square = derivative * derivative
        squared = square if squared is None else squared + square
    return np.sqrt(squared)


def combine_margins(rows: list[tuple[float, int]]) -> dict[str, Any]:
    """The smallest margin and the total tie count of several comparisons."""
    margin = min((row[0] for row in rows), default=float("inf"))
    return {"margin": margin, "ties": int(sum(row[1] for row in rows))}


def assert_observations(case: str, observed: list[dict[str, Any]], margins: dict[str, Any]) -> None:
    """Assert that a run gives the recorded observations and keeps the margin bound."""
    expected = oracle(case)
    assert len(observed) == len(expected["observations"]), (case, len(observed))
    for index, (want, got) in enumerate(zip(expected["observations"], observed, strict=True)):
        assert got == want, (case, index)
    assert margins["margin"] >= MARGIN, (case, margins)
    assert margins["ties"] == expected["ties"], (case, margins)


# The comparisons of the advection parity cases (test_refinement_criterion_parity.py): which
# value each one reads, u or |grad u|, and the name of its threshold.
ADVECTION_COMPARISONS = {
    "value": (("u", "value_refine"), ("u", "value_coarsen")),
    "gradient": (("g", "refine"), ("g", "coarsen")),
    "composite": (("u", "composite_refine"), ("g", "composite_gradient"),
                  ("u", "composite_level"), ("u", "composite_coarsen"), ("g", "composite_flat")),
    "negation": (("u", "negation_refine"), ("u", "band_lower"), ("u", "band_upper")),
}


def advection_ghosts(field: Any) -> Any:
    """The ghost cells that the boundaries of the advection cases give the gradient stencil.

    The cases inflow 0.05 at x_min and y_min and outflow at x_max and y_max. ``field`` is indexed
    ``[y, x]`` (the native arrays are x fastest).
    """
    padded = np.pad(field, 1, mode="edge")
    padded[:, 0] = 0.05
    padded[0, :] = 0.05
    return padded


def advection_values(state: Any, cells: tuple[int, int]) -> dict[str, Any]:
    """u and |grad u| on the valid cells of level 0, from its x-fastest state."""
    nx, ny = cells
    u = np.asarray(state, dtype=np.float64)[: nx * ny].reshape(ny, nx)
    return {"u": u, "g": central_gradient_norm(u, (1.0 / ny, 1.0 / nx), advection_ghosts)}


def advection_requests(kind: str, values: dict[str, Any], t: dict[str, float]) -> Any:
    """The level-0 requests of an advection case without hysteresis, by NumPy.

    This is the semantics of the tag graph under ``EqualityPolicy.HOLD`` and
    ``ConflictPolicy.REFINE_WINS``: a comparison at exact equality is unknown, and so is its
    negation (Kleene); a cell is refined where a refine rule is certainly true, else coarsened
    where a coarsen rule is.
    """
    u, g = values["u"], values["g"]
    if kind == "value":
        refine, coarsen = u > t["value_refine"], u < t["value_coarsen"]
    elif kind == "gradient":
        refine, coarsen = g > t["refine"], g < t["coarsen"]
    elif kind == "negation":
        lower, upper = t["band_lower"], t["band_upper"]
        # (u > lower) & ~(u > upper) is certainly true off both ties; ~(u < upper) likewise.
        refine = (u > t["negation_refine"]) | ((u > lower) & (u < upper))
        coarsen = u > upper
    else:
        raise ValueError("the %s case holds requests through hysteresis" % kind)
    return np.where(refine, 1, np.where(coarsen, 2, 0)).astype(np.uint8).ravel()


def advection_trajectory(runtime: Any, t_end: float, kind: str, thresholds: dict[str, float],
                         cells: tuple[int, int]) -> tuple[list[dict[str, Any]], dict[str, Any],
                                                          list[tuple[Any, Any]]]:
    """Step one advection parity case to ``t_end``: its observations, margins and samples.

    Each observation is taken at bind and after every accepted step, as the parity test did on
    both routes; each sample is the level-0 (state, requests) pair of that observation.
    """
    import pops

    observations, rows, samples = [], [], []
    while True:
        # The requests of level 0 are read once per observation, as the parity test read them.
        levels = int(runtime.n_levels())
        requests = bytes(runtime._executor._tagging_requests(0))
        if levels != 2:
            raise ValueError("the advection parity cases keep two levels")
        observations.append({"levels": levels, "boxes": _boxes(runtime.patch_boxes()),
                             "marks": [mark_counts(requests)],
                             "regrids": int(runtime.amr.explain_regrid().regrid_count)})
        state = np.asarray(runtime.block_level_state_global("tracer", 0)).copy()
        samples.append((state, requests))
        values = advection_values(state, cells)
        rows.extend(relative_margin(values[read], thresholds[name])
                    for read, name in ADVECTION_COMPARISONS[kind])
        if float(runtime.time()) >= t_end:
            return observations, combine_margins(rows), samples
        pops.step(runtime, t_bound=t_end)


def layouts_trajectory(plan: Any, names: tuple[str, ...], thresholds: dict[str, float],
                       ) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[Any, bytes]]]:
    """Compile and bind a plan of several adaptive layouts, observe it at bind and after one step.

    Each observation maps each layout, named by its block, to its levels, boxes and marks; the
    margins compare the first component of each level-0 state with its layout's threshold; the
    images are the bytes of every level state, for a comparison of two routes.
    """
    import pops

    from tests.python.support.native_execution_context import artifact_execution_context

    artifact = pops.compile(plan)
    instance = pops.bind(
        artifact, resources={"execution_context": artifact_execution_context(artifact)})
    observations, rows, images = [], [], []
    for index in range(2):
        observation = {}
        for name in names:
            executor = instance._executor.executor_for_block(name)
            levels = int(executor.n_levels())
            observation[name] = observe_system(executor._s, levels=levels,
                                               boxes=executor._s.patch_boxes())
            cells = len(executor._s._tagging_requests(0))
            state = np.asarray(instance.block_level_state_global(name, 0))
            # The criterion reads the first component, packed first (component * cells, x fastest).
            rows.append(relative_margin(state[:cells], thresholds[name]))
        observations.append(observation)
        images.append({(name, level): np.asarray(instance.block_level_state_global(name, level))
                       .tobytes()
                       for name in names
                       for level in range(instance._executor.executor_for_block(name).n_levels())})
        if index == 0:
            pops.run(instance, t_end=0.01, max_steps=1)
    return observations, combine_margins(rows), images


__all__ = [
    "ADVECTION_COMPARISONS",
    "MARGIN",
    "ORACLES",
    "advection_ghosts",
    "advection_requests",
    "advection_trajectory",
    "advection_values",
    "assert_observations",
    "central_gradient_norm",
    "combine_margins",
    "layouts_trajectory",
    "mark_counts",
    "observe_system",
    "oracle",
    "relative_margin",
]
