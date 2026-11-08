"""Native runs of the decision to regrid written in the Program, ``Regrid(criterion, when)`` (N12).

The physics is scalar-advection tutorial 09 with its criterion written as functions of ``cell``
(N12 lot 1). Here the Program also writes the decision to regrid (spec 0.3, DM13b): the adaptive
layout has neither ``tagging=`` nor ``regrid=``, and the decision is generated into the compiled
Program, which the core calls after each accepted step (CC4, CF9: no Python runs in the step).

- ``every(2)`` written in the Program regrids as the layout's ``AMRRegrid(every(2))`` does: the
  same times, boxes, coarse state and regrid count after every step, byte for byte.
- A predicate on an indicator of the state, ``largest_cells(cell.value(u)) > peak``, regrids after
  exactly the steps where it holds: the native decision reads the largest value of the accepted
  state over every level, as NumPy computes it on the steps that do not regrid, and the hierarchy
  regrids on the steps where the value exceeds the bound threshold, and only on those.

Every call is collective, so the same file runs on each rank of an MPI world:
``tests/python/integration/mpi/test_public_regrid_decision_mpi{1,2}.py`` run it there, where the
indicators are reduced over the ranks and every rank must decide the same.
"""
from __future__ import annotations

import numpy as np
import pops
import pytest
from pops.amr import Coarsen, Tag, largest_cells
from pops.lib.time import every
from tests.python.support.collective_checks import collective_call, collective_check
from tests.python.support.integral_state_receipts import collective_directory
from tests.python.support.native_execution_context import artifact_execution_context
from tests.python.support.public_native_profile import compile_profile_plan_once, native_profile
from tests.python.support.refinement_criterion_cases import T_END, advection_case

pytestmark = [pytest.mark.kokkos, pytest.mark.regression]

THRESHOLDS = {"refine": 0.30, "coarsen": 0.20, "peak": 0.0}


def _rules(names):
    return (Tag(lambda cell: cell.value(names.state) > names.value("refine")),
            Coarsen(lambda cell: cell.value(names.state) < names.value("coarsen")))


def _peak(names):
    return lambda B: largest_cells(lambda cell: cell.value(names.state)) > names.value("peak")


@pytest.fixture(scope="module")
def decision_native_cache(tmp_path_factory):
    cache = tmp_path_factory.mktemp("regrid-decision-native-cache")
    environment = pytest.MonkeyPatch()
    environment.setenv("POPS_CACHE_DIR", str(cache))
    environment.setenv("POPS_NATIVE_CACHE_DIR", str(cache))
    try:
        yield cache
    finally:
        environment.undo()


def _compiled(world, when):
    case, layout, names = collective_call(world, lambda: advection_case(
        _rules, thresholds=THRESHOLDS, when=when, name="regrid_decision_native"))
    resolved = collective_call(world, lambda: pops.resolve(pops.validate(case), layout=layout))
    return compile_profile_plan_once(world, resolved, route="regrid decision",
                                     compile_artifact=pops.compile), names


def _bind(world, compiled, params=None):
    context = collective_call(world, lambda: artifact_execution_context(compiled))
    return collective_call(world, lambda: pops.bind(
        compiled, params=params, resources={"execution_context": context}))


def _levels(world, runtime):
    count = int(collective_call(world, runtime.n_levels))
    return [np.asarray(collective_call(world, lambda level=level: runtime.block_level_state_global(
        "tracer", level))) for level in range(count)]


def _run(world, compiled, params=None):
    runtime = _bind(world, compiled, params)
    rows = []
    while True:
        executor = runtime._executor
        levels = _levels(world, runtime)
        rows.append({
            "time": float(collective_call(world, runtime.time)),
            "boxes": tuple(collective_call(world, runtime.patch_boxes)),
            "levels": len(levels),
            "coarse": levels[0].tobytes(),
            "regrids": collective_call(world, lambda: runtime.amr.explain_regrid().regrid_count),
            "decision": tuple(collective_call(world, executor._last_regrid_decision)),
            "largest": max(float(np.nanmax(level)) for level in levels),
        })
        if rows[-1]["time"] >= T_END:
            return runtime, rows
        collective_call(world, lambda: pops.step(runtime, t_bound=T_END))


@pytest.mark.compiler
@pytest.mark.native_loader
def test_every_written_in_the_program_regrids_as_the_layout_cadence(
        decision_native_cache) -> None:
    del decision_native_cache
    _, world = native_profile(scope="regrid decision")
    layout_compiled, _ = _compiled(world, None)
    program_compiled, _ = _compiled(world, lambda names: every(2))
    _, expected = _run(world, layout_compiled)
    runtime, actual = _run(world, program_compiled)

    with collective_check(world):
        _assert_cadence(runtime, expected, actual)


def _assert_cadence(runtime, expected, actual):
    assert runtime._executor._generated_regrid_decision_identity().startswith(
        "pops.resolved-amr-regrid-decision.v1::")
    assert len(expected) == len(actual) > 4
    for step, (want, got) in enumerate(zip(expected, actual, strict=True)):
        for key in ("time", "boxes", "levels", "coarse", "regrids"):
            assert got[key] == want[key], (step, key)
        if step:
            # The generated decision ran after every step, true after every second one.
            assert got["decision"][:2] == (step, float(step % 2 == 0)), step
    assert actual[-1]["regrids"] > actual[0]["regrids"]
    assert len({row["boxes"] for row in actual}) > 1


@pytest.mark.compiler
@pytest.mark.native_loader
def test_a_predicate_on_the_state_regrids_after_the_steps_where_it_holds(
        decision_native_cache) -> None:
    del decision_native_cache
    _, world = native_profile(scope="regrid decision")
    compiled, names = _compiled(world, _peak)
    peak = names.params["peak"]
    # With peak = 0 the predicate holds after every step: the decision reads the largest value.
    _, always = _run(world, compiled, params={peak: 0.0})
    largest = [row["decision"][2] for row in always[1:]]
    with collective_check(world):
        assert all(row["decision"][1] == 1.0 for row in always[1:])
        assert largest[0] > largest[-1]
    # A threshold between the first and the last largest value: the predicate holds, then not.
    threshold = 0.5 * (largest[0] + largest[-1])
    _, rows = _run(world, compiled, params={peak: threshold})

    with collective_check(world):
        _assert_predicate(rows, threshold)


def _assert_predicate(rows, threshold):
    outcomes = []
    for step in range(1, len(rows)):
        steps, decided, value = rows[step]["decision"]
        assert steps == step
        assert decided == float(value > threshold), step
        regridded = rows[step]["regrids"] - rows[step - 1]["regrids"]
        assert regridded == int(decided), step
        if not decided:
            # No regrid after this step: the state that the decision read is the one we read.
            assert value == rows[step]["largest"], step
        outcomes.append(bool(decided))
    assert True in outcomes and False in outcomes


@pytest.mark.compiler
@pytest.mark.native_loader
def test_a_checkpoint_of_a_generated_decision_is_refused(decision_native_cache, tmp_path) -> None:
    del decision_native_cache
    _, world = native_profile(scope="regrid decision")
    compiled, _ = _compiled(world, lambda names: every(2))
    runtime = _bind(world, compiled)
    collective_call(world, lambda: pops.step(runtime, t_bound=T_END))
    # Every rank names the same target, as a collective checkpoint requires.
    directory = collective_directory(world, tmp_path / "checkpoint")
    with collective_check(world):
        with pytest.raises(NotImplementedError, match="Regrid\\(criterion, when\\)"):
            runtime.checkpoint(directory / "decision.ckpt")
