"""N8 lot « reuse across stages », native Dim2 (spec 0.3 T7; design note, revision 2, sections 2
and 5).

The authored method ``twice`` requests the field of M10 and of M08 at ``B.state`` at two stages.
With the lot (A on), the second request aliases the first solve: one prepared solve per step and
one reuse in the field counters. With the lot off (``_time_independent_request`` patched to
refuse), each stage solves. The final states are bitwise equal.

Reject-then-accept (lot C): a test-only C++ patch rejects the first candidate of the run once,
at the default assessment ``finite`` (``tests/python/support/one_shot_rejection.py``). Under
Retry(2), the second attempt keeps the stage-0 solve of its fixed site (lot 6) and the stage-1
request aliases it: twice solves once in its first step, M08 three times. The patched runs end
bitwise where the unpatched ones do.
"""
import contextlib

import numpy as np
import pytest

import pops
import pops.codegen.program_field_reuse as field_reuse
import examples.api_v02.models_M01_M14 as early
from pops.runtime._simulation import Simulation
from pops.time._program.elaboration import Advance, Proposal, Retry, elaborate
from tests.python.unit.time.test_reuse_across_stages_n8 import _guarded_route
from pops.time._program.traced_method import Candidate, method
from tests.python.support.collective_checks import collective_call, collective_check
from tests.python.support.native_execution_context import artifact_execution_context
from tests.python.support.one_shot_rejection import one_shot_rejection
from tests.python.support.public_native_profile import compile_profile_plan_once, native_profile

pytestmark = [pytest.mark.compiler, pytest.mark.kokkos, pytest.mark.native_loader]


@method
def twice(B, p, flow):
    k0 = flow(B.state, B.time)
    k1 = flow(B.state, B.time + p.dt)
    return Candidate(B.state + .5*p.dt*(k0 + k1), B.time + p.dt)


def _twice_route(attempts):
    def route(case, block, state, flow, dt, *, name, memory, record, b6):
        del memory, record, b6
        return elaborate(Advance(twice(flow=flow), retry=Retry(max_attempts=attempts)),
                         case=case, state=block[state], proposal=Proposal(dt), name=name)
    return route


def _recorded_route(attempts):
    def route(case, block, state, flow, dt, *, name, memory, record, b6):
        del b6
        return elaborate(Advance(early.recorded_ssprk2(memory, record)(flow=flow),
                                 retry=Retry(max_attempts=attempts)),
                         case=case, state=block[state], proposal=Proposal(dt), name=name)
    return route


def _b6_route(case, block, state, flow, dt, *, name, memory, record, b6):
    return early._ssprk2(case, block, state, b6, dt, name=name)


def _bind(world, monkeypatch, build, route, label, *, reuse=True, rejection=False):
    with monkeypatch.context() as patch:
        patch.setattr(early, "_traced_recorded_ssprk2", route)
        spec = collective_call(world, build)
    plan = collective_call(world, lambda: pops.resolve(pops.validate(spec.case),
                                                       layout=spec.layout))
    with monkeypatch.context() as patch:
        if not reuse:
            patch.setattr(field_reuse, "_time_independent_request",
                          lambda value, *, target: False)
        route = "N8 reuse across stages %s%s%s" % (
            label, "" if reuse else " A off", " one-shot rejection" if rejection else "")
        with one_shot_rejection(patch, route) if rejection else contextlib.nullcontext():
            artifact = compile_profile_plan_once(world, plan, route=route,
                                                 compile_artifact=pops.compile)
    runtime = collective_call(world, lambda: pops.bind(artifact, initial_values={
        subject: value.copy() for subject, value in spec.initial_values.items()},
        resources={"execution_context": artifact_execution_context(artifact)}))
    return spec, runtime


def _states(world, spec, runtime):
    del spec  # M08 and M10 hold one block, ``fluid``
    value = collective_call(world, lambda: runtime.state_global("fluid"))
    return None if value is None else {"fluid": np.asarray(value).copy()}


def _counters(world, runtime):
    def read():
        diagnostics = runtime.program_report().diagnostics
        solves = [key for key in diagnostics if key.startswith("field.solves/")]
        assert len(solves) == 1, sorted(diagnostics)
        field = solves[0][len("field.solves/"):]
        return (int(diagnostics["field.solves/" + field]),
                int(diagnostics["field.reuses/" + field]))
    return collective_call(world, read)


def _bitwise(left, right):
    if left is None and right is None:
        return
    assert left is not None and right is not None and sorted(left) == sorted(right)
    for subject in left:
        assert left[subject].dtype == right[subject].dtype == np.float64
        # Bitwise, not within a tolerance: the alias reads the bits that the solve wrote.
        assert left[subject].tobytes() == right[subject].tobytes(), subject


def _no_negative_zero(states):
    # Note section 2.7: the stage-1 copy of B.state turns a -0 cell into +0, so the alias is
    # bitwise the second solve only where no state holds -0. Asserted on the initial state and
    # after every step, so that a -0 is named here rather than by a bitwise mismatch.
    if states is not None:
        for subject, state in states.items():
            assert not np.any((state == 0.0) & np.signbit(state)), subject


@pytest.mark.parametrize("profile", ["M10", "M08"])
def test_twice_solves_once_per_step_and_ends_bitwise_like_two_solves(
        profile, monkeypatch, isolated_native_cache, native_cxx, kokkos_root):
    """Review P2: A on gives one prepared solve and one reuse per step, A off two solves and no
    reuse; the states are bitwise equal."""
    del isolated_native_cache, native_cxx, kokkos_root
    build = {"M10": early.build_m10, "M08": early.build_m08}[profile]
    _, world = native_profile(dimension=2, scope="N8 reuse across stages %s" % profile)
    runs = {}
    for reuse in (True, False):
        spec, runtime = _bind(world, monkeypatch, build, _twice_route(1), "twice %s" % profile,
                              reuse=reuse)
        with collective_check(world):
            _no_negative_zero({str(subject): np.asarray(value)
                               for subject, value in spec.initial_values.items()})
        result = collective_call(world, lambda: Simulation(runtime).run(spec.t_end))
        runs[reuse] = (result, _counters(world, runtime), _states(world, spec, runtime))
    (on, on_counters, on_states), (off, off_counters, off_states) = runs[True], runs[False]
    with collective_check(world):
        assert on.steps == off.steps > 0 and on.rejections == off.rejections == 0
        assert on.diagnostics["native"]["prepared_solves"] == on.steps
        assert off.diagnostics["native"]["prepared_solves"] == 2 * off.steps
        assert on_counters == (1, 1) and off_counters == (2, 0)
        _bitwise(on_states, off_states)
        _no_negative_zero(on_states)


def test_a_guard_between_the_requests_keeps_both_solves(
        monkeypatch, isolated_native_cache, native_cxx, kokkos_root):
    """Review R1 of #781: ``twice`` with a guard of ``B.state`` between its two requests (a real
    barrier of the elaborated IR) solves twice per step and reuses nothing: counters (2, 0)."""
    del isolated_native_cache, native_cxx, kokkos_root
    _, world = native_profile(dimension=2, scope="N8 reuse across stages guarded M10")
    spec, runtime = _bind(world, monkeypatch, early.build_m10, _guarded_route, "twice guarded M10")
    result = collective_call(world, lambda: Simulation(runtime).run(spec.t_end))
    counters = _counters(world, runtime)
    with collective_check(world):
        assert result.steps > 0 and result.rejections == 0
        assert result.diagnostics["native"]["prepared_solves"] == 2 * result.steps
        assert counters == (2, 0)


# --- reject-then-accept (lot C) ------------------------------------------------------------------

def _two_steps(world, runtime):
    """Two runs of one step each; per step, the field counters of its accepted invocation and the
    prepared solves of the run, every attempt included."""
    simulation = Simulation(runtime)
    rows = []
    for _ in range(2):
        def one():
            until = runtime.time() + simulation._control.first_dt
            return simulation.run(until)
        result = collective_call(world, one)
        rows.append((result.steps, result.rejections, _counters(world, runtime),
                     result.diagnostics["native"]["prepared_solves"]))
        states = _states(world, None, runtime)
        with collective_check(world):
            _no_negative_zero(states)
    return rows


def _expected(rows):
    """Cumulative prepared solves, as the table of the note gives them."""
    total, cumulative = 0, []
    for steps, rejections, counters, solves in rows:
        total += solves
        cumulative.append((steps, rejections, counters, total))
    return cumulative


def test_a_rejected_twice_step_keeps_its_solve_and_aliases_the_second_stage(
        monkeypatch, isolated_native_cache, native_cxx, kokkos_root):
    """twice/M10 under Retry(2), its first candidate rejected once: the second attempt solves
    nothing (stage 0 kept, stage 1 aliased); the second step solves once. Bitwise like the
    unpatched run and like A off."""
    del isolated_native_cache, native_cxx, kokkos_root
    _, world = native_profile(dimension=2, scope="N8 reuse across stages reject twice")
    spec, patched = _bind(world, monkeypatch, early.build_m10, _twice_route(2), "twice M10",
                          rejection=True)
    rows = _two_steps(world, patched)
    _, plain = _bind(world, monkeypatch, early.build_m10, _twice_route(2), "twice M10")
    plain_rows = _two_steps(world, plain)
    _, off = _bind(world, monkeypatch, early.build_m10, _twice_route(2), "twice M10",
                   reuse=False)
    _two_steps(world, off)
    states = [_states(world, spec, runtime) for runtime in (patched, plain, off)]
    with collective_check(world):
        assert _expected(rows) == [(1, 1, (0, 2), 1), (1, 0, (1, 1), 2)]
        assert _expected(plain_rows) == [(1, 0, (1, 1), 1), (1, 0, (1, 1), 2)]
        _bitwise(states[0], states[1])
        _bitwise(states[0], states[2])
        _no_negative_zero(states[0])


def test_a_rejected_m08_step_keeps_its_fixed_solve_and_ends_like_b6(
        monkeypatch, isolated_native_cache, native_cxx, kokkos_root):
    """M08's own method under Retry(2), its first candidate rejected once: stage 1 reads the
    proposal, so no request is time-independent; the second attempt keeps the stage-0 solve
    (lot 6) and solves stage 1. Bitwise like b6's Program."""
    del isolated_native_cache, native_cxx, kokkos_root
    _, world = native_profile(dimension=2, scope="N8 reuse across stages reject M08")
    spec, patched = _bind(world, monkeypatch, early.build_m08, _recorded_route(2), "M08",
                          rejection=True)
    rows = _two_steps(world, patched)
    _, b6 = _bind(world, monkeypatch, early.build_m08, _b6_route, "M08 b6")
    # b6's Program is not an Advance: pops.step takes its two accepted steps.
    for _ in range(2):
        report = collective_call(world, lambda: pops.step(b6, t_bound=spec.t_end))
        with collective_check(world):
            assert report.accepted_steps == 1 and report.rejected_steps == 0
    states = [_states(world, spec, runtime) for runtime in (patched, b6)]
    with collective_check(world):
        assert _expected(rows) == [(1, 1, (1, 1), 3), (1, 0, (2, 0), 5)]
        _bitwise(states[0], states[1])
        _no_negative_zero(states[0])
