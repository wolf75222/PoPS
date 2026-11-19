"""G03 lot M1a (ROM-21): the members X and V of a moving configuration, run natively in rank one.

The corner copies (x_left, x_right) and the volume V of each cell are the state of a block of their
own, advanced by SSP-RK3 with the rates X' = w(X, t) and V' = w(x_right, t) - w(x_left, t); the
stage time reaches the rates through an analytic auxiliary of the Program's clock (note §7.2). The
oracle is an independent NumPy run of the same members (``ssprk3_oracle``), and, for a node rate
derived from a map, the map itself.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from examples.api_v02.models_G03_moving_members import (
    AMPLITUDE, DT, STEPS, build_members, ssprk3_oracle)
from tests.python.integration.runtime.test_n17_static_geometry_graphs import run_graph
from tests.python.support.collective_checks import collective_call, collective_check

pytestmark = [pytest.mark.compiler, pytest.mark.kokkos, pytest.mark.native_loader]

EPS = np.finfo(float).eps


def _run(scope, *, steps=STEPS, dt=DT, **keywords):
    holder = {}

    def build():
        case, layout, values, configuration = build_members(dt=dt, **keywords)
        holder["configuration"] = configuration
        (initial,) = values.values()
        return case, layout, values, initial

    world, runtime, report, initial = run_graph(build, dimension=1, scope=scope, steps=steps,
                                                dt=dt)
    members = collective_call(world, lambda: np.asarray(
        runtime.state_global("configuration")).copy())
    return world, runtime, report, members, holder["configuration"]


def _gcl_bound(members, steps):
    """V - (x_right - x_left) is rounding only: each of the 3 stages of a step rounds the three
    members once, relative to the largest position."""
    return 3*steps*4*EPS*max(1.0, float(np.max(np.abs(members[:2]))))


def _check_members(members, steps):
    left, right, volume = members
    # The left copy of a cell and the right copy of its neighbour are one node: bitwise equal.
    assert left[1:].tobytes() == right[:-1].tobytes()
    assert np.max(np.abs(volume - (right - left))) <= _gcl_bound(members, steps)
    assert np.all(volume > 0)


def test_the_members_of_a_node_rate_read_at_the_copies_on_graded_cells(
        isolated_native_cache, native_cxx, kokkos_root, record_property):
    """Form 1, w(x, t) = A sin(2 pi x) cos(2 pi t) at each copy's own position."""
    del isolated_native_cache, native_cxx, kokkos_root
    world, _, report, members, configuration = _run(
        "G03 lot M1a members, node rate at the copies", base="graded")
    oracle = ssprk3_oracle(configuration, STEPS, DT)
    gcl = float(np.max(np.abs(members[2] - (members[1] - members[0]))))
    with collective_check(world):
        assert report.accepted_steps == STEPS
        _check_members(members, STEPS)
        # The nodes moved by about A dt per step: the run is not the base.
        start = configuration.initial_members()
        assert np.max(np.abs(members[0] - start[0])) > 0.1*AMPLITUDE*DT
        np.testing.assert_allclose(members, oracle, rtol=0, atol=8*EPS)
    record_property("gcl", gcl)
    record_property("oracle", float(np.max(np.abs(members - oracle))))


def _map_error(members, configuration, t_end):
    """The nodes against the map: X(T) - X(0) against Φ(ξ, T) - Φ(ξ, 0)."""
    xi = configuration.reference_nodes()
    displacement = AMPLITUDE*np.sin(2*np.pi*xi)*np.sin(2*np.pi*t_end)/(2*np.pi)
    nodes = np.concatenate((members[0], members[1][-1:]))
    return float(np.max(np.abs(nodes - xi - displacement)))


def test_the_nodes_follow_a_map_to_the_order_of_the_method(
        isolated_native_cache, native_cxx, kokkos_root, record_property):
    """Form 2, Φ(ξ, t) = ξ + A sin(2 pi ξ) sin(2 pi t)/(2 pi): w = ∂ₜΦ, read at the reference
    coordinate, is integrated by SSP-RK3. On a rate of t alone its stages (0, 1, 1/2) and weights
    (1/6, 1/6, 2/3) are Simpson's rule, so the nodes follow the map to order four."""
    del isolated_native_cache, native_cxx, kokkos_root
    errors = []
    t_end = STEPS*DT
    for refinement in (1, 2):
        steps, dt = STEPS*refinement, DT/refinement
        world, _, report, members, configuration = _run(
            "G03 lot M1a members, map, dt/%d" % refinement, form="map", steps=steps, dt=dt)
        oracle = ssprk3_oracle(configuration, steps, dt)
        with collective_check(world):
            assert report.accepted_steps == steps
            _check_members(members, steps)
            np.testing.assert_allclose(members, oracle, rtol=0, atol=8*EPS)
        errors.append(_map_error(members, configuration, t_end))
    order = math.log2(errors[0]/errors[1])
    with collective_check(world):
        assert errors[0] > 1e3*EPS, errors
        assert 3.7 <= order <= 4.3, (errors, order)
    record_property("map_errors", errors)
    record_property("observed_order", order)


def test_a_rejected_attempt_restores_the_members(
        isolated_native_cache, native_cxx, kokkos_root, record_property):
    """Every attempt is rejected by a guard that never holds: the run fails, and X, V, the time
    and the macro step are those of the accepted state, bitwise."""
    del isolated_native_cache, native_cxx, kokkos_root
    import pops
    from tests.python.support.native_execution_context import artifact_execution_context
    from tests.python.support.public_native_profile import compile_profile_plan_once, native_profile

    _, world = native_profile(dimension=1, scope="G03 lot M1a rejected attempt")
    case, layout, values, configuration = collective_call(world, lambda: build_members(reject=True))
    plan = collective_call(world, lambda: pops.resolve(pops.validate(case), layout=layout))
    artifact = compile_profile_plan_once(world, plan, route="G03 lot M1a rejected attempt",
                                         compile_artifact=pops.compile)
    runtime = collective_call(world, lambda: pops.bind(
        artifact, initial_values=values,
        resources={"execution_context": artifact_execution_context(artifact)}))

    def state():
        return np.asarray(runtime.state_global("configuration")).copy()
    before = collective_call(world, state)

    def failed():
        try:
            pops.run(runtime, t_end=DT, max_steps=1, console=False)
        except RuntimeError as error:
            return str(error)
        raise AssertionError("an attempt that the guard rejects was accepted")
    message = collective_call(world, failed)
    after = collective_call(world, state)
    with collective_check(world):
        assert "never_accepted" in message
        assert after.tobytes() == before.tobytes()
        assert before.tobytes() == configuration.initial_members().tobytes()
        assert runtime.time() == 0.0 and runtime.macro_step() == 0
    record_property("failure", message)


def _inverting_trial(*, transient, split):
    import pops
    from pops.analytic import coordinate, time
    from pops.domain import CartesianDomain
    from pops.frames import Cartesian1D
    from pops.layouts import Uniform
    from pops.mesh import CartesianGrid, GeometryEvolution
    from pops.mesh.layout_plan import LayoutPlanBuilder
    from pops.mesh.moving_configuration import MovingConfiguration
    from pops.time import FixedDt

    case = pops.Case("moving_trial")
    program = pops.Program("moving_trial")
    frame = CartesianDomain("moving_trial", (0.0,), (1.0,)).frame(Cartesian1D())
    x, t = coordinate(frame, frame.axes[0]), time(program.clock)
    dt = 0.25
    values, layouts = {}, {}
    names = ("a_valid", "z_inverting") if split else ("z_inverting",)
    for name in names:
        invert = name == "z_inverting"
        layout = Uniform(CartesianGrid(frame=frame, cells=(4 if invert else 8,)))
        displacement = t - 4*t*t if invert and transient else t
        law = x + (8 if invert else 1)*displacement*x*(1 - x)
        configuration = MovingConfiguration(layout, node_rate=GeometryEvolution((law,)))
        members = configuration.members(case, name)
        temporal = program.state(members.block[members.state])
        k0 = members.rate(temporal.n)
        if transient:
            stage = program.stage(name + "_heun", c=1)
            u1 = program.value(name + "_stage", temporal.n + program.dt*k0, at=stage)
            k1 = members.rate(u1)
            update = temporal.n + program.dt*(0.5*k0 + 0.5*k1)
        else:
            update = temporal.n + program.dt*k0
        program.commit(temporal.next, program.value(name + "_candidate", update,
                                                    at=temporal.next.point))
        values.update(members.initial_values())
        layouts[name] = layout
    program.step_strategy(FixedDt(dt))
    case.program(program)
    validated = pops.validate(case)
    if not split:
        return pops.resolve(validated, layout=layout), values, dt, names
    subjects = validated.layout_subjects()
    builder = LayoutPlanBuilder(validated.owner_path.canonical())
    providers = {builder.layout(name, descriptor): descriptor for name, descriptor in layouts.items()}
    by_name = {handle.local_id: handle for handle in providers}
    for block in subjects.blocks:
        builder.assign_block(block, by_name[block.local_id])
    for state in subjects.states:
        builder.assign_state(state, by_name[state.block_ref.local_id])
    plan = builder.resolve(**subjects.to_dict())
    return pops.resolve(validated, layout=plan, layout_providers=providers), values, dt, names


@pytest.mark.parametrize("transient,split", [(True, False), (False, False), (True, True)])
def test_moving_geometry_rejects_inversion_before_stage_use_and_publication(
        isolated_native_cache, native_cxx, kokkos_root, monkeypatch, transient, split):
    import pops
    from pops._bootstrap import StepAttemptRejected
    from tests.python.support.native_execution_context import artifact_execution_context
    from tests.python.support.public_native_profile import (
        compile_profile_plan_once, native_profile, rank_and_size,
    )

    del isolated_native_cache, native_cxx, kokkos_root
    scope = "moving geometric trial %s/%s" % (transient, split)
    _, world = native_profile(dimension=1, scope=scope)
    plan, values, dt, names = collective_call(
        world, lambda: _inverting_trial(transient=transient, split=split))
    artifact = compile_profile_plan_once(world, plan, route=scope, compile_artifact=pops.compile)

    def bind(initial):
        return pops.bind(artifact, initial_values=initial,
                         resources={"execution_context": artifact_execution_context(artifact)})

    if not transient:
        # A single bad rank must refuse the bind on every participant, before installing state.
        rank, _ = rank_and_size(world)
        for component, invalid in ((0, float("nan")), (1, float("inf")), (2, 0.0), (2, -0.125)):
            bad = {handle: array.copy() for handle, array in values.items()}
            if rank == 0:
                (array,) = bad.values()
                array[component, 0] = invalid

            def reject_bind():
                with pytest.raises(ValueError, match="NotAdmissible"):
                    bind(bad)
            collective_call(world, reject_bind)

    runtime = collective_call(world, lambda: bind(values))
    pre_rollback_times = []
    if split:
        from dataclasses import replace

        executor = runtime._executor
        restore = executor._restore_rejected_native_attempt
        expected_guards = tuple(
            guard.to_data() for row in artifact.layout_programs
            for guard in row.program.program.transaction_plan().guards
        )
        with collective_check(world):
            assert len(expected_guards) == 6
            assert runtime.program_report().step_transaction["guards"] == list(expected_guards)
            assert runtime.bound_snapshot.to_dict()["step_transaction"]["guards"] == list(expected_guards)

        child = executor._engines[executor._block_layouts[names[0]]]
        transaction = child._step_transaction_plan
        with monkeypatch.context() as patched:
            renamed = replace(transaction.guards[0], name="other guard")
            patched.setattr(child, "_step_transaction_plan",
                           replace(transaction, guards=(renamed, *transaction.guards[1:])))
            with pytest.raises(RuntimeError, match="child transaction report differs"):
                runtime.program_report()

        def check_children_at_rejection(*args):
            pre_rollback_times.extend(executor._engines[executor._block_layouts[name]].time()
                                      for name in names)
            return restore(*args)

        monkeypatch.setattr(executor, "_restore_rejected_native_attempt", check_children_at_rejection)

    def states():
        return tuple(np.asarray(runtime.state_global(name)).tobytes() for name in names)

    before = collective_call(world, states)

    def run():
        with pytest.raises(StepAttemptRejected) as failed:
            pops.run(runtime, t_end=dt, max_steps=1, console=False)
        return failed.value.phase, str(failed.value)

    phase, message = collective_call(world, run)
    after = collective_call(world, states)
    with collective_check(world):
        # Heun's second stage uses V=-1/8 in the last cell. Its final V would be 1/4
        # because the stage velocities are opposite. Euler reaches V=-1/8 at commit.
        assert phase == "trial_failure"
        assert "NotAdmissible" in message
        assert ("moving geometry stage" if transient else "moving geometry commit") in message
        if split:
            assert pre_rollback_times == [dt, 0.0]
        assert before == after == tuple(array.tobytes() for array in values.values())
        assert runtime.time() == 0.0 and runtime.macro_step() == 0


def test_map_domain_is_checked_before_motion_at_the_reached_stage(
        isolated_native_cache, native_cxx, kokkos_root):
    import pops
    from pops._bootstrap import StepAttemptRejected
    from pops.analytic import coordinate, time, where
    from pops.domain import CartesianDomain
    from pops.frames import Cartesian1D
    from pops.layouts import Uniform
    from pops.mesh import CartesianGrid, GeometryEvolution
    from pops.mesh.moving_configuration import MovingConfiguration
    from pops.time import FixedDt
    from tests.python.support.native_execution_context import artifact_execution_context
    from tests.python.support.public_native_profile import compile_profile_plan_once, native_profile

    del isolated_native_cache, native_cxx, kokkos_root
    scope = "moving rational map with guarded selection"
    _, world = native_profile(dimension=1, scope=scope)

    def build():
        case = pops.Case("guarded_rational_map")
        program = pops.Program("guarded_rational_map")
        frame = CartesianDomain("guarded_rational_map", (0.0,), (1.0,)).frame(Cartesian1D())
        layout = Uniform(CartesianGrid(frame=frame, cells=(4,)))
        x, t = coordinate(frame, frame.axes[0]), time(program.clock)
        law = (x + 0.125*x*(1-x)*t/(2-t)
               + where(x > 1, 0/(t - 0.25), 0)
               + where(x > 0.75, 0/(t - 0.5), 0))
        config = MovingConfiguration(layout, node_rate=GeometryEvolution((law,)))
        members = config.members(case)
        state = program.state(members.block[members.state])
        first = members.rate(state.n)
        stage = program.stage("heun", c=1)
        trial = program.value("trial", state.n + program.dt*first, at=stage)
        second = members.rate(trial)
        candidate = state.n + program.dt*(0.5*first + 0.5*second)
        program.commit(state.next, program.value("candidate", candidate, at=state.next.point))
        program.step_strategy(FixedDt(0.25))
        case.program(program)
        return pops.resolve(pops.validate(case), layout=layout), members.initial_values(), config

    plan, initial, config = collective_call(world, build)
    artifact = compile_profile_plan_once(world, plan, route=scope, compile_artifact=pops.compile)
    runtime = collective_call(world, lambda: pops.bind(
        artifact, initial_values=initial,
        resources={"execution_context": artifact_execution_context(artifact)}))

    # The pole at t=1/4 lies outside the selected branch at every reference node.
    report = collective_call(world, lambda: pops.run(runtime, t_end=0.25, max_steps=1, console=False))
    accepted = collective_call(world, lambda: np.asarray(runtime.state_global("configuration")).copy())
    with collective_check(world):
        assert report.accepted_steps == 1
        xi = config.reference_nodes()
        left, right, _ = accepted
        increment = 0.25*0.5*(0.25*xi*(1-xi)/2**2 + 0.25*xi*(1-xi)/(2-0.25)**2)
        np.testing.assert_allclose(np.concatenate((left, right[-1:])), xi + increment, rtol=0, atol=4*EPS)

    def reject():
        with pytest.raises(StepAttemptRejected) as failed:
            pops.run(runtime, t_end=0.5, max_steps=1, console=False)
        return failed.value.phase, str(failed.value)

    phase, failure = collective_call(world, reject)
    restored = collective_call(world, lambda: np.asarray(runtime.state_global("configuration")).copy())
    with collective_check(world):
        assert phase == "trial_failure"
        assert "moving rate domain stage" in failure and "NotAdmissible" in failure
        assert restored.tobytes() == accepted.tobytes()
        assert runtime.time() == 0.25 and runtime.macro_step() == 1
