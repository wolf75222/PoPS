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
