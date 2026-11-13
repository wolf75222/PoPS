"""G03 lot M1b (ROM-21): a total transported on the members of a moving configuration, rank one.

The totals U of block "line" and the members (x_left, x_right, V) of block "configuration" advance
together by SSP-RK3; at each stage the rate of U reads ū = U/V and the members' rate of the same
stage (``moving_total_rate``, note §7.2, §7.6). The oracle is an independent NumPy assembly of the
route (``ssprk3_oracle``, M28 in rank one); CF11 is read bit for bit.
"""
from __future__ import annotations

import numpy as np
import pytest

from examples.api_v02.models_G03_moving_transport import (
    CELLS, CONSTANT, DT, GRADED_DT, STEPS, build_transport, initial_density, ssprk3_oracle)
from tests.python.integration.runtime.test_n17_static_geometry_graphs import run_graph
from tests.python.support.collective_checks import collective_call, collective_check

pytestmark = [pytest.mark.compiler, pytest.mark.kokkos, pytest.mark.native_loader]

EPS = np.finfo(float).eps


def _run(scope, *, dt=DT, steps=STEPS, **keywords):
    holder = {}

    def build():
        case, layout, values, configuration = build_transport(dt=dt, **keywords)
        holder["configuration"] = configuration
        return case, layout, values, None

    world, runtime, report, _ = run_graph(build, dimension=1, scope=scope, steps=steps, dt=dt)
    totals, members = collective_call(world, lambda: (
        np.asarray(runtime.state_global("line")).copy()[0],
        np.asarray(runtime.state_global("configuration")).copy()))
    return world, report, totals, members, holder["configuration"]


def _bound(values, steps):
    """The three stages of a step round each value a few times, relative to the largest."""
    return 3*steps*4*EPS*float(np.max(np.abs(values)))


def _check_members(members):
    left, right, volume = members
    assert left[1:].tobytes() == right[:-1].tobytes()
    assert np.all(volume > 0)


def test_a_constant_density_keeps_the_total_equal_to_u0_times_v_bit_for_bit(
        isolated_native_cache, native_cxx, kokkos_root, record_property):
    """CF11: from ū = 2, a power of two, U = 2 V after every step, bit for bit. The physical parts
    of the faces of a cell cancel exactly, and the swept parts give 2 (w_right - w_left), twice
    the rate of V."""
    del isolated_native_cache, native_cxx, kokkos_root
    world, report, totals, members, configuration = _run(
        "G03 lot M1b CF11", initial="constant")
    with collective_check(world):
        assert report.accepted_steps == STEPS
        _check_members(members)
        start = configuration.initial_members()
        assert np.max(np.abs(members[2] - start[2])) > 0.1*0.05*DT   # V moved
        assert totals.tobytes() == (CONSTANT*members[2]).tobytes()
    record_property("volume_change", float(np.max(np.abs(members[2] - start[2]))))


def test_a_total_follows_the_numpy_assembly_of_the_route_and_its_inventory(
        isolated_native_cache, native_cxx, kokkos_root, record_property):
    """M28 in rank one: Burgers from ū = 1 + sin(2 pi x)/2 on 32 uniform cells; the totals and
    the members against the NumPy assembly, and the inventory Σ U against the flux through the
    two ends integrated by the weights of the method."""
    del isolated_native_cache, native_cxx, kokkos_root
    world, report, totals, members, configuration = _run("G03 lot M1b Burgers")
    oracle, oracle_members, budget = ssprk3_oracle(configuration, STEPS, DT)
    start = configuration.initial_members()
    initial = initial_density(0.5*(start[0] + start[1]), "wave")*start[2]
    drift = float(np.sum(totals) - np.sum(initial) - budget)
    with collective_check(world):
        assert report.accepted_steps == STEPS
        _check_members(members)
        np.testing.assert_allclose(members, oracle_members, rtol=0, atol=8*EPS)
        np.testing.assert_allclose(totals, oracle, rtol=0, atol=_bound(totals, STEPS))
        # The ends let mass out: the inventory moved, by the budget of its two ends.
        assert abs(budget) > 1e-3
        assert abs(drift) <= 4*CELLS*EPS*float(np.sum(np.abs(totals)))
    record_property("oracle", float(np.max(np.abs(totals - oracle))))
    record_property("inventory_drift", drift)


def test_a_total_on_graded_cells_follows_the_numpy_assembly(
        isolated_native_cache, native_cxx, kokkos_root, record_property):
    """The members start from N17's graded segment (widths in a ratio of about 92); constant
    advection a = 1/2 on the members of the logical grid."""
    del isolated_native_cache, native_cxx, kokkos_root
    world, report, totals, members, configuration = _run(
        "G03 lot M1b graded advection", base="graded", law="advection", dt=GRADED_DT)
    oracle, oracle_members, _ = ssprk3_oracle(configuration, STEPS, GRADED_DT, law="advection")
    with collective_check(world):
        assert report.accepted_steps == STEPS
        _check_members(members)
        np.testing.assert_allclose(members, oracle_members, rtol=0, atol=8*EPS)
        np.testing.assert_allclose(totals, oracle, rtol=0, atol=_bound(totals, STEPS))
    record_property("oracle", float(np.max(np.abs(totals - oracle))))
