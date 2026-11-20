"""G03 lot A (ROM-21): ``nonlinear_inventory_witness``, a nonlinear accumulation run natively.

The state holds U, the cell averages of Q(T) = T + T²; T is recovered from U on each point by the
bracketed Newton root of Q(T) = U on T >= -1/2, and the conduction -k ∂ₓT reads it like a
definition over U (note §6). The oracle is written by hand on U with the closed root
T = (√(1 + 4U) − 1)/2, which PoPS never sees.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from examples.api_v02.models_G03_accumulation import (
    CELLS, CONDUCTIVITY, DT, STEPS, build_witness)
from examples.api_v02.models_static_geometry import graded_nodes
from tests.python.integration.runtime.test_n17_static_geometry_graphs import run_graph
from tests.python.support.collective_checks import collective_call, collective_check

pytestmark = [pytest.mark.compiler, pytest.mark.kokkos, pytest.mark.native_loader]


def closed_root(u):
    return (np.sqrt(1.0 + 4.0*u) - 1.0)/2.0


def _operator(nodes):
    measures = np.diff(nodes)
    distances = np.diff(0.5*(nodes[:-1] + nodes[1:]))

    def divergence(t):
        faces = np.zeros(len(nodes))
        faces[1:-1] = CONDUCTIVITY*np.diff(t)/distances
        return np.diff(faces)/measures
    return divergence


def inventory_oracle(nodes, initial, steps, dt):
    """SSP-RK3 in its increment form on U, the rate reading T(U): the scheme of lot A."""
    divergence = _operator(nodes)

    def rate(u):
        return divergence(closed_root(u))

    u = np.array(initial, dtype=float)
    for _ in range(steps):
        k0 = rate(u)
        k1 = rate(u + dt*k0)
        k2 = rate(u + dt*(0.25*k0 + 0.25*k1))
        u = u + dt*(k0/6 + k1/6 + (2/3)*k2)
    return u


def chain_rule_variant(nodes, initial, steps, dt):
    """The variant that K6 forbids: T advanced by R/Q'(T), then U = Q(T). It conserves nothing
    exactly, and it is not the scheme of lot A."""
    divergence = _operator(nodes)

    def rate(t):
        return divergence(t)/(1.0 + 2.0*t)

    t = closed_root(np.array(initial, dtype=float))
    for _ in range(steps):
        k0 = rate(t)
        k1 = rate(t + dt*k0)
        k2 = rate(t + dt*(0.25*k0 + 0.25*k1))
        t = t + dt*(k0/6 + k1/6 + (2/3)*k2)
    return t + t*t


def _run(scope, steps=STEPS, **keywords):
    holder = {}

    def build():
        case, layout, values, initial, stored = build_witness(**keywords)
        holder["stored"] = stored
        return case, layout, values, initial

    world, runtime, report, initial = run_graph(build, dimension=1, scope=scope, steps=steps,
                                                dt=DT)
    state = np.asarray(collective_call(world, lambda: runtime.state_global("rod")))[0]
    return world, runtime, report, state, initial[0]


def test_the_witness_advances_the_stored_inventory_on_the_graded_cells(
        isolated_native_cache, native_cxx, kokkos_root):
    """On the 64 graded cells: the run is the update of U written by hand, the chain-rule variant
    differs from it, and the inventory Σ μ U drifts by at most N·eps·Σ|μ U|, N = steps × stages ×
    cells (the bound of the lot's pr/ file; O1 of the review). Measured when the lot was written:
    agreement 2.4e-14, chain-rule departure 7.5e-10, drift 0.0 under an exact sum (bound 3.5e-13)."""
    del isolated_native_cache, native_cxx, kokkos_root
    world, _, report, state, initial = _run("G03 A graded witness")
    nodes = graded_nodes(CELLS)
    measures = np.diff(nodes)
    expected = inventory_oracle(nodes, initial, STEPS, DT)
    variant = chain_rule_variant(nodes, initial, STEPS, DT)
    agreement = float(np.max(np.abs(state - expected)/np.abs(expected)))
    departure = float(np.max(np.abs(variant - expected)/np.abs(expected)))
    drift = abs(math.fsum(measures*state) - math.fsum(measures*initial))
    bound = STEPS*3*CELLS*np.finfo(float).eps*math.fsum(np.abs(measures*initial))
    with collective_check(world):
        assert report.accepted_steps == STEPS
        assert measures.max()/measures.min() > 90
        assert agreement <= 1e-12
        assert departure >= 1e3*max(agreement, 1e-16)
        assert drift <= bound


def test_the_commit_adds_the_combination_of_the_stage_rates_bitwise(
        isolated_native_cache, native_cxx, kokkos_root):
    """U_new = U_old + Δ in binary64, Δ = dt (k0/6 + k1/6 + 2 k2/3) as the program forms it: the
    history of the last step holds U_old and Δ, and their sum is the state, bit for bit. The
    finite difference U_new − U_old is not the test, since (a + b) − a ≠ b in binary64."""
    del isolated_native_cache, native_cxx, kokkos_root
    world, runtime, report, state, initial = _run("G03 A uniform increment", graded=False)

    def ring(name):
        return [np.asarray(collective_call(world, lambda slot=slot: runtime.history_global(
            name, slot))).reshape(-1) for slot in range(runtime.history_depth(name))]

    previous, increment = ring("previous"), ring("increment")
    nodes = np.arange(CELLS + 1)/CELLS
    expected = inventory_oracle(nodes, initial, STEPS, DT)
    with collective_check(world):
        assert report.accepted_steps == STEPS
        assert len(previous) == len(increment) == 2
        # The ring holds the last two steps; the newest is the one whose sum is the state, and
        # the older one's sum is the newest's U_old. Both commits are checked bit for bit.
        newest = [slot for slot in range(2) if np.array_equal(previous[slot] + increment[slot],
                                                              state)]
        assert len(newest) == 1
        older = 1 - newest[0]
        assert np.array_equal(previous[older] + increment[older], previous[newest[0]])
        assert np.all(increment[newest[0]] != 0.0)
        np.testing.assert_allclose(state, expected, rtol=1e-12, atol=0.0)
