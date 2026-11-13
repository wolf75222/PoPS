"""G03 lot A (ROM-21): a nonlinear accumulation with its coordinates recovered on each point.

Source-only checks of note §6: the recovery q(U) (its host twin, its derivative, its native form),
the declaration ``model.accumulation`` and its refusals, the initial projection encode(Q(e)), and
the explicit route that reads q like a definition over U.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

import pops
from pops._ir.expr import Var
from pops._ir.local_root import (
    BOUND_KIND, LocalRoot, PointwiseRoot, derivative_interval)
from pops._ir.lowering import diff
from pops.surface import AmbiguousElaboration

from examples.api_v02.models_G03_accumulation import (
    CELLS, LOWER, build_witness, coordinate_profile, project_accumulation)
from examples.api_v02.models_static_geometry import graded_nodes


def _root(**keywords):
    q = Var("pops_root_T", BOUND_KIND)
    return PointwiseRoot(q + q*q, q, Var("e", "cons"), lower=LOWER, upper=math.inf,
                         recovery=LocalRoot(**keywords), name="T"), q


@pytest.mark.parametrize("value", [0.0, 1e-9, 0.3, 0.75, 2.0, 1e3, -0.25, -0.2])
def test_the_host_recovery_meets_its_tolerance_on_the_admissible_set(value):
    root, _ = _root(tolerance=1e-13)
    q = root.solve(value)
    assert q >= LOWER
    assert abs(q + q*q - value) <= 1e-13*(1 + abs(value))
    assert q == pytest.approx((math.sqrt(1 + 4*value) - 1)/2, rel=1e-12, abs=1e-12)


def test_a_value_below_the_admissible_set_fails_closed():
    """U < Q(-1/2) = -1/4 has no admissible root: NaN, which every recovery reads as a failure."""
    root, _ = _root()
    assert math.isnan(root.solve(-0.3))
    assert math.isnan(root.solve(math.nan))
    assert math.isnan(_root(max_iterations=2)[0].solve(5.0))


def test_the_derivative_of_the_recovery_is_one_over_the_slope_of_the_law():
    """dq/dU = 1/Q'(q): the native diffusion reads it for its frequency, never a chain rule."""
    root, _ = _root()
    derivative = diff(root, Var("e", "cons"))
    for value in (0.1, 0.6, 3.0):
        q = root.solve(value)
        assert float(derivative.eval({"e": value})) == pytest.approx(1/(1 + 2*q), rel=1e-12)


def test_the_native_recovery_is_one_bracketed_iteration_shared_by_its_readers():
    from pops.codegen.cpp_writer import _cse_emit
    root, _ = _root()
    lines, values = _cse_emit([root, diff(root, Var("e", "cons"))], "pops::Real", "  ")
    native = "\n".join(lines)
    assert native.count("for (int pops_root_it = 0; pops_root_it < 200;") == 1
    assert "quiet_NaN" in native and "pops_root_tolerance = 1e-13" in native
    assert values[1].startswith("(pops::Real(1) / ")


def test_the_enclosure_of_the_slope_decides_monotonicity():
    q = Var("pops_root_T", BOUND_KIND)
    slope = diff(q + q*q, q)
    assert derivative_interval(slope, q.name, -0.5, math.inf)[0] >= 0
    assert derivative_interval(slope, q.name, -1.0, math.inf)[0] < 0


def test_the_declaration_records_a_recovered_law():
    _, _, _, _, stored = build_witness()
    assert not stored.is_identity and stored.advances_stored_state
    assert stored.recovery.lower == LOWER and stored.recovery.upper is None
    data = stored.to_data()
    assert data["kind"] == "discrete_constitutive_map" and "recovery" in data


def test_a_representation_without_a_rule_is_refused_naming_quadrature():
    with pytest.raises(AmbiguousElaboration, match="quadrature="):
        build_witness(quadrature=False)


def test_a_law_not_increasing_on_its_admissible_set_is_refused_naming_admissible():
    """Q = q + q² is not injective on q >= -1: the selection is the admissible set (R8 item 5)."""
    with pytest.raises(AmbiguousElaboration, match="admissible="):
        build_witness(admissible=(-1.0, None))


def test_a_recovery_without_its_tolerance_is_refused():
    import examples.api_v02.models_G03_accumulation as witness
    model = pops.Model("G03_A_no_recovery", frame=pops.domain.CartesianDomain(
        "g03_a_no_recovery", (0.0,), (1.0,)).frame(pops.frames.Cartesian1D()))
    E = model.state("E", components=("e",))
    T = model.coordinate("T")
    with pytest.raises(AmbiguousElaboration, match="recovery=LocalRoot"):
        model.accumulation(E, coordinates=T, law=T + T*T, admissible=(LOWER, None),
                           representation=witness.cell_average(1, witness.GaussLegendre(2)))


def test_the_initial_value_is_the_rule_applied_to_the_law_of_the_profile():
    """K5, K6: initial={T: e} gives encode(Q(e)), which differs measurably from Q(encode(e))."""
    _, _, _, initial, stored = build_witness()
    nodes = graded_nodes(CELLS)
    encoded_law = project_accumulation(stored, coordinate_profile, nodes)
    np.testing.assert_array_equal(initial[0], encoded_law)
    averages = project_accumulation(
        type("Identity", (), {"quadrature": stored.quadrature, "coordinates": stored.coordinates,
                              "law": stored.coordinates})(), coordinate_profile, nodes)
    law_of_average = averages + averages*averages
    assert np.max(np.abs(encoded_law - law_of_average)) > 1e-6
    # Under one rule, encode(Q(e)) - Q(encode(e)) is the rule's variance of e on each cell, since
    # the linear part of Q cancels: nonnegative up to rounding.
    assert np.all(encoded_law - law_of_average > -1e-15)


def test_the_witness_resolves_on_the_explicit_route():
    """The balance is lifted (balance.py), and the diffusion reads T as a primitive over U."""
    case, layout, _, _, _ = build_witness()
    plan = pops.resolve(pops.validate(case), layout=layout)
    assert plan is not None


def test_a_balance_whose_coordinates_are_not_recovered_keeps_its_refusal():
    from pops._ir.balance import accumulation
    model = pops.Model("G03_A_unrecovered", frame=pops.domain.CartesianDomain(
        "g03_a_unrecovered", (0.0,), (1.0,)).frame(pops.frames.Cartesian1D()))
    E = model.state("E", components=("e",))
    mapping = accumulation(E, coordinates=E[0], law=E[0]*E[0],
                           representation={"output": "cell_average"},
                           quadrature={"rule": "gauss", "order": 2})
    assert not mapping.advances_stored_state
