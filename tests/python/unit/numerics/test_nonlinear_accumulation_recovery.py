"""G03 lot A (ROM-21): a nonlinear accumulation with its coordinates recovered on each point.

Checks of note §6: the recovery q(U) (its host twin, its derivative, its native form),
the declaration ``model.accumulation`` and its refusals, the initial projection encode(Q(e)), and
the explicit route that reads q like a definition over U.
"""
from __future__ import annotations

import math
from pathlib import Path
import subprocess

import numpy as np
import pytest

import pops
from pops._ir.expr import Exp, Var
from pops._ir.local_root import (
    BOUND_KIND, LocalRoot, PointwiseRoot, derivative_interval)
from pops._ir.lowering import diff
from pops.surface import AmbiguousElaboration

from examples.api_v02.models_G03_accumulation import (
    CELLS, LOWER, build_witness, coordinate_profile, project_accumulation)
from examples.api_v02.models_static_geometry import graded_nodes


def _root(*, lower=LOWER, upper=math.inf, law=None, **keywords):
    q = Var("pops_root_T", BOUND_KIND)
    return PointwiseRoot(q + q*q if law is None else law(q), q, Var("e", "cons"),
                         lower=lower, upper=upper,
                         recovery=LocalRoot(**{"tolerance": 1e-13, "max_iterations": 200,
                                               **keywords}), name="T"), q


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
    assert native.count("for (std::int64_t pops_root_it = 0; pops_root_it < 200LL;") == 1
    assert "quiet_NaN" in native and "pops_root_tolerance = 1e-13" in native
    assert values[1].startswith("(pops::Real(1) / ")


def _compiled_recovery(root, target, targets, native_cxx, tmp_path):
    from pops.codegen.cpp_writer import _cse_emit

    derivative = diff(root, target)
    declarations, values = _cse_emit([root, derivative], "pops::Real", "  ")
    source = tmp_path / "recover.cpp"
    source.write_text("\n".join((
        "#include <pops/core/foundation/types.hpp>",
        "#include <cmath>",
        "#include <cstdio>",
        "#include <cstdlib>",
        "#include <limits>",
        "namespace Kokkos {",
        "using std::exp;",
        "using std::fabs;",
        "using std::fmax;",
        "using std::isfinite;",
        "namespace Experimental {",
        "template<class T> struct infinity { static constexpr T value = std::numeric_limits<T>::infinity(); };",
        "template<class T> struct quiet_NaN { static constexpr T value = std::numeric_limits<T>::quiet_NaN(); };",
        "}",
        "}",
        "void recover(pops::Real %s) {" % target.name,
        *declarations,
        f'  std::printf("%.17g %.17g\\n", {values[0]}, {values[1]});',
        "}",
        "int main(int argc, char** argv) {",
        "  for (int arg = 1; arg < argc; ++arg) recover(std::strtod(argv[arg], nullptr));",
        "}",
    )), encoding="utf-8")
    include = Path(__file__).resolve().parents[4] / "include"
    executable = tmp_path / "recover"
    compiled = subprocess.run(
        [native_cxx, "-std=c++20", "-O2", "-fno-fast-math", "-ffp-contract=off",
         "-I" + str(include), str(source), "-o", str(executable)],
        capture_output=True, text=True,
    )
    assert compiled.returncode == 0, compiled.stderr
    executed = subprocess.run([str(executable), *map(str, targets)],
                              capture_output=True, text=True, check=True)
    return [tuple(map(float, line.split())) for line in executed.stdout.splitlines()]


@pytest.mark.compiler
@pytest.mark.parametrize(("coordinate", "component", "budget"), (
    ("x", "e", 200),
    ("f", "e", 200),
    ("u", "e", 200),
    ("T", "pops_root_u", 200),
    ("x", "e", 10001),
    ("u", "e", (1 << 63) - 1),
    ("x", "e", 2),
))
def test_the_compiled_recovery_and_its_derivative_use_the_declared_coordinate(
        coordinate, component, budget, native_cxx, tmp_path):
    from pops._ir.application import substitute_quantities
    from pops.lib.quadrature import GaussLegendre
    from pops.representations.definition import cell_average

    model = pops.Model("recovery_binding", frame=pops.domain.CartesianDomain(
        "recovery_binding", (0.0,), (1.0,)).frame(pops.frames.Cartesian1D()))
    state = model.state("stored", components=(component,))
    q = model.coordinate(coordinate)
    root = model.accumulation(
        state, coordinates=q, law=q + q*q, admissible=(LOWER, None),
        representation=cell_average(1, GaussLegendre(2)),
        recovery=LocalRoot(tolerance=1e-13, max_iterations=budget),
    ).recovery
    stored = state[0]
    target = Var(component, "cons")
    root = substitute_quantities(root, {(stored.handle, stored.index): target})
    derivative = diff(root, target)
    targets = (5.0,) if budget == 2 else (0.1, 0.75, 3.0)
    results = _compiled_recovery(root, target, targets, native_cxx, tmp_path)
    for value, (recovered, slope) in zip(targets, results, strict=True):
        host = root.solve(value)
        expected = (math.sqrt(1 + 4*value) - 1)/2
        if budget == 2:
            assert math.isnan(host) and math.isnan(recovered) and math.isnan(slope)
        else:
            assert recovered == pytest.approx(expected, rel=1e-12)
            assert recovered == pytest.approx(host, rel=1e-12)
            assert slope == pytest.approx(1/(1 + 2*expected), rel=1e-12)
            assert slope == pytest.approx(float(derivative.eval({component: value})), rel=1e-12)


@pytest.mark.compiler
@pytest.mark.parametrize(("upper", "budget", "law", "targets"), (
    (1.0, 100, None, (
        (math.nextafter(0.0, -math.inf), False), (0.0, True), (1e-7, True), (0.56, True),
        (math.nextafter(2.0, 0.0), True), (2.0, False), (math.nextafter(2.0, math.inf), False),
        (math.inf, False), (math.nan, False))),
    (math.inf, 100, lambda q: 1.0 - Exp(-q), (
        (-1e-7, False), (0.0, True), (1e-7, True), (0.5, True),
        (1.0, False), (math.nextafter(1.0, math.inf), False))),
    (1.0, 1, None, ((0.0, True), (1e-7, False), (2.0, False))),
    (1.0, 2, None, ((1e-7, True), (0.56, False), (2.0, False))),
    (math.inf, 2, None, ((1e-7, True), (2.0, False))),
    (math.inf, 3, None, ((2.0, True),)),
))
def test_admissibility_precedes_the_residual_check_in_host_and_generated_cpp(
        upper, budget, law, targets, native_cxx, tmp_path):
    root, _ = _root(lower=0.0, upper=upper, law=law, tolerance=1e-6, max_iterations=budget)
    results = _compiled_recovery(root, root.target, [value for value, _ in targets],
                                 native_cxx, tmp_path)
    for (target, succeeds), (recovered, slope) in zip(targets, results, strict=True):
        host = root.solve(target)
        if succeeds:
            assert 0.0 <= host < upper and 0.0 <= recovered < upper
            assert abs(float(root.law.eval({root.bound.name: host})) - target) <= 1e-6*(1 + abs(target))
            assert recovered == pytest.approx(host, rel=1e-14, abs=1e-15)
            assert math.isfinite(slope)
        else:
            assert math.isnan(host) and math.isnan(recovered)


@pytest.mark.parametrize("foreign_kind", ["coordinate", "same_name_aux", "same_name_prim"])
def test_accumulation_checks_the_raw_law_before_differentiating(foreign_kind):
    from pops.lib.quadrature import GaussLegendre
    from pops.representations.definition import cell_average

    model = pops.Model("closed_accumulation", frame=pops.domain.CartesianDomain(
        "closed_accumulation", (0.0,), (1.0,)).frame(pops.frames.Cartesian1D()))
    state = model.state("stored", components=("e",))
    q = model.coordinate("q")
    foreign = {
        "coordinate": model.coordinate("pops_root_x"),
        "same_name_aux": Var(q.name, "aux"),
        "same_name_prim": Var(q.name, "prim"),
    }[foreign_kind]
    options = dict(admissible=(0.0, None), representation=cell_average(1, GaussLegendre(2)),
                   recovery=LocalRoot(tolerance=1e-6, max_iterations=100))
    with pytest.raises(ValueError, match="outside its coordinate declaration"):
        model.accumulation(state, coordinates=q, law=q + q*q + foreign, **options)
    with pytest.raises(ValueError, match="takes one undefined model.coordinate"):
        model.accumulation(state, coordinates=Var(q.name, "prim"), law=q + q*q, **options)
    root = model.accumulation(state, coordinates=q, law=q + q*q, **options).recovery
    assert root.solve(0.56) == pytest.approx(0.4, abs=1e-6)


def test_the_root_node_rejects_an_unbound_term_even_if_its_derivative_erases_it():
    with pytest.raises(ValueError, match="outside its coordinate declaration"):
        _root(law=lambda q: q + q*q + Var(q.name, "aux"))


@pytest.mark.parametrize("law", [lambda q: 2 - q - q*q, lambda q: q - q])
def test_decreasing_and_constant_laws_remain_outside_the_increasing_recovery(law):
    from pops.lib.quadrature import GaussLegendre
    from pops.representations.definition import cell_average

    model = pops.Model("increasing_accumulation", frame=pops.domain.CartesianDomain(
        "increasing_accumulation", (0.0,), (1.0,)).frame(pops.frames.Cartesian1D()))
    state = model.state("stored", components=("e",))
    q = model.coordinate("q")
    with pytest.raises(AmbiguousElaboration, match="not proved increasing.*admissible="):
        model.accumulation(
            state, coordinates=q, law=law(q), admissible=(0.0, 1.0),
            representation=cell_average(1, GaussLegendre(2)),
            recovery=LocalRoot(tolerance=1e-6, max_iterations=100))


@pytest.mark.parametrize("budget", [False, 0, -1, 1.0, 1 << 63])
def test_the_local_root_budget_is_a_positive_signed_64_bit_integer(budget):
    with pytest.raises(ValueError, match="max_iterations"):
        LocalRoot(tolerance=1e-13, max_iterations=budget)


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
