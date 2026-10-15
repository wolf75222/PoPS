"""Equation, independent quadrature and source-admission checks, without native execution."""
import numpy as np
import pops
import pytest

from examples.api_v02.method_nonlinear_interval_transport import (
    build_nonlinear_interval_transport, equilibrium, equilibrium_direction)
from examples.api_v02 import nonlinear_interval_transport_oracle as oracle


def test_nonlinear_isothermal_equilibrium_preserves_both_moments():
    rho = np.array((.8, 1.2, 1.1))
    momentum = np.array((-.07, .14, .02))
    values = np.asarray(equilibrium(rho, momentum, .25))
    np.testing.assert_allclose(values.sum(axis=0), rho, atol=3.e-16, rtol=0)
    np.testing.assert_allclose(values[2]-values[0], momentum, atol=3.e-17, rtol=0)
    mixture = np.asarray(equilibrium(rho.mean(), momentum.mean(), .25))
    assert np.max(abs(mixture-values.mean(axis=1))) > 1.e-3
    drho, dm = np.array((.03, -.12, .1)), np.array((-.2, .01, .06))
    exact = np.asarray(equilibrium_direction(rho, momentum, drho, dm, .25))
    independent = oracle.equilibrium(rho+1.e-30j*drho, momentum+1.e-30j*dm).imag/1.e-30
    np.testing.assert_allclose(exact, independent, atol=4.e-17, rtol=0)
    np.testing.assert_allclose(exact.sum(axis=0), drho, atol=4.e-17, rtol=0)
    np.testing.assert_allclose(exact[2]-exact[0], dm, atol=4.e-17, rtol=0)


def test_quadrature_converges_conserves_and_distinguishes_three_incorrect_methods():
    initial = oracle.initial_values()
    expected = oracle.reference(initial, .002, 1/24, 3, nodes=14)
    refined = oracle.reference(initial, .002, 1/24, 3, nodes=22)
    np.testing.assert_allclose(expected, refined, atol=7.e-16, rtol=0)
    for old, new in zip(oracle.moments(initial), oracle.moments(expected), strict=True):
        assert abs(old.sum()-new.sum()) < 4.e-14
    assert np.min(expected) > 0
    for mutant in ({"temporal_equilibrium": False}, {"flux_duration": .004},
                   {"transported_equilibrium": False}):
        wrong = oracle.reference(initial, .002, 1/24, 3, **mutant)
        assert np.max(abs(wrong-expected)) > 1.e-8


@pytest.mark.parametrize("retry", (False, True))
def test_nonlinear_interval_is_composed_and_lowered_without_new_core_operator(retry):
    from pops.codegen.program_codegen import emit_cpp_program
    from pops.codegen.program_source_map import source_map
    from pops.codegen.program_models import ProgramModelGraph
    spec = build_nonlinear_interval_transport(retry=retry)
    plan = pops.resolve(pops.validate(spec.case), layout=spec.layout)
    assert len(plan.blocks) == 1
    cpp = emit_cpp_program(plan.time, model_graph=ProgramModelGraph.from_resolved_blocks(plan.blocks))
    assert "POPS_HD" in cpp
    assert "ctx.attach_affine_flux_expression" in cpp
    assert "runtime_affine_spatial_stability" in cpp
    operations = {row["op"] for row in source_map(cpp)["operations"]}
    assert {"stencil_flux_divergence", "pointwise_expression", "linear_combine"} <= operations
    assert ("acceptance_guard" in operations) == retry
    expected = oracle.reference(oracle.initial_values(), .002, 1/24, 3)
    assert spec.invariants({"method": expected})["passed"]
