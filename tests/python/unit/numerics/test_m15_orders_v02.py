"""S04 formula/spectrum and source emission checks, not native qualification."""
import numpy as np
import pops
import pytest

from examples.api_v02.models_M15_orders import (
    build_m15_orders, hankel_margins, hyqmom_flux, initial_moments,
    reference_flux, reference_rhs, reference_spectrum)


def test_corpus_dispatcher_uses_s04_order_four_on_a_true_dim1_mesh():
    from examples.api_v02.models_M15_M28 import build_model
    spec = build_model("M15")
    plan = pops.resolve(pops.validate(spec.case), layout=spec.layout)
    assert plan.resolved_dimension == 1
    assert "S04" in spec.profile and "order4" in spec.profile
    assert next(iter(spec.initial_values.values())).shape == (5, 16)


@pytest.mark.parametrize("order",(2,4))
def test_chebyshev_formula_matches_independent_central_formula_and_jacobi_spectrum(order):
    raw=initial_moments(order,16,"smooth")
    np.testing.assert_allclose(np.asarray(hyqmom_flux(raw)),reference_flux(raw),rtol=2e-14,atol=2e-14)
    for state in raw.T:
        columns=[]
        for i in range(order+1):
            shifted=state.astype(complex);shifted[i]+=1e-30j
            columns.append(np.asarray(hyqmom_flux(shifted)).imag/1e-30)
        spectrum=np.linalg.eigvals(np.stack(columns,axis=1))
        assert np.max(abs(spectrum.imag))<1e-12
        np.testing.assert_allclose(np.sort(spectrum.real),reference_spectrum(state),atol=2e-12,rtol=2e-12)


def test_non_gaussian_order_four_closure_is_not_the_input_measure_or_zero_central_fifth():
    raw=initial_moments(4,16,"smooth")[:,3]
    u=raw[1]/raw[0]
    terminal=hyqmom_flux(raw)[-1]
    # The exact initial four-atom fifth moment is not the HyQMOM closure.
    exact=initial_moments(5,16,"smooth")[5,3]
    central=terminal-5*u*raw[4]+10*u*u*raw[3]-10*u**3*raw[2]+4*raw[0]*u**5
    assert abs(terminal-exact)>1e-3
    assert abs(central)>1e-3


def test_approach_to_hankel_boundary_has_coalescing_real_spectrum_and_refuses_invalid_side():
    # S3=-1, H4=epsilon: the exact family plotted in S04 Figure1/section5.2.
    gaps=[]
    for epsilon in (1e-2,1e-4,1e-6):
        raw=np.array([1.,0.,1.,-1.,2.+epsilon])
        assert min(hankel_margins(raw))>0
        spectrum=reference_spectrum(raw)
        assert np.isfinite(spectrum).all()
        gaps.append(np.min(np.diff(spectrum)))
        np.testing.assert_allclose(hyqmom_flux(raw),reference_flux(raw),atol=1e-13)
    assert gaps[2]<gaps[1]<gaps[0]
    for raw in ([1.,0.,1.,-1.,1.9],[1.,0.,0.,0.,0.],[-1.,0.,1.]):
        with pytest.raises(ValueError,match="Hankel"):
            reference_spectrum(raw)


@pytest.mark.parametrize("order",(2,4))
@pytest.mark.parametrize("regime",("smooth","riemann"))
def test_both_orders_and_regimes_resolve_dim1_and_wrong_spatial_state_fails(order,regime):
    from pops.codegen.program_codegen import emit_cpp_program
    from pops.codegen.program_models import ProgramModelGraph
    spec=build_m15_orders(order=order,regime=regime)
    plan=pops.resolve(pops.validate(spec.case),layout=spec.layout)
    assert plan.resolved_dimension==1
    cpp=emit_cpp_program(plan.time,model_graph=ProgramModelGraph.from_resolved_blocks(plan.blocks))
    assert "strict_candidate_domain" in cpp
    initial=next(iter(spec.initial_values.values()))
    expected=initial.copy()
    for _ in range(4):expected+=spec.dt*reference_rhs(expected)
    assert spec.invariants({"moments":expected})["passed"]
    wrong=spec.invariants({"moments":np.roll(expected,1,axis=1)})
    assert wrong["inventory_error"]<2e-12
    assert wrong["minimum_hankel_determinant"]>0
    assert not wrong["passed"]
