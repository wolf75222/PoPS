"""ADC-945: physical-map commutation on nested uniform cell-average supports.

The compiled providers implement M: 1x1v -> 1x velocity quadrature and
P: 1x -> 1x1v spatial pullback.  At all three declared resolutions, compare
them with independently integrated, asymmetric, nonconstant polynomial data.
For the two nested 2:1 pairs, verify R_x M_f = M_c R_xv and R_xv P_f = P_c R_x.
R denotes the explicit volume-weighted mathematical restriction below; this
does NOT execute native AMR restriction, regridding, or an AMR physical map.

Native scope: Dim2 storage, CPU float64, MPI size 1, one uniform patch per
independently owned layout, physical x in [0, 1], velocity v in [-2, 2].
Provider sessions are called inside a native transaction without evolution or
a field solve, isolating mapping properties from the lifecycle test.  These
claims are distinct from conservation, constant preservation, admissibility,
and discrete weighted dual work; none substitutes for commutation.
"""

from __future__ import annotations

from fractions import Fraction
from types import SimpleNamespace

import numpy as np
import pops
import pytest

from pops.runtime._physical_mapping import validate_physical_geometry
from tests.python.integration.runtime.test_physical_support_mapping import (
    REFINEMENTS,
    resolve_physical_case,
)
from tests.python.support.native_execution_context import artifact_execution_context


NESTED_PAIRS = tuple(zip(REFINEMENTS[:-1], REFINEMENTS[1:], strict=True))
ATOL = 3e-13


def _power_average(lower, upper, power):
    """Analytic integral of a monomial divided by its cell measure."""
    return (upper ** (power + 1) - lower ** (power + 1)) / ((power + 1) * (upper - lower))


def _polynomial_cell_averages(nx, nv):
    x_edges = np.arange(nx + 1, dtype=np.float64) / nx
    v_edges = -2 + 4 * np.arange(nv + 1, dtype=np.float64) / nv
    x, x2, x3 = (_power_average(x_edges[:-1], x_edges[1:], power) for power in (1, 2, 3))
    v, v2, v3, v4 = (
        _power_average(v_edges[:-1], v_edges[1:], power)[:, None] for power in (1, 2, 3, 4)
    )
    # Nonseparable polynomial: odd velocity terms and mixed x/v powers prevent
    # the input from being merely a constant or one separable Fourier mode.
    distribution = (
        1
        + 0.3 * x
        + 0.2 * x2
        + (0.4 - 0.7 * x + 0.11 * x3) * v
        + (0.6 + 0.13 * x + 0.17 * x2) * v2
        + (0.09 - 0.12 * x2) * v3
        + 0.025 * x3 * v4
    )
    field = -0.7 + 1.1 * x - 0.8 * x2 + 0.6 * x3
    # Integrate analytically over the *whole* velocity interval.  This oracle
    # never invokes the provider or numerically sums the source's velocity axis.
    # Integral(v^0)=4, integral(v^2)=16/3, integral(v^4)=64/5; odd terms vanish.
    moment = (
        4 * (1 + 0.3 * x + 0.2 * x2)
        + (16 / 3) * (0.6 + 0.13 * x + 0.17 * x2)
        + (64 / 5) * 0.025 * x3
    )
    return distribution, field, moment


def _restrict_x(values, coarse_nx):
    fine_nx = values.shape[-1]
    assert fine_nx == 2 * coarse_nx
    fine_volume, coarse_volume = Fraction(1, fine_nx), Fraction(1, coarse_nx)
    return values.reshape(*values.shape[:-1], coarse_nx, 2).sum(axis=-1) * float(
        fine_volume / coarse_volume
    )


def _restrict_phase(values, coarse_nx, coarse_nv):
    fine_nv, fine_nx = values.shape
    assert (fine_nx, fine_nv) == (2 * coarse_nx, 2 * coarse_nv)
    fine_volume = Fraction(1, fine_nx) * Fraction(4, fine_nv)
    coarse_volume = Fraction(1, coarse_nx) * Fraction(4, coarse_nv)
    return values.reshape(coarse_nv, 2, coarse_nx, 2).sum(axis=(1, 3)) * float(
        fine_volume / coarse_volume
    )


@pytest.mark.parametrize("nx,nv", REFINEMENTS)
def test_physical_property_oracle_uses_true_cell_averages(nx, nv):
    """Source-only premise check, including a point-sampling negative control."""
    distribution, field, moment = _polynomial_cell_averages(nx, nv)
    assert distribution.shape == (nv, nx)
    assert field.shape == moment.shape == (nx,)
    x = (np.arange(nx) + 0.5) / nx
    point_field = -0.7 + 1.1 * x - 0.8 * x**2 + 0.6 * x**3
    assert np.max(np.abs(field - point_field)) > 1e-8
    # Whole-domain integration provides a separate scalar anchor for M's oracle.
    expected_integral = (
        4 * (1 + 0.3 / 2 + 0.2 / 3) + (16 / 3) * (0.6 + 0.13 / 2 + 0.17 / 3) + (64 / 5) * 0.025 / 4
    )
    np.testing.assert_allclose(moment.sum() / nx, expected_integral, rtol=0, atol=ATOL)
    if (nx, nv) != REFINEMENTS[0]:
        coarse_f, coarse_g, coarse_m = _polynomial_cell_averages(nx // 2, nv // 2)
        np.testing.assert_allclose(
            _restrict_phase(distribution, nx // 2, nv // 2), coarse_f, rtol=0, atol=ATOL
        )
        np.testing.assert_allclose(_restrict_x(field, nx // 2), coarse_g, rtol=0, atol=ATOL)
        np.testing.assert_allclose(_restrict_x(moment, nx // 2), coarse_m, rtol=0, atol=ATOL)


@pytest.fixture(scope="module")
def native_property_maps(tmp_path_factory):
    results = {}
    for nx, nv in REFINEMENTS:
        directory = tmp_path_factory.mktemp(f"physical-properties-{nx}-{nv}")
        _, resolved = resolve_physical_case(directory, nx, nv)
        artifact = pops.compile(resolved)
        distribution, field, _ = _polynomial_cell_averages(nx, nv)
        initial = {
            "distribution": distribution[None],
            "field_observation": field[None, None],
            "density": np.full((1, 1, nx), -19.0),
            "observation": np.full((1, nv, nx), 23.0),
        }
        instance = pops.bind(
            artifact,
            initial_state=initial,
            resources={"execution_context": artifact_execution_context(artifact)},
        )
        native = instance._executor
        native._begin_step_transaction()
        try:
            generation = native._active_transfer_generation
            for route in native._transfer_routes:
                route.session.capture(generation, 1)
                receipt = route.session.apply(generation, 1)
                native._authenticate_mapping_receipt(
                    route, receipt, generation=generation, attempt=1
                )
            results[nx, nv] = (
                np.asarray(instance.get_state("density")).reshape(nx).copy(),
                np.asarray(instance.get_state("observation")).reshape(nv, nx).copy(),
            )
        finally:
            native._rollback_step_transaction()
    return results


@pytest.mark.compiler
@pytest.mark.native_loader
@pytest.mark.parametrize("coarse,fine", NESTED_PAIRS)
def test_native_velocity_moment_commutes_with_reference_restriction(
    native_property_maps, coarse, fine
):
    coarse_nx, coarse_nv = coarse
    coarse_moment = native_property_maps[coarse][0]
    fine_moment = native_property_maps[fine][0]
    # Coarse input was authored independently by analytic integration, not
    # generated by this restriction; establish R_xv f_f = f_c explicitly.
    fine_f, _, fine_expected = _polynomial_cell_averages(*fine)
    coarse_f, _, coarse_expected = _polynomial_cell_averages(*coarse)
    np.testing.assert_allclose(
        _restrict_phase(fine_f, coarse_nx, coarse_nv), coarse_f, rtol=0, atol=ATOL
    )
    np.testing.assert_allclose(fine_moment, fine_expected, rtol=0, atol=ATOL)
    np.testing.assert_allclose(coarse_moment, coarse_expected, rtol=0, atol=ATOL)
    np.testing.assert_allclose(
        _restrict_x(fine_moment, coarse_nx), coarse_moment, rtol=0, atol=ATOL
    )


@pytest.mark.compiler
@pytest.mark.native_loader
@pytest.mark.parametrize("coarse,fine", NESTED_PAIRS)
def test_native_spatial_pullback_commutes_with_reference_restriction(
    native_property_maps, coarse, fine
):
    coarse_nx, coarse_nv = coarse
    coarse_pullback = native_property_maps[coarse][1]
    fine_pullback = native_property_maps[fine][1]
    _, fine_field, _ = _polynomial_cell_averages(*fine)
    _, coarse_field, _ = _polynomial_cell_averages(*coarse)
    np.testing.assert_allclose(_restrict_x(fine_field, coarse_nx), coarse_field, rtol=0, atol=ATOL)
    np.testing.assert_allclose(
        fine_pullback, np.broadcast_to(fine_field, fine_pullback.shape), rtol=0, atol=ATOL
    )
    np.testing.assert_allclose(
        coarse_pullback, np.broadcast_to(coarse_field, coarse_pullback.shape), rtol=0, atol=ATOL
    )
    np.testing.assert_allclose(
        _restrict_phase(fine_pullback, coarse_nx, coarse_nv), coarse_pullback, rtol=0, atol=ATOL
    )


@pytest.mark.parametrize("lower,upper", [(0.125, 1.125), (0.0, 1.25)])
def test_physical_commutation_refuses_changed_x_range(tmp_path, lower, upper):
    """Equal shape (even equal width) cannot authenticate a different x support."""
    _, resolved = resolve_physical_case(tmp_path)
    plan = resolved.layout_plan
    for row in plan.mappings:
        requirement = row.requirement
        source = plan.normalized(requirement.source_layout).native_spatial_layout
        target = plan.normalized(requirement.target_layout).native_spatial_layout
        changed_target = SimpleNamespace(
            shape=target.shape,
            lower=(lower, target.lower[1]),
            upper=(upper, target.upper[1]),
            periodicity=target.periodicity,
            coordinate_system=target.coordinate_system,
        )
        with pytest.raises(ValueError, match="exactly aligned physical x geometry/topology"):
            validate_physical_geometry(requirement, source, changed_target)
