"""Fixed B.1 axial and oblique criteria, independent of a native build."""
from __future__ import annotations

import importlib.util
from math import comb, factorial, isfinite, sqrt
from pathlib import Path

import numpy as np
import pytest

import pops
from pops.codegen.module_codegen import emit_cpp
from pops.codegen.program_codegen import emit_cpp_program
from pops.codegen.program_models import ProgramModelGraph
from pops.moments import hyqmom_b1_axial_flux


ROOT = Path(__file__).resolve().parents[4]
SCIENTIFIC = ROOT / "examples/migration/scientific"


def _example(name):
    path = SCIENTIFIC / name
    spec = importlib.util.spec_from_file_location(name.removesuffix(".py"), path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _normal_moment(order, mean, variance):
    return sum(comb(order, 2 * k) * factorial(2 * k) / (2**k * factorial(k))
               * variance**k * mean**(order - 2 * k)
               for k in range(order // 2 + 1))


def _mixture(variance):
    return tuple(.4 * _normal_moment(order, -.7, variance)
                 + .6 * _normal_moment(order, .8, 2 * variance)
                 for order in range(5))


def _b1_fifth_oracle(raw):
    rho, m1, m2, m3, m4 = raw
    mean = m1 / rho
    c2 = m2 / rho - mean**2
    c3 = m3 / rho - 3 * mean * m2 / rho + 2 * mean**3
    c4 = m4 / rho - 4 * mean * m3 / rho + 6 * mean**2 * m2 / rho - 3 * mean**4
    s30 = c3 / c2**1.5
    s40 = c4 / c2**2
    c5 = c2**2.5 * (.5 * s30 * (5 * s40 - 3 * s30**2 - 1))
    return rho * (mean**5 + 10 * mean**3 * c2 + 10 * mean**2 * c3
                  + 5 * mean * c4 + c5)


def _number(expression):
    return float(expression.eval({}) if hasattr(expression, "eval") else expression)


def _emit(case, layout):
    resolved = pops.resolve(pops.validate(case), layout=layout)
    graph = ProgramModelGraph.from_resolved_blocks(resolved.blocks)
    return (emit_cpp_program(resolved.time, model=graph),
            emit_cpp(graph.model_for_block("moments")._m))


def test_m15_axial_b1_matches_nonlinear_oracle_and_approaches_moment_boundary():
    # Criteria fixed before the run: 1e-11 flux agreement, positive Hankel
    # determinant at every chosen mixture, and no Gaussian fifth substitution.
    for variance in (.08, .01, .0001):
        raw = _mixture(variance)
        flux = tuple(map(_number, hyqmom_b1_axial_flux(raw)))
        np.testing.assert_allclose(flux[:4], raw[1:], rtol=0, atol=0)
        assert abs(flux[4] - _b1_fifth_oracle(raw)) < 1e-11
        assert isfinite(flux[4])
        hankel = np.array([[raw[i + j] for j in range(3)] for i in range(3)])
        assert np.linalg.eigvalsh(hankel).min() > 0
    raw = _mixture(.08)
    mean = raw[1] / raw[0]
    variance = raw[2] / raw[0] - mean**2
    gaussian_fifth = raw[0] * (mean**5 + 10 * mean**3 * variance + 15 * mean * variance**2)
    assert abs(_number(hyqmom_b1_axial_flux(raw)[4]) - gaussian_fifth) > .01
    with pytest.raises(ValueError, match="exactly M0..M4"):
        hyqmom_b1_axial_flux(raw[:4])


def test_m15_source_validate_resolve_and_emit_uses_generic_five_moment_model():
    example = _example("api040_m15_hyqmom_axial_b1.py")
    case, layout, state = example.build_case(8)
    assert len(state.components) == 5
    program, physical_flux = _emit(case, layout)
    assert len(program) > 1000
    assert "sqrt" in physical_flux and physical_flux.count("F[") == 5
    assert "hyqmom_b1_axial_order4" in physical_flux


def test_m16_exact_polynomial_and_public_flux_show_four_nonreal_oblique_roots():
    # Fixed criteria: polynomial error <1e-8, axial imaginary parts <1e-6,
    # oblique unit-normal imaginary magnitude in (.06,.07).
    example = _example("api040_m16_hyqmom_b1_oblique.py")
    witness = example.oblique_witness()
    assert witness["discriminant"] == -4627764
    np.testing.assert_allclose(witness["polynomial"], witness["expected_polynomial"],
                               rtol=0, atol=1e-8)
    assert witness["max_imag_x"] < 1e-6
    assert witness["max_imag_y"] < 1e-6
    assert .06 < witness["max_imag_unit_oblique"] < .07
    raw = example.CORRELATED_GAUSSIAN
    basis = ((0, 0), (1, 0), (0, 1), (2, 0), (1, 1), (0, 2))
    moments = dict(zip(((p, q) for q in range(5) for p in range(5 - q)), raw))
    gram = np.array([[moments[p + i, q + j] for i, j in basis] for p, q in basis])
    assert np.linalg.eigvalsh(gram).min() > 0
    eig = np.linalg.eigvals(witness["oblique"])
    assert np.count_nonzero(abs(eig.imag) > .05) == 4


def test_m16_full_b1_model_validate_resolve_and_emit_without_native_execution():
    example = _example("api040_m16_hyqmom_b1_oblique.py")
    case, layout, state = example.build_case(8)
    assert len(state.components) == 15
    program, physical_flux = _emit(case, layout)
    assert len(program) > 1000
    assert "hyqmom15_B1" in physical_flux
    assert physical_flux.count("F[") == 30
