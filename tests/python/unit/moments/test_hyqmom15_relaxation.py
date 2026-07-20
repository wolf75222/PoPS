"""MATLAB parity for the explicit HyQMOM15 relaxation transform."""
from __future__ import annotations

import numpy as np
import pytest

import pops
from pops.codegen.program_codegen import emit_cpp_program
from pops.frames import Cartesian2D
from pops.moments import CartesianVelocityMoments, HyQMOM15Closure, HyQMOM15Relaxation
from pops.moments._relaxation_reference import _apply_hyqmom15_relaxation_array
from pops.time import Program


def _apply(values):
    return _apply_hyqmom15_relaxation_array(
        values, cutoff=1.0e-12, mach=20.0, small=1.0e-6, spectral_tolerance=0.0)


def test_relaxation15_matches_non_idempotent_matlab_golden_once() -> None:
    source = np.array([
        1.0, 0.0, 1.0, 6.0, 46.0,
        0.0, 0.99, 0.1, 0.0,
        1.0, 0.0, 3.0,
        1.0, 0.0, 4.0,
    ])
    expected = np.array([
        1.0, 0.0, 1.0, 1.5, 8.52985,
        0.0, 0.99, 1.485, 0.0,
        1.0, 1.5, 8.52985,
        0.325, 0.0, 8.52985,
    ])
    once = _apply(source)
    twice = _apply(once)

    assert once == pytest.approx(expected, rel=2.0e-12, abs=2.0e-12)
    assert np.max(np.abs(twice - once)) == pytest.approx(8.52985)


def test_relaxation15_preserves_the_matlab_conserved_moments() -> None:
    source = np.array([
        1.0, 0.0, 1.0, 6.0, 46.0,
        0.0, 0.99, 0.1, 0.0,
        1.0, 0.0, 3.0,
        1.0, 0.0, 4.0,
    ])
    result = _apply(source)
    assert result[[0, 1, 2, 5, 9]] == pytest.approx(source[[0, 1, 2, 5, 9]])


def test_relaxation15_emits_a_bounded_native_program_kernel() -> None:
    model = CartesianVelocityMoments(
        4,
        closure=HyQMOM15Closure(),
        robust=False,
        exact_speeds=True,
    ).build("relaxation15_emit", frame=Cartesian2D())
    state = model.states["U"]
    transform = HyQMOM15Relaxation().declare(model, state)

    case = pops.Case("relaxation15_emit_case")
    block = case.block("fluid", model=model)
    program = Program("relaxation15_emit_program")
    moments = program.state(block[state])
    candidate = program.value("candidate", moments.n, at=moments.next.point)
    transformed = program.transform(
        candidate, transform=transform, name="transformed_candidate")
    program.commit(moments.next, transformed)

    source = emit_cpp_program(program, model=model)
    assert len(source) < 250_000
    assert "static POPS_HD pops::Real pops_eig_real_status_3x3" in source
    assert "static POPS_HD pops::Real pops_eig_lmin_3x3" in source
    assert "if (!bounds.valid()) return std::numeric_limits<pops::Real>::quiet_NaN();" in source
    assert "Kokkos::fmin" in source
    assert "Kokkos::fmax" in source
    assert "transform_valid_" in source
    assert "ctx.pointwise_active_mask(0," in source
    assert "ctx.pointwise_status_max(0," in source
    assert "ctx.apply_projection" not in source
    assert source.count("if (!Kokkos::isfinite(cse") == source.count(
        "const pops::Real cse")
