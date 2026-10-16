"""Receive complete native method states and compare independent dense oracles."""
import numpy as np
import pytest

from examples.api_v02.methods_coupled_nonsymmetric import (
    build_t04_nonsymmetric, build_t06_selected_global, build_t02_imex_second_order)
from examples.api_v02.methods_patankar_network import build_t11_network
from tests.python.support import coupled_methods_oracle as reference
from tests.python.support.api_v02_native import execute_case
from tests.python.support.api_v02_receipts import _decode, _image
from tests.python.support.collective_checks import collective_call, collective_check
from tests.python.support.public_native_profile import native_profile


pytestmark = [pytest.mark.compiler, pytest.mark.kokkos, pytest.mark.native_loader]


def _execute(builder, name):
    _, world = native_profile(dimension=2, scope=name)
    spec = collective_call(world, builder)
    runtime, artifact, _ = execute_case(lambda: spec, world, name)
    image = collective_call(world, lambda: _image(runtime))
    with collective_check(world):
        packet = _decode(spec, image)["root"]
    return packet, artifact, world


def test_native_global_nonsymmetric_schur(isolated_native_cache, native_cxx, kokkos_root, record_property):
    del isolated_native_cache, native_cxx, kokkos_root
    packet, artifact, world = _execute(build_t04_nonsymmetric, "T04 global nonsymmetric")
    with collective_check(world):
        state = packet["values"]
        schur = np.array([state[k]["values"] for k in ("x", "y")])
        monolithic = np.array([state[k]["values"] for k in ("monolithic_x", "monolithic_y")])
        expected = reference.block_solution(5, 2)
        error = float(np.max(abs(schur-expected)))
        np.testing.assert_allclose(schur, monolithic, rtol=0, atol=2.e-10)
        assert error < 2.e-10
    record_property("artifact_identity", artifact.artifact_identity.hexdigest)
    record_property("original_dense_reference_error", error)


@pytest.mark.parametrize("mode,branch", (("frozen", 1), ("unknown", 1), ("unknown", -1)))
def test_native_global_selected_roots(mode, branch, isolated_native_cache, native_cxx,
        kokkos_root, record_property):
    del isolated_native_cache, native_cxx, kokkos_root
    packet, artifact, world = _execute(lambda: build_t06_selected_global(mode=mode, branch=branch),
                                       "T06 selected global relation")
    with collective_check(world):
        actual = np.asarray(packet["values"]["root"]["values"])
        expected = reference.selected_root(5, mode, branch)
        assert np.all(branch*actual > 0)
        error = float(np.max(abs(actual-expected)))
        assert error < 2.e-10
    record_property("artifact_identity", artifact.artifact_identity.hexdigest)
    record_property("original_nonlinear_reference_error", error)


def test_native_imex_second_order(isolated_native_cache, native_cxx, kokkos_root, record_property):
    del isolated_native_cache, native_cxx, kokkos_root
    errors = []
    for steps in (4, 8, 16):
        dt = .1/steps
        packet, artifact, world = _execute(lambda: build_t02_imex_second_order(dt=dt, steps=steps),
                                           "T02 IMEX ARS222")
        with collective_check(world):
            actual = np.asarray(packet["values"]["values"])
            expected = reference.imex_reference(7, dt, steps)
            error = float(np.max(abs(actual-expected)))
            errors.append(float(np.max(abs(actual-reference.imex_exact(7, .1)))))
            assert error < 2.e-10
        record_property(f"artifact_identity_{steps}", artifact.artifact_identity.hexdigest)
        record_property(f"discrete_reference_error_{steps}", error)
    with collective_check(world):
        assert all(3.5 < a/b < 4.5 for a, b in zip(errors, errors[1:]))
    record_property("semidiscrete_exact_errors", str(errors))


@pytest.mark.parametrize("circulation,empty", ((0., False), (1.5, True)))
def test_native_mprk22_network(circulation, empty, isolated_native_cache, native_cxx,
        kokkos_root, record_property):
    del isolated_native_cache, native_cxx, kokkos_root
    packet, artifact, world = _execute(lambda: build_t11_network(circulation=circulation, empty=empty),
                                       "T11 general network MPRK22")
    with collective_check(world):
        actuals = [np.asarray(packet[key]["values"]) for key in ("coarse", "medium", "fine")]
        expected = [reference.patankar_reference(5, .1/k, 6*k, circulation, empty) for k in (1, 2, 4)]
        error = max(float(np.max(abs(a-e))) for a, e in zip(actuals, expected, strict=True))
        assert error < 2.e-10
        assert all(np.min(a) >= 0 for a in actuals)
        total = 7.-(1 if empty else 0)
        assert all(abs(a.sum()-total) < 2.e-11 for a in actuals)
        errors = [float(np.max(abs(a-reference.network_exact(5, .6, empty)))) for a in actuals]
        # Zero starting species have separate reception; no inherited order claim.
        if not empty:
            assert all(3.5 < a/b < 4.5 for a, b in zip(errors, errors[1:]))
    record_property("artifact_identity", artifact.artifact_identity.hexdigest)
    record_property("original_dense_reference_error", error)
    record_property("exact_solution_errors", str(errors))
