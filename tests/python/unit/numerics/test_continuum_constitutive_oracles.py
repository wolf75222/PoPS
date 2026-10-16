"""Independent NumPy constitutive references and term-removal counterexamples.

The authored functions are imported here only as the subject under test. The
reference module imports neither those functions nor the model/IR evaluator.
"""

import ast
from pathlib import Path

import numpy as np
import pytest

from examples.api_v02 import continuum_oracles as reference


def _bn_state():
    y, x = np.meshgrid(np.arange(4) / 4, np.arange(7) / 7, indexing="ij")
    alpha = 0.45 + 0.1 * np.sin(2 * np.pi * x)
    fractions = np.stack((alpha, 1 - alpha))
    densities = np.stack((1.2 + 0.05 * np.cos(2 * np.pi * y), 0.8 + 0.03 * np.cos(2 * np.pi * x)))
    velocities = np.stack(
        (0.25 + 0.04 * np.cos(2 * np.pi * x), -0.1 + 0.03 * np.sin(2 * np.pi * x))
    )
    pressures = np.stack((1.1 + 0.1 * np.cos(2 * np.pi * x), 0.9 + 0.06 * np.sin(2 * np.pi * x)))
    masses = fractions * densities
    energies = (
        fractions * pressures / np.array([0.4, 0.6])[:, None, None] + 0.5 * masses * velocities**2
    )
    return np.concatenate((alpha[None], masses, masses * velocities, energies))


def _mesoscopic_energy(a, impulse):
    # Independent invariant representation, also valid for complex-step probes.
    gram = np.swapaxes(a, -2, -1) @ a
    trace = np.trace(gram, axis1=-2, axis2=-1)
    return (
        0.25 * np.trace(gram @ gram, axis1=-2, axis2=-1)
        - trace**2 / 12
        + 0.5 * np.sum(impulse**2, axis=-1)
    )


def _gpr_state():
    random = np.random.default_rng(937)
    q = np.zeros((17, 4, 6))
    q[0] = random.uniform(0.8, 1.2, (4, 6))
    velocity = random.uniform(-0.15, 0.2, (4, 6, 3))
    impulse = random.uniform(-0.1, 0.1, (4, 6, 3))
    a = np.eye(3) + random.uniform(-0.06, 0.06, (4, 6, 3, 3))
    q[1:4] = np.moveaxis(q[0][..., None] * velocity, -1, 0)
    q[5:14] = np.moveaxis(a.reshape(4, 6, 9), -1, 0)
    q[14:17] = np.moveaxis(q[0][..., None] * impulse, -1, 0)
    q[4] = q[0] * (1.2 + 0.5 * np.sum(velocity**2, axis=-1) + _mesoscopic_energy(a, impulse))
    return q


def _dense(entries, shape):
    return np.stack([np.broadcast_to(entry, shape) for entry in entries])


def _authored_bn(q):
    from examples.api_v02.models_M22_M25_M28 import bn_equations

    flux, matrix, _ = bn_equations(tuple(q))
    fx = _dense(flux, q.shape[1:])
    bx = np.stack([_dense(row, q.shape[1:]) for row in matrix])
    return np.stack((fx, np.zeros_like(q))), np.stack((bx, np.zeros_like(bx))), np.zeros_like(q)


def _authored_gpr(q):
    from examples.api_v02.models_M22_M25_M28 import gpr_equations

    fluxes, matrices, source, _ = gpr_equations(tuple(q))
    return (
        np.stack([_dense(row, q.shape[1:]) for row in fluxes]),
        np.stack([np.stack([_dense(row, q.shape[1:]) for row in matrix]) for matrix in matrices]),
        _dense(source, q.shape[1:]),
    )


def test_reference_has_no_model_dependency_and_runs_when_authored_equations_are_disabled(
    monkeypatch,
):
    from examples.api_v02 import models_M22_M25_M28 as authored

    tree = ast.parse(Path(reference.__file__).read_text())
    imports = [node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
    assert imports == ["__future__"]
    assert [
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    ] == ["numpy"]

    def disabled(*args, **kwargs):
        raise AssertionError("authored constitutive equation used by its oracle")

    monkeypatch.setattr(authored, "bn_equations", disabled)
    monkeypatch.setattr(authored, "gpr_equations", disabled)
    assert np.isfinite(reference.advance_m24(_bn_state(), steps=2)).all()
    assert np.isfinite(reference.advance_m25(_gpr_state(), steps=2)).all()


@pytest.mark.parametrize(
    "state, oracle, subject",
    (
        (_bn_state, reference._bn_equations, _authored_bn),
        (_gpr_state, reference._gpr_equations, _authored_gpr),
    ),
)
def test_all_constitutive_flux_matrix_and_source_entries_match_independent_reference(
    state, oracle, subject
):
    q = state()
    for expected, actual in zip(oracle(q), subject(q), strict=True):
        np.testing.assert_allclose(actual, expected, atol=4e-14, rtol=4e-14)


def test_gpr_tensor_gradient_stress_and_relaxation_follow_the_energy_potential():
    q = _gpr_state()
    c = reference._gpr_constitutive(q)
    a, impulse = c["distortion"], c["impulse"]
    gradient = np.empty_like(a)
    for i in range(3):
        for j in range(3):
            perturbed = a.astype(complex)
            perturbed[..., i, j] += 1.0e-30j
            gradient[..., i, j] = _mesoscopic_energy(perturbed, impulse).imag / 1.0e-30
    np.testing.assert_allclose(c["distortion_gradient"], gradient, atol=5e-16, rtol=3e-13)
    stress = -q[0][..., None, None] * (np.swapaxes(a, -2, -1) @ gradient)
    np.testing.assert_allclose(c["stress"], stress, atol=6e-16, rtol=3e-13)
    _, _, source = reference._gpr_equations(q)
    sa = np.moveaxis(source[5:14], 0, -1).reshape(*q.shape[1:], 3, 3)
    sj = np.moveaxis(source[14:17] / q[0], 0, -1)
    meso_rate = np.sum(gradient * sa, axis=(-2, -1)) + np.sum(impulse * sj, axis=-1)
    assert np.max(meso_rate) < -1e-2
    assert np.array_equal(source[:5], np.zeros_like(source[:5]))
    determinant_rate = c["determinant"] * np.trace(np.linalg.solve(a, sa), axis1=-2, axis2=-1)
    np.testing.assert_allclose(determinant_rate, 0.0, atol=2e-14)


def test_gpr_noncurlfree_distortion_path_uses_both_transport_terms():
    q = _gpr_state()
    _, matrices, _ = reference._gpr_equations(q)
    velocity = np.moveaxis(q[1:4] / q[0], 0, -1)
    tangent = np.random.default_rng(64).uniform(-0.1, 0.1, (*q.shape[1:], 3, 3))
    jump = np.zeros_like(q)
    jump[5:14] = np.moveaxis(tangent.reshape(*q.shape[1:], 9), -1, 0)
    for axis in (0, 1):
        expected = velocity[..., axis, None, None] * tangent
        expected[..., :, axis] -= np.einsum("...ij,...j->...i", tangent, velocity)
        actual = reference._matrix_action(matrices[axis], jump)[5:14]
        np.testing.assert_allclose(
            actual, np.moveaxis(expected.reshape(*q.shape[1:], 9), -1, 0), atol=5e-18, rtol=3e-14
        )


@pytest.mark.parametrize("missing", ("fraction_advection", "interface_force", "interface_work"))
def test_bn_missing_nonconservative_terms_are_rejected_by_independent_oracle(missing):
    q = _bn_state()
    expected = reference.rhs_m24(q)

    def mutant(values):
        fluxes, matrices, source = _authored_bn(values)
        rows = {
            "fraction_advection": slice(0, 1),
            "interface_force": slice(3, 5),
            "interface_work": slice(5, 7),
        }[missing]
        matrices[:, rows] = 0.0
        return fluxes, matrices, source

    actual = reference._path_rhs(q, mutant, reference._bn_speed, path_points=2)
    assert np.max(abs(actual - expected)) > 1.0e-3


@pytest.mark.parametrize(
    "missing",
    (
        "distortion_relaxation",
        "thermal_relaxation",
        "distortion_curl",
        "elastic_stress",
        "heat_energy_flux",
    ),
)
def test_gpr_missing_constitutive_terms_are_rejected_by_independent_oracle(missing):
    q = _gpr_state()
    expected = reference.rhs_m25(q)

    def mutant(values):
        fluxes, matrices, source = _authored_gpr(values)
        if missing == "distortion_relaxation":
            source[5:14] = 0.0
        elif missing == "thermal_relaxation":
            source[14:17] = 0.0
        elif missing == "distortion_curl":
            matrices[:] = 0.0
        else:
            c = reference._gpr_constitutive(values)
            for axis in (0, 1):
                if missing == "elastic_stress":
                    traction = c["stress"][..., :, axis]
                    fluxes[axis, 1:4] += np.moveaxis(traction, -1, 0)
                    fluxes[axis, 4] += np.sum(c["velocity"] * traction, axis=-1)
                else:
                    fluxes[axis, 4] -= c["temperature"] * c["impulse"][..., axis]
        return fluxes, matrices, source

    actual = reference._path_rhs(q, mutant, lambda left, right, axis: 64.0, path_points=2)
    assert np.max(abs(actual - expected)) > 1.0e-3


def test_bn_exchange_has_opposite_impulses_and_energy_without_changing_phase_masses():
    q = _bn_state()
    exchange = reference.rhs_m24(
        q, pressure_relaxation=0.7, velocity_relaxation=0.9
    ) - reference.rhs_m24(q)
    np.testing.assert_array_equal(exchange[1:3], np.zeros_like(exchange[1:3]))
    np.testing.assert_allclose(exchange[3] + exchange[4], 0.0, atol=3e-16)
    np.testing.assert_allclose(exchange[5] + exchange[6], 0.0, atol=4e-16)
    assert np.max(abs(exchange[0])) > 0.05
    assert np.max(abs(exchange[3])) > 0.1
    assert np.max(abs(exchange[5])) > 0.01
