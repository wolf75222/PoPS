"""Independent finite Hoffart source oracle from the v0.4.0 M09 witness.

This 12-unknown algebraic witness is not the conducting-disk PDE, its FEM
discretisation, or a PoPS native execution.  It executes no legacy binary.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


SEED = 20260928
N_VELOCITY = 8
N_POTENTIAL = 4
S = 0.03
ALPHA = 0.5
MAX_ERROR = 1.0e-11  # fixed by M09 before either solve


@dataclass(frozen=True)
class Witness:
    a: np.ndarray
    b: np.ndarray
    c: np.ndarray
    k: np.ndarray
    f: np.ndarray
    g: np.ndarray
    rho: np.ndarray
    velocity_old: np.ndarray
    potential_old: np.ndarray


def witness(seed: int = SEED) -> Witness:
    """Use the reference's exact RNG order and full rotation/field coupling."""
    rng = np.random.default_rng(seed)
    gradient = rng.normal(size=(N_VELOCITY, N_POTENTIAL))
    divergence = -gradient.T
    poisson = gradient.T @ gradient
    density = np.array((1.0, 1.1, 0.9, 1.2))
    rho = np.diag(np.tile(density, 2))
    eye = np.eye(N_POTENTIAL)
    rotation = np.block([[np.zeros_like(eye), eye], [-eye, np.zeros_like(eye)]])
    velocity_old = rng.normal(size=N_VELOCITY)
    potential_old = rng.normal(size=N_POTENTIAL)
    return Witness(
        a=np.eye(N_VELOCITY) - S * rotation,
        b=S * gradient,
        c=S * ALPHA * divergence @ rho,
        k=poisson,
        f=velocity_old.copy(),
        g=poisson @ potential_old,
        rho=rho,
        velocity_old=velocity_old,
        potential_old=potential_old,
    )


def monolithic(problem: Witness) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.block([[problem.a, problem.b], [problem.c, problem.k]])
    solution = np.linalg.solve(matrix, np.concatenate((problem.f, problem.g)))
    return solution[:N_VELOCITY], solution[N_VELOCITY:]


def schur(problem: Witness) -> tuple[np.ndarray, np.ndarray]:
    """Independent elimination; no call to monolithic or PoPS kernels."""
    ainv_f = np.linalg.solve(problem.a, problem.f)
    ainv_b = np.linalg.solve(problem.a, problem.b)
    phi = np.linalg.solve(problem.k - problem.c @ ainv_b,
                          problem.g - problem.c @ ainv_f)
    velocity = np.linalg.solve(problem.a, problem.f - problem.b @ phi)
    return velocity, phi


def original_residual(problem: Witness, velocity: np.ndarray,
                      potential: np.ndarray) -> float:
    first = problem.a @ velocity + problem.b @ potential - problem.f
    second = problem.c @ velocity + problem.k @ potential - problem.g
    return float(max(np.max(np.abs(first)), np.max(np.abs(second))))


def endpoint(problem: Witness, midpoint: tuple[np.ndarray, np.ndarray]
             ) -> tuple[np.ndarray, np.ndarray]:
    velocity, potential = midpoint
    return 2 * velocity - problem.velocity_old, 2 * potential - problem.potential_old


def energy(problem: Witness, velocity: np.ndarray, potential: np.ndarray) -> float:
    return float(0.5 * velocity @ problem.rho @ velocity
                 + 0.5 / ALPHA * potential @ problem.k @ potential)


def measurements(problem: Witness) -> dict[str, float | int | str]:
    dense = monolithic(problem)
    reduced = schur(problem)
    new_velocity, new_potential = endpoint(problem, reduced)
    old_complex = problem.velocity_old[:4] + 1j * problem.velocity_old[4:]
    new_complex = new_velocity[:4] + 1j * new_velocity[4:]
    return {
        "scope": "finite 12-unknown source witness; no native PoPS or disk PDE",
        "seed": SEED,
        "original_residual_monolithic": original_residual(problem, *dense),
        "original_residual_schur": original_residual(problem, *reduced),
        "mono_schur_error": float(max(np.max(np.abs(dense[0] - reduced[0])),
                                       np.max(np.abs(dense[1] - reduced[1])))),
        "cn_energy_defect": abs(energy(problem, new_velocity, new_potential)
                                - energy(problem, problem.velocity_old,
                                         problem.potential_old)),
        "velocity_norm_before": float(np.linalg.norm(old_complex)),
        "velocity_norm_after": float(np.linalg.norm(new_complex)),
        "velocity_phase_before": float(np.angle(np.sum(old_complex))),
        "velocity_phase_after": float(np.angle(np.sum(new_complex))),
    }


def main() -> None:
    import json

    report = measurements(witness())
    for key in ("original_residual_monolithic", "original_residual_schur",
                "mono_schur_error", "cn_energy_defect"):
        if report[key] > MAX_ERROR:
            raise AssertionError(f"M09 finite source {key}: {report[key]} > {MAX_ERROR}")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
