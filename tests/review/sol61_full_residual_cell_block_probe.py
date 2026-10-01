"""Offline source/math probe; no PoPS import, native execution or HPC realization."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re

import numpy as np


def pivot_inverse(block):
    """Partial-pivot Gauss-Jordan, finite signed square blocks; no SPD assumption."""
    a = np.array(block, dtype=float, copy=True)
    if a.ndim != 2 or a.shape[0] != a.shape[1] or not np.isfinite(a).all():
        raise ValueError("finite square block required")
    m = len(a)
    inverse = np.eye(m)
    for column in range(m):
        pivot = column + int(np.argmax(np.abs(a[column:, column])))
        if a[pivot, column] == 0:
            raise ValueError("singular cell block")
        a[[column, pivot]] = a[[pivot, column]]
        inverse[[column, pivot]] = inverse[[pivot, column]]
        value = a[column, column]
        a[column] /= value
        inverse[column] /= value
        for row in range(m):
            if row != column:
                factor = a[row, column]
                a[row] -= factor * a[column]
                inverse[row] -= factor * inverse[column]
        if not np.isfinite(a).all() or not np.isfinite(inverse).all():
            raise ValueError("cell block elimination overflow")
    return inverse


def scalar_composite_image(values, cells):
    """Independent flux construction for the actual 1D partial-refinement witness."""
    active = list(range(cells // 4)) + list(range(3 * cells // 4, cells))
    parent = np.zeros(cells)
    parent[active] = values[: len(active)]
    child = values[len(active) :]
    for c in range(cells // 4, 3 * cells // 4):
        parent[c] = np.mean(child[2 * c - cells // 2 : 2 * c - cells // 2 + 2])
    # Ratio-two quadratic parent interpolation at the two fine ghost centres.
    left = (-3 * parent[cells // 4 - 2] + 30 * parent[cells // 4 - 1] + 5 * parent[cells // 4]) / 32
    right = (
        5 * parent[3 * cells // 4 - 1]
        + 30 * parent[3 * cells // 4]
        - 3 * parent[3 * cells // 4 + 1]
    ) / 32
    fine = np.r_[left, child, right]
    coarse_flux = np.r_[0.0, np.diff(parent) * cells, 0.0]
    fine_flux = np.diff(fine) * (2 * cells)
    # Replace the two coarse interface fluxes by the actual fine flux.
    coarse_flux[cells // 4] = fine_flux[0]
    coarse_flux[3 * cells // 4] = fine_flux[-1]
    coarse_image = -np.diff(coarse_flux) * cells
    fine_image = -np.diff(fine_flux) * (2 * cells)
    return np.r_[coarse_image[active], fine_image]


def witness(source, cells, permutation):
    fixture = source / "tests/cpp/unit/elliptic/amr_original_field_residual.inc"
    text = fixture.read_text()
    if "Real(1e-5), lane, realization" not in text:
        raise ValueError("witness finite-difference step no longer matches this probe")

    def array(name):
        return np.array(
            [float(x) for x in re.search(r"\b" + name + r"\{([^}]+)\}", text)[1].split(",")]
        )

    controls = {}
    for name in (
        "tolerance",
        "max_iterations",
        "linear_tolerance",
        "linear_max_iterations",
        "restart",
        "armijo",
        "minimum_step",
    ):
        value = re.search(r"\." + name + r"\s*=\s*(?:Real\()?([\deE.+-]+)", text)[1]
        controls[name] = (
            int(value)
            if name in ("max_iterations", "linear_max_iterations", "restart")
            else float(value)
        )
    cubic = float(re.search(r"Real\(([^)]+)\)\s*\*\s*u\s*\*\s*u\s*\*\s*u", text)[1])
    order = np.array(permutation)
    D = array("diffusion").reshape(3, 3)[np.ix_(order, order)]
    R = array("reaction").reshape(3, 3)[np.ix_(order, order)]
    active = list(range(cells // 4)) + list(range(3 * cells // 4, cells))
    fine = list(range(cells // 2, 3 * cells // 2))
    size = len(active) + len(fine)
    S = np.column_stack([scalar_composite_image(e, cells) for e in np.eye(size)])
    measure = np.r_[np.full(len(active), 1 / cells), np.full(len(fine), 1 / (2 * cells))]
    assert np.max(np.abs(S @ np.ones(size))) == 0
    assert np.max(np.abs(measure @ S)) == 0
    centres = np.r_[(np.array(active) + 0.5) / cells, (np.array(fine) + 0.5) / (2 * cells)]
    target = array("means")[order] + np.cos(2 * np.pi * centres[:, None]) * array("waves")[order]
    A = np.kron(S, D) + np.kron(np.eye(size), R)
    target = target.ravel()
    forcing = A @ target + cubic * target**3
    return A, np.kron(S, D), forcing, target, np.repeat(measure, 3), cubic, controls, len(D)


def gmres(jvp, rhs, apply_right, weights, controls):
    def dot(a, b):
        return float(weights @ (a * b))

    def norm(a):
        return float(np.sqrt(dot(a, a)))

    stop = controls["linear_tolerance"] * norm(rhs)
    correction = np.zeros_like(rhs)
    defect = rhs.copy()
    columns, cycles = 0, []
    while columns < controls["linear_max_iterations"]:
        beta = norm(defect)
        if beta <= stop:
            break
        count = min(controls["restart"], controls["linear_max_iterations"] - columns)
        basis = [defect / beta]
        H = np.zeros((count + 1, count))
        cosines, sines, rotated = np.zeros(count), np.zeros(count), np.zeros(count + 1)
        rotated[0] = beta
        for k in range(count):
            work = jvp(apply_right(basis[k]))
            for _ in range(2):
                for row in range(k + 1):
                    projection = dot(work, basis[row])
                    H[row, k] += projection
                    work -= projection * basis[row]
            H[k + 1, k] = norm(work)
            basis.append(work / H[k + 1, k] if H[k + 1, k] else np.zeros_like(work))
            for row in range(k):
                upper, lower = H[row, k], H[row + 1, k]
                H[row, k] = cosines[row] * upper + sines[row] * lower
                H[row + 1, k] = -sines[row] * upper + cosines[row] * lower
            pivot = np.hypot(H[k, k], H[k + 1, k])
            if not np.isfinite(pivot) or pivot == 0:
                raise ValueError("Arnoldi breakdown")
            cosines[k], sines[k] = H[k, k] / pivot, H[k + 1, k] / pivot
            H[k, k], H[k + 1, k] = pivot, 0
            rotated[k + 1] = -sines[k] * rotated[k]
            rotated[k] *= cosines[k]
            columns += 1
            if abs(rotated[k + 1]) <= stop:
                break
        used = k + 1
        coefficients = np.zeros(used)
        for row in reversed(range(used)):
            coefficients[row] = (
                rotated[row] - H[row, row + 1 : used] @ coefficients[row + 1 :]
            ) / H[row, row]
        for index, coefficient in enumerate(coefficients):
            correction += coefficient * apply_right(basis[index])
        defect = rhs - jvp(correction)  # Actual full correction, never only projected RHS.
        actual = norm(defect)
        cycles.append(dict(columns=columns, projected=float(abs(rotated[used])), actual=actual))
        if actual <= stop:
            break
    return correction, dict(
        converged=norm(defect) <= stop,
        stop=stop,
        actual=norm(defect),
        columns=columns,
        cycles=cycles,
    )


def run(source, cells, permutation, mode, preconditioner):
    A, spatial, forcing, target, weights, cubic, controls, width = witness(
        source, cells, permutation
    )

    def norm(x):
        return float(np.sqrt(weights @ (x * x)))

    def F(q):
        return A @ q + cubic * q**3 - forcing

    q = np.zeros_like(target)
    reference = norm(F(q))
    stop = controls["tolerance"] * max(1.0, reference)
    steps = []
    for iteration in range(controls["max_iterations"]):
        defect = F(q)
        if norm(defect) <= stop:
            break
        J = A + np.diag(3 * cubic * q * q)
        if preconditioner == "full_cell_block":
            # Full residual block, including every local reaction derivative.
            if mode == "analytic":
                blocks = [J[i : i + width, i : i + width] for i in range(0, len(q), width)]
                preparation_evaluations = 0
            else:
                # Actual original-F evaluations: two per global active DOF.
                blocks = [np.zeros((width, width)) for _ in range(len(q) // width)]
                for column in range(len(q)):
                    direction = np.zeros_like(q)
                    direction[column] = 1
                    h = 1e-5 * (1 + norm(q)) / norm(direction)
                    derivative = 0.5 / h * F(q + h * direction) - 0.5 / h * F(q - h * direction)
                    cell = column // width
                    blocks[cell][:, column % width] = derivative[cell * width : (cell + 1) * width]
                preparation_evaluations = 2 * len(q)
            inverse = [pivot_inverse(block) for block in blocks]

            def apply_right(v, frozen_inverse=tuple(inverse)):
                return np.concatenate(
                    [
                        block @ v[i * width : (i + 1) * width]
                        for i, block in enumerate(frozen_inverse)
                    ]
                )
        else:
            diagonal = np.diag(spatial)
            preparation_evaluations = 0

            def apply_right(v, frozen_diagonal=diagonal):
                return v / frozen_diagonal

        if mode == "analytic":

            def jvp(v, frozen_J=J):
                return frozen_J @ v
        else:

            def jvp(v, frozen_q=q):
                magnitude = norm(v)
                h = 1e-5 * (1 + norm(frozen_q)) / (magnitude if magnitude > 0 else 1)
                # Same normalized central FD as the native @1 witness.
                return 0.5 / h * F(frozen_q + h * v) - 0.5 / h * F(frozen_q - h * v)

        delta, linear = gmres(jvp, -defect, apply_right, weights, controls)
        steps.append(
            dict(
                newton=iteration,
                original_norm=norm(defect),
                linear=linear,
                preparation_evaluations=preparation_evaluations,
            )
        )
        if not linear["converged"]:
            break
        alpha = 1.0
        while alpha >= controls["minimum_step"]:
            trial = q + alpha * delta
            if norm(F(trial)) <= (1 - controls["armijo"] * alpha) * norm(defect):
                q = trial
                break
            alpha *= 0.5
        else:
            break
    final = norm(F(q))
    return dict(
        cells=cells,
        permutation=permutation,
        jvp=mode,
        preconditioner=preconditioner,
        controls=controls,
        solved=final <= stop,
        final_original_norm=final,
        nonlinear_stop=stop,
        target_error=float(np.max(np.abs(q - target))),
        steps=steps,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    args = parser.parse_args()
    paths = (
        "tests/cpp/unit/elliptic/amr_original_field_residual.inc",
        "tests/cpp/unit/elliptic/test_composite_general_field.cpp",
        "include/pops/numerics/elliptic/mg/composite_fac_nlevel.hpp",
        "include/pops/numerics/elliptic/amr/composite_fac_poisson.hpp",
        "include/pops/numerics/elliptic/interface/amr_field_newton_krylov.hpp",
        "include/pops/runtime/program/prepared_amr_field_residual.hpp",
    )
    pins = {p: hashlib.sha256((args.source / p).read_bytes()).hexdigest() for p in paths}
    # Pivot / signed-nonsymmetric support and explicit singular refusal.
    for width in (1, 3, 5):
        block = np.diag(-np.arange(1, width + 1, dtype=float))
        if width > 1:
            block[[0, -1]] = block[[-1, 0]]
            block[0, 0] = 0.25
        assert np.max(np.abs(block @ pivot_inverse(block) - np.eye(width))) < 1e-14
    try:
        pivot_inverse([[1, 2], [2, 4]])
    except ValueError:
        pass
    else:
        raise AssertionError("singular block accepted")
    profiles = [
        run(args.source, n, perm, mode, "full_cell_block")
        for n in (16, 32)
        for perm in ((0, 1, 2), (2, 0, 1))
        for mode in ("analytic", "central_fd")
    ]
    profiles.append(run(args.source, 32, (0, 1, 2), "analytic", "spatial_diagonal"))
    expected_negative = (
        all(p["solved"] == (p["cells"] == 16) for p in profiles[:-1]) and not profiles[-1]["solved"]
    )
    print(
        json.dumps(
            dict(
                scope="offline source/math; no native qualification",
                pins=pins,
                verdict="full-cell blocks insufficient at N32",
                expected_negative_verified=expected_negative,
                profiles=profiles,
            ),
            indent=2,
        )
    )
    if not expected_negative:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
