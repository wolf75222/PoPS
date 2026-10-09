"""Dense source-derived first-Jacobian algebra; no PoPS/AMR runtime or native states.

Only the frozen C++ 1D manufactured request is covered. The matrix rows are
derived from its actual quadratic C/F ghosts, restriction and reflux formulas.
No nonlinear/time integration, field resource or publication is implemented.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess

import numpy as np

FROZEN = "9fb7e8fb64c3e3de289d9c139ba310e9e47e43a6"
ROOT = Path(__file__).resolve().parents[2]
SOURCES = (
    "tests/cpp/unit/elliptic/test_composite_general_field.cpp",
    "tests/cpp/unit/elliptic/amr_original_field_residual.inc",
    "include/pops/numerics/elliptic/mg/composite_fac_nlevel.hpp",
    "include/pops/numerics/elliptic/amr/composite_fac_poisson.hpp",
    "include/pops/numerics/elliptic/nd/prepared_composite_general_field.hpp",
    "include/pops/runtime/program/prepared_amr_field_residual.hpp",
    "include/pops/numerics/elliptic/interface/amr_field_newton_krylov.hpp",
)


def source_data():
    texts, pins = {}, []
    for path in SOURCES:
        raw = subprocess.check_output(["git", "show", FROZEN + ":" + path], cwd=ROOT)
        texts[path] = raw.decode()
        pins.append(dict(path=path, sha256=hashlib.sha256(raw).hexdigest()))
    witness = texts[SOURCES[1]]

    def array(name):
        values = re.search(r"\b" + name + r"\{([^}]+)\}", witness).group(1)
        return np.array([float(value.strip()) for value in values.split(",")])

    controls = {
        name: int(re.search(r"\." + name + r"\s*=\s*(\d+)", witness).group(1))
        for name in ("restart", "linear_max_iterations")
    }
    controls["linear_tolerance"] = float(
        re.search(r"\.linear_tolerance\s*=\s*Real\(([^)]+)\)", witness).group(1)
    )
    cubic = float(re.search(r"Real\(([^)]+)\)\s*\*\s*u\s*\*\s*u\s*\*\s*u", witness).group(1))
    return dict(
        diffusion=array("diffusion").reshape(3, 3),
        reaction=array("reaction").reshape(3, 3),
        cubic=cubic,
        means=array("means"),
        waves=array("waves"),
        controls=controls,
        pins=pins,
    )


def source_matrix(cells, permutation=(0, 1, 2)):
    data = source_data()
    active = list(range(cells // 4)) + list(range(3 * cells // 4, cells))
    fine = list(range(cells // 2, 3 * cells // 2))
    width = len(active) + len(fine)
    # Each row below is a linear coefficient vector on active scalar DOFs.
    parent = np.zeros((cells, width))
    child = np.eye(width)[len(active) :]
    for position, cell in enumerate(active):
        parent[cell, position] = 1.0
    for cell in range(cells // 4, 3 * cells // 4):
        parent[cell] = 0.5 * child[2 * cell - cells // 2] + 0.5 * child[2 * cell + 1 - cells // 2]
    # QuadraticInterpolationTransfer weights for ratio2, child1/0: s=+/-.25.
    ghosts = {
        cells // 2 - 1: -0.09375 * parent[cells // 4 - 2]
        + 0.9375 * parent[cells // 4 - 1]
        + 0.15625 * parent[cells // 4],
        3 * cells // 2: 0.15625 * parent[3 * cells // 4 - 1]
        + 0.9375 * parent[3 * cells // 4]
        - 0.09375 * parent[3 * cells // 4 + 1],
    }

    def fine_row(cell):
        return child[cell - cells // 2] if cell in fine else ghosts[cell]

    rows = []
    for cell in active:
        # Homogeneous Neumann reflection at the actual physical faces.
        row = cells**2 * (
            2 * parent[cell] - parent[max(0, cell - 1)] - parent[min(cells - 1, cell + 1)]
        )
        if cell in (cells // 4 - 1, 3 * cells // 4):
            side = -1 if cell == cells // 4 - 1 else 1
            inner = 2 * cell + (2 if side < 0 else -1)
            # Actual matrix_entry sign=-1 and 1D fine_face_weight=ratio=2.
            row -= cells**2 * (
                (parent[cell] - parent[cell - side])
                - 2 * (fine_row(inner + side) - fine_row(inner))
            )
        rows.append(row)
    for cell in fine:
        rows.append(4 * cells**2 * (2 * fine_row(cell) - fine_row(cell - 1) - fine_row(cell + 1)))
    scalar = np.array(rows)
    order = np.array(permutation)
    diffusion = data["diffusion"][np.ix_(order, order)]
    reaction = data["reaction"][np.ix_(order, order)]
    spatial = np.kron(scalar, diffusion)
    matrix = spatial + np.kron(np.eye(width), reaction)
    centers = np.array(
        [(cell + 0.5) / cells for cell in active] + [(cell + 0.5) / (2 * cells) for cell in fine]
    )
    exact = data["means"][order] + np.cos(2 * np.pi * centers[:, None]) * data["waves"][order]
    # Exact first Jacobian at the actual null seed: cubic derivative is zero.
    rhs = matrix @ exact.ravel() + data["cubic"] * (exact**3).ravel()
    metric = np.repeat(
        np.r_[np.full(len(active), 1 / cells), np.full(len(fine), 1 / (2 * cells))], 3
    )
    return dict(
        matrix=matrix,
        spatial=spatial,
        rhs=rhs,
        metric=metric,
        scalar=scalar,
        active=active,
        fine=fine,
        controls=data["controls"],
        pins=data["pins"],
    )


def linear_algebra(problem, passes=1, spatial_right_diagonal=False):
    """Standard dense Arnoldi algebra with the unchanged authored controls.

    W^(1/2) changes coordinates only; the stop remains the actual physical norm.
    A right diagonal changes unknown coordinates only. Every cycle checks the
    actual unchanged A and b, independently of the projected residual estimate.
    """
    root_metric = np.sqrt(problem["metric"])
    matrix = root_metric[:, None] * problem["matrix"] / root_metric[None, :]
    rhs = root_metric * problem["rhs"]
    diagonal = np.diag(problem["spatial"]) if spatial_right_diagonal else np.ones(len(rhs))
    assert np.all(np.isfinite(diagonal)) and np.all(diagonal > 0)
    operator = matrix / diagonal[None, :]
    controls = problem["controls"]
    budget, restart = controls["linear_max_iterations"], controls["restart"]
    solution, residual = np.zeros(len(rhs)), rhs.copy()
    beta = float(np.linalg.norm(residual))
    stop = controls["linear_tolerance"] * beta
    completed, history = 0, []
    while completed < budget:
        capacity = min(restart, budget - completed)
        basis = [residual / beta]
        h = np.zeros((capacity + 1, capacity))
        cosine, sine, rotated = np.zeros(capacity), np.zeros(capacity), np.zeros(capacity + 1)
        rotated[0] = beta
        for column in range(capacity):
            work = operator @ basis[column]
            for _ in range(passes):
                for row in range(column + 1):
                    projection = float(work @ basis[row])
                    h[row, column] += projection
                    work -= projection * basis[row]
            h[column + 1, column] = np.linalg.norm(work)
            basis.append(
                work / h[column + 1, column] if h[column + 1, column] else np.zeros(len(rhs))
            )
            for rotation in range(column):
                upper, lower = h[rotation, column], h[rotation + 1, column]
                h[rotation, column] = cosine[rotation] * upper + sine[rotation] * lower
                h[rotation + 1, column] = -sine[rotation] * upper + cosine[rotation] * lower
            magnitude = math.hypot(h[column, column], h[column + 1, column])
            assert math.isfinite(magnitude) and magnitude > 0
            cosine[column], sine[column] = (
                h[column, column] / magnitude,
                h[column + 1, column] / magnitude,
            )
            h[column, column], h[column + 1, column] = magnitude, 0.0
            rotated[column + 1] = -sine[column] * rotated[column]
            rotated[column] *= cosine[column]
            completed += 1
            if abs(rotated[column + 1]) <= stop:
                break
        used = column + 1
        coefficients = np.zeros(used)
        for row in reversed(range(used)):
            coefficients[row] = (
                rotated[row] - float(h[row, row + 1 : used] @ coefficients[row + 1 : used])
            ) / h[row, row]
        solution += np.column_stack(basis[:used]) @ coefficients
        residual = rhs - matrix @ (solution / diagonal)
        beta = float(np.linalg.norm(residual))
        history.append(dict(columns=completed, projected=float(abs(rotated[used])), actual=beta))
        if beta <= stop:
            break
    return dict(
        converged=beta <= stop,
        columns=completed,
        stop=stop,
        actual=beta,
        passes=passes,
        spatial_right_diagonal=spatial_right_diagonal,
        cycles=history,
    )


def receive():
    cases = []
    for cells in (16, 32):
        for permutation in ((0, 1, 2), (2, 0, 1)):
            problem = source_matrix(cells, permutation)
            weighted = problem["metric"][:, None] * problem["matrix"]
            case = dict(
                cells=cells,
                permutation=list(permutation),
                dofs=len(problem["rhs"]),
                initial_norm=float(np.linalg.norm(np.sqrt(problem["metric"]) * problem["rhs"])),
                scalar_constant_error=float(
                    np.max(
                        abs(
                            problem["scalar"]
                            @ np.ones(len(problem["active"]) + len(problem["fine"]))
                        )
                    )
                ),
                conservation_error=float(np.max(abs(problem["metric"][::3] @ problem["scalar"]))),
                weighted_asymmetry=float(np.linalg.norm(weighted - weighted.T)),
                controls=problem["controls"],
                variants=[
                    linear_algebra(problem, 1),
                    linear_algebra(problem, 2),
                    linear_algebra(problem, 2, True),
                ],
            )
            cases.append(case)
    return dict(
        schema="sol61.amr-original.first-jacobian-source-math@1",
        source_commit=FROZEN,
        source_files=source_data()["pins"],
        cases=cases,
        native_execution_here=False,
        native_states_consumed=False,
        native_reproduction_claimed=False,
        scope="actual frozen 1D two-level request first Jacobian only, not an AMR runtime",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = receive()
    value = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.write_text(value)
    else:
        print(value, end="")
