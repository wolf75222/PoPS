"""Independent finite-quadrature oracle; no PoPS algebra or runtime imports."""

from itertools import combinations

import numpy as np


NODES = np.asarray((-1., -.5, 0., .5, 1.))
WEIGHTS = np.asarray((.1, .2, .4, .2, .1))
BASIS = np.vstack((np.ones_like(NODES), NODES, NODES*NODES))


def target_from_multiplier(multiplier, basis=BASIS, weights=WEIGHTS):
    multiplier = np.asarray(multiplier, dtype=float)
    return basis @ (weights * np.exp(basis.T @ multiplier))


def cone_position(target, basis=BASIS, *, tolerance=2.e-11):
    """Enumerate basic feasible populations, then maximize their common floor.

    For three independent constraints and five nonnegative populations, every
    nonempty feasible polytope has an extreme point with at most three nonzeros.
    Its two-dimensional nullspace permits a separate max-min interior check.
    """
    target = np.asarray(target, dtype=float)
    rows, columns = basis.shape
    if target.shape != (rows,):
        raise ValueError("target has a different moment count")
    feasible = None
    for active in combinations(range(columns), rows):
        submatrix = basis[:, active]
        if np.linalg.matrix_rank(submatrix) != rows:
            continue
        values = np.linalg.solve(submatrix, target)
        if np.min(values) >= -tolerance and np.max(np.abs(submatrix@values-target)) <= tolerance:
            feasible = np.zeros(columns)
            feasible[list(active)] = values
            break
    if feasible is None:
        return "outside", None
    _, _, right = np.linalg.svd(basis, full_matrices=True)
    nullspace = right[rows:].T
    if nullspace.shape[1] != 2:
        raise ValueError("this small LP oracle expects a two-dimensional nullspace")
    best_floor = -np.inf
    for active in combinations(range(columns), 3):
        equations = np.column_stack((nullspace[list(active)], -np.ones(3)))
        if np.linalg.matrix_rank(equations) != 3:
            continue
        xyt = np.linalg.solve(equations, -feasible[list(active)])
        populations = feasible + nullspace @ xyt[:2]
        if np.min(populations-xyt[2]) >= -tolerance:
            best_floor = max(best_floor, xyt[2])
    if not np.isfinite(best_floor):
        raise AssertionError("feasible population polytope has no max-min LP vertex")
    return ("interior" if best_floor > tolerance else "boundary"), best_floor


def damped_dual_newton(target, basis=BASIS, weights=WEIGHTS, *, tolerance=1.e-12):
    """Minimize the independent convex dual with backtracking Armijo descent."""
    target = np.asarray(target, dtype=float)
    multiplier = np.zeros(basis.shape[0])
    for _ in range(80):
        population = weights * np.exp(basis.T @ multiplier)
        residual = basis @ population - target
        if np.max(np.abs(residual)) <= tolerance:
            return multiplier, population
        hessian = (basis * population) @ basis.T
        direction = np.linalg.solve(hessian, -residual)
        objective = population.sum() - target @ multiplier
        descent = residual @ direction
        step = 1.
        while step >= 2.**-30:
            proposal = multiplier + step*direction
            trial = weights * np.exp(basis.T @ proposal)
            trial_residual = basis @ trial - target
            armijo = trial.sum()-target@proposal <= objective + .0001*step*descent
            # Near the minimizer, the dual decrease is below one binary64 ulp.
            # Require a strict residual decrease there rather than accepting
            # equal rounded objectives and stalling the independent oracle.
            rounded_objective = abs(step*descent) <= 8*np.finfo(float).eps*max(1., abs(objective))
            small_step_progress = (rounded_objective and
                                   np.max(np.abs(trial_residual)) < np.max(np.abs(residual)))
            if np.all(np.isfinite(trial)) and (small_step_progress or (armijo and not rounded_objective)):
                multiplier = proposal
                break
            step *= .5
        else:
            raise AssertionError("independent damped dual Newton could not descend")
    raise AssertionError("independent damped dual Newton did not converge")


def entropy(population, weights=WEIGHTS):
    population = np.asarray(population, dtype=float)
    if np.any(population <= 0):
        raise ValueError("entropy comparison requires positive populations")
    return float(np.sum(population*(np.log(population/weights)-1)))
