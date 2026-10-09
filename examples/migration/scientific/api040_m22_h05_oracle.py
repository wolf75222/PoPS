"""Independent H05 linear reservoirs and scalar M1 endpoint closure.

No PoPS import. H05 is not the physical T**4 matter/radiation exchange.
"""
import math
import numpy as np


def _admit(initial, k, duration):
    values = np.asarray(initial, dtype=float)
    if values.shape[0] != 2 or not np.all(np.isfinite(values)) or np.any(values < 0):
        raise ValueError("H05 requires two finite nonnegative reservoir energies")
    if not math.isfinite(k) or k < 0 or not math.isfinite(duration) or duration < 0:
        raise ValueError("H05 requires finite k>=0 and duration>=0")
    return values


def backward_euler(initial, k, dt, steps):
    values = _admit(initial, k, dt).copy()
    if type(steps) is not int or steps < 0:
        raise ValueError("steps must be a nonnegative integer")
    coupling = dt*k
    if not math.isfinite(coupling):
        raise ValueError("H05 discrete coefficient is not representable")
    matrix = np.array([[1+coupling, -coupling], [-coupling, 1+coupling]])
    shape = values.shape
    for _ in range(steps):
        values = np.linalg.solve(matrix, values.reshape(2,-1)).reshape(shape)
        if not np.all(np.isfinite(values)):
            raise ValueError("H05 oracle solve returned a nonfinite value")
    return values


def continuous(initial, k, duration):
    values = _admit(initial, k, duration)
    mean = .5*(values[0]+values[1])
    difference = .5*(values[0]-values[1])*math.exp(-2*k*duration)
    return np.stack((mean+difference, mean-difference))


def chi(energy, flux, c=1.):
    """Only scalar Eddington factor from the corpus; no spatial M1 PDE claim."""
    flux = np.asarray(flux,dtype=float)
    if not math.isfinite(energy) or energy <= 0 or not math.isfinite(c) or c <= 0:
        raise ValueError("closure requires finite E>0 and c>0; no vacuum extension supplied")
    if flux.ndim != 1 or flux.size == 0 or not np.all(np.isfinite(flux)):
        raise ValueError("closure requires a finite flux vector")
    reduced = np.linalg.norm(flux/energy/c)
    if reduced > 1:
        raise ValueError("M1 admissibility requires |F|<=cE; no clipping")
    # Rationalized formula, independent of the corpus quotient implementation.
    return (5-2*math.sqrt(4-3*reduced*reduced))/3
