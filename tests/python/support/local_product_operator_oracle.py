"""Independent NumPy equations; no PoPS import or compiled residual evaluation."""

import numpy as np

WIDTHS = (1, 2, 4)
GAINS = ((.2, .65, 1.1), (1.3, .35, .8))
RESIDUAL_TOLERANCE = 1.e-11


def manufactured_solution(widths=WIDTHS, cells=4):
    y, x = np.meshgrid((np.arange(cells) + .5) / cells,
                       (np.arange(cells) + .5) / cells, indexing="ij")
    return tuple(np.stack([.25 + .07 * block + .035 * component + .04 * x + .06 * y
                           for component in range(width)])
                 for block, width in enumerate(widths))


def equation_without_capture(values, gains):
    total = sum(np.sum(value, axis=0) for value in values)
    return tuple(4. * gain * value**2
                 + np.arange(1, value.shape[0] + 1)[:, None, None] * value
                 + .1 * total
                 for value, gain in zip(values, gains, strict=True))


def original_residual(values, captures, gains):
    return tuple(lhs - old for lhs, old in
                 zip(equation_without_capture(values, gains), captures, strict=True))


def incompatible_sum_lower_bound(captures, gains):
    """Min sum(R) = -sum((c+1+0.1*N)^2/(16*g)) - sum(old), for g>0.

    Completing each independent square minimizes over all real x. A strictly
    positive result proves the original equations have no simultaneous zero.
    """
    count = sum(value.shape[0] for value in captures)
    if not all(gain > 0. for gain in gains):
        raise ValueError("incompatibility proof requires strictly positive gains")
    minimum_quadratics = sum(np.sum((np.arange(1, value.shape[0] + 1) + .1 * count)**2
                                    / (16. * gain))
                             for value, gain in zip(captures, gains, strict=True))
    return -minimum_quadratics - sum(np.sum(value, axis=0) for value in captures)
