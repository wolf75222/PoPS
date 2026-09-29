"""Independent finite constrained-friction oracle for the exact W10 data."""

import numpy as np

FRICTION = np.array(((0., 2., 1.), (2., 0., 3.), (1., 3., 0.)))
LAPLACIAN = np.diag(FRICTION.sum(axis=1)) - FRICTION
BASE_FORCE = np.array((.3, -.4, .1))
THRESHOLD = 1.e-12


def augmented_matrix():
    result = np.zeros((4, 4))
    result[:3, :3] = LAPLACIAN
    result[:3, 3] = 1.
    result[3, :3] = 1.
    return result


def force_samples(cells=4):
    y, x = np.meshgrid((np.arange(cells) + .5) / cells,
                       (np.arange(cells) + .5) / cells, indexing="ij")
    baseline = np.broadcast_to(BASE_FORCE[:, None, None], (3, cells, cells)).copy()
    results = [baseline]
    for shift, scale in ((0., 1.), (.31, .7)):
        first = scale * (.3 + .1 * np.sin(2*np.pi*(x+shift)) + .03*np.cos(2*np.pi*y))
        second = scale * (-.4 + .04*np.cos(2*np.pi*x) - .02*np.sin(2*np.pi*(y+shift)))
        results.append(np.stack((first, second, -first-second)))
    return tuple(results)


def reference(force):
    """Solve the 4x4 reference independently; never invert singular L."""
    rhs = np.concatenate((force, np.zeros_like(force[:1])), axis=0)
    result = np.linalg.solve(augmented_matrix(), rhs.reshape(4, -1)).reshape(rhs.shape)
    return result[:3], result[3:]


def residuals(force, flux, multiplier):
    original = np.einsum("ij,j...->i...", LAPLACIAN, flux) - force
    constraint = flux.sum(axis=0)
    augmented = original + multiplier
    return {"original_residual": float(np.max(np.abs(original))),
            "constraint": float(np.max(np.abs(constraint))),
            "augmented_residual": float(np.max(np.abs(augmented))),
            "lambda_max_abs": float(np.max(np.abs(multiplier)))}
