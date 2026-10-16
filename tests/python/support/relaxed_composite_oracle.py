"""Independent volume pairing and affine reception of gathered native levels."""
import numpy as np
from examples.api_v02.method_amr_reception import composite_base_projection


def shaped(levels, cells=(32, 16)):
    nx, ny = cells
    return [np.asarray(value).reshape(3, ny*2**level, nx*2**level)
            for level, value in enumerate(levels)]


def pair(left, right, boxes, *, cells=(32, 16), weights=(1., 1., 0.)):
    weights = np.asarray(weights)[:, None, None]
    products = [np.sum(weights*a*b, axis=0) for a, b in
                zip(shaped(left, cells), shaped(right, cells), strict=True)]
    projection = composite_base_projection(products, boxes, cells=cells)
    assert all(count > 0 for count in projection["active_counts"])
    return float(projection["means"].mean())


def relaxation(old, provisional, boxes, *, cells=(32, 16)):
    a = pair(old, provisional, boxes, cells=cells)-pair(old, old, boxes, cells=cells)
    b = (pair(provisional, provisional, boxes, cells=cells)
         -2*pair(old, provisional, boxes, cells=cells)+pair(old, old, boxes, cells=cells))
    assert b > 0 and a < 0
    return -2*a/b


def affine_error(actual, old, provisional, factor, *, cells=(32, 16)):
    return max(float(np.max(abs(a-(o+factor*(b-o))))) for a, o, b in zip(
        shaped(actual, cells), shaped(old, cells), shaped(provisional, cells), strict=True))


def synthetic_rotation(temporal_ratio, *, cells=(4, 4)):
    """Half the unit square is refined, exposing both true composite levels."""
    nx, ny = cells
    boxes = ((0, (0, 0), (nx-1, ny-1)), (1, (0, 0), (nx-1, 2*ny-1)))
    old, provisional = [], []
    for level in (0, 1):
        shape = (3, ny*2**level, nx*2**level)
        base = np.zeros(shape); base[0] = 1; base[2] = .5
        value = base.copy()
        steps = temporal_ratio if level else 1
        h = 1/steps
        for _ in range(steps):
            x, y = value[0].copy(), value[1].copy()
            value[0], value[1] = (1-h*h/2)*x-h*y, h*x+(1-h*h/2)*y
        old.append(base); provisional.append(value)
    return old, provisional, boxes
