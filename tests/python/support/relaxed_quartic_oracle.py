"""Independent polynomial root oracle on gathered native AMR cell averages."""
import numpy as np
from examples.api_v02.method_amr_reception import composite_base_projection
from tests.python.support.relaxed_composite_oracle import shaped


def integral(values, boxes, *, cells=(32, 16)):
    result = composite_base_projection(values, boxes, cells=cells)
    assert all(n > 0 for n in result["active_counts"])
    return float(result["means"].mean())


def energy(levels, boxes, *, cells=(32, 16)):
    return integral([a[0]**4+a[1]**4 for a in shaped(levels, cells)], boxes, cells=cells)


def relaxation(old, provisional, boxes, *, cells=(32, 16)):
    base = shaped(old, cells)
    delta = [b-a for a, b in zip(base, shaped(provisional, cells), strict=True)]
    coefficients = [integral([np.sum(weight*a[:2]**(4-power)*d[:2]**power, axis=0)
        for a, d in zip(base, delta, strict=True)], boxes, cells=cells)
        for power, weight in ((1, 4), (2, 6), (3, 4), (4, 1))]
    roots = np.roots(coefficients[::-1])
    physical = [float(z.real) for z in roots if abs(z.imag) < 1.e-10 and z.real > .25]
    assert len(physical) == 1, roots
    return physical[0]
