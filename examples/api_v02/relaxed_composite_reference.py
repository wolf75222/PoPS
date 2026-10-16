"""Independent NumPy rotation oracle for the declared two-level Heun witness.

Transport is orthogonal to these two components. Actual patch boxes specify the
volume fraction; their values never enter the reference recurrence.
"""
from __future__ import annotations

import numpy as np

from .method_amr_reception import composite_base_projection


def rotation_reference(boxes, *, cells, ratio, requested, end):
    nx, ny = cells
    masks = [np.ones((ny, nx), dtype=bool), np.zeros((2*ny, 2*nx), dtype=bool)]
    for level, (x0, y0), (x1, y1) in boxes:
        if level == 1:
            masks[1][y0:y1+1, x0:x1+1] = True
            masks[0][y0//2:y1//2+1, x0//2:x1//2+1] = False
    weights = np.array([masks[0].sum(), masks[1].sum()/4], dtype=np.longdouble)/(nx*ny)
    if weights.sum() != 1 or any(weight <= 0 for weight in weights):
        raise ValueError("reference requires a complete, partly refined two-level domain")
    state = np.array([[1., 0.], [1., 0.]], dtype=np.longdouble)
    time = 0.
    requests = []
    for _ in range(32):
        if time >= end:
            break
        h = min(requested, end-time)
        requests.append(h)
        h = np.longdouble(h)
        # Exact real/imaginary parts of R(h)-1 and R(h/2)^2-1,
        # R(h)=1+i*h-h^2/2. No subtraction of nearly equal energies.
        real = np.array([-h*h/2, -h*h/2 + (h**4/64 if ratio == 2 else 0)])
        imag = np.array([h, h - (h**3/8 if ratio == 2 else 0)])
        norm = (state*state).sum(axis=1)
        gamma = -2*np.sum(weights*norm*real)/np.sum(weights*norm*(real*real+imag*imag))
        x, y = state[:, 0].copy(), state[:, 1].copy()
        state[:, 0] += gamma*(real*x-imag*y)
        state[:, 1] += gamma*(imag*x+real*y)
        time += float(gamma*h)
    if time < end:
        raise ValueError("reference did not reach its declared frontier")
    return state.astype(float), masks, requests, time


def trajectory_metrics(levels, boxes, *, cells, ratio, requested, end, initial_passive):
    expected, masks, requests, reached = rotation_reference(
        boxes, cells=cells, ratio=ratio, requested=requested, end=end)
    nx, ny = cells
    shaped = [np.asarray(value).reshape(3, ny*2**level, nx*2**level)
              for level, value in enumerate(levels)]
    error = max(float(np.max(abs(field[:2, mask]-expected[level, :, None])))
                for level, (field, mask) in enumerate(zip(shaped, masks, strict=True)))
    energy = composite_base_projection([(a[:2]*a[:2]).sum(axis=0) for a in shaped],
                                       boxes, cells=cells)["means"].mean()
    mass = composite_base_projection([a[2] for a in shaped], boxes, cells=cells)["means"].mean()
    return dict(passed=bool(error < 2.e-13 and abs(energy-1) < 2.e-13
                            and abs(mass-initial_passive) < 2.e-13),
                rotation_error=error, energy_error=float(abs(energy-1)),
                passive_mass_error=float(abs(mass-initial_passive)),
                reference_requests=requests, reference_time=reached)
