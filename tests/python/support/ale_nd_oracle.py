"""Independent NumPy moving-volume reference with Gaussian face quadrature."""
from __future__ import annotations

import numpy as np


def geometry(shape, amplitudes, time):
    lower, upper = [], []
    for axis, (count, amplitude) in enumerate(zip(shape, amplitudes, strict=True)):
        nodes = np.arange(count + 1, dtype=float) / count
        positions = nodes + amplitude * np.sin(2 * np.pi * nodes) * np.sin(2 * np.pi * time)
        positions[[0, -1]] = (0., 1.)
        axis_shape = [1] * len(shape)
        axis_shape[axis] = count
        lower.append(np.broadcast_to(positions[:-1].reshape(axis_shape), shape))
        upper.append(np.broadcast_to(positions[1:].reshape(axis_shape), shape))
    widths = np.asarray(upper) - np.asarray(lower)
    return dict(lower=np.asarray(lower), upper=np.asarray(upper), volume=np.prod(widths, axis=0))


def ale_reference(*, shape=(8, 6), amplitudes=(.025, .018), velocities=(.7, -.3),
                  dt=.001, steps=4, nonuniform=True, old_volume_mutant=False):
    """Integrate area by quadrature, independently of the symbolic subset expansion."""
    dimension = len(shape)
    coordinates = np.meshgrid(*[(np.arange(n)+.5)/n for n in shape], indexing="ij")
    density = np.full(shape, 1.3)
    if nonuniform:
        density += sum(.04/dimension*np.cos(2*np.pi*x) for x in coordinates)
    uniform = np.full(shape, .8)
    quadrature, weights = np.polynomial.legendre.leggauss(max(1, (dimension+1)//2))
    records = []
    old = geometry(shape, amplitudes, 0.)
    for step in range(steps):
        new = geometry(shape, amplitudes, (step+1)*dt)
        amounts = [density*old["volume"], uniform*old["volume"]]
        swept_divergence = np.zeros(shape)
        outgoing = np.zeros(shape)
        for axis, velocity in enumerate(velocities):
            area = np.zeros(shape)
            for point, weight in zip((quadrature+1)/2, weights/2, strict=True):
                widths = ((1-point)*(old["upper"]-old["lower"])
                          + point*(new["upper"]-new["lower"]))
                area += weight*np.prod(np.delete(widths, axis, axis=0), axis=0)
            swept = (new["upper"][axis]-old["upper"][axis])*area
            swept_divergence += swept-np.roll(swept, 1, axis)
            relative = dt*velocity*area-swept
            outgoing += np.maximum(relative, 0)+np.maximum(-np.roll(relative, 1, axis), 0)
            for index, q in enumerate((density, uniform)):
                flux = relative*np.where(relative >= 0, q, np.roll(q, -1, axis))
                amounts[index] += np.roll(flux, 1, axis)-flux
        denominator = old["volume"] if old_volume_mutant else new["volume"]
        next_density, next_uniform = (a/denominator for a in amounts)
        records.append(dict(
            gcl=float(np.max(abs(new["volume"]-old["volume"]-swept_divergence))),
            mass=float(abs(np.sum(next_density*new["volume"])-np.sum(density*old["volume"]))),
            uniform=float(np.max(abs(next_uniform-.8))),
            cfl_margin=float(np.min(old["volume"]-outgoing)),
        ))
        density, uniform, old = next_density, next_uniform, new
    return dict(density=density, uniform=uniform, geometry=old), records
