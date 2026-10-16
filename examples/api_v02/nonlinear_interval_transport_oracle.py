"""Independent NumPy reception oracle; never used by generated equations.

Gauss quadrature follows the Duhamel characteristic itself. Directional
derivatives use complex-step differentiation, without the authored Jacobian or
closed time-integration coefficients.
"""
from __future__ import annotations

import numpy as np

VELOCITIES = np.array((-1., 0., 1.))


def equilibrium(rho, momentum, theta=.25):
    pressure = theta*rho+momentum*momentum/rho
    return np.stack(((pressure-momentum)/2, rho-pressure, (pressure+momentum)/2))


def initial_values(cells=(24, 4), theta=.25):
    x, _ = np.meshgrid((np.arange(cells[0])+.5)/cells[0],
                      (np.arange(cells[1])+.5)/cells[1])
    rho = 1+.1*np.sin(2*np.pi*x)
    momentum = rho*.07*np.cos(2*np.pi*x)
    perturbation = .015*np.sin(4*np.pi*x)
    return equilibrium(rho, momentum, theta)+np.stack(
        (.5*perturbation, -perturbation, .5*perturbation))


def moments(values):
    return values.sum(axis=0), values[2]-values[0]


def reconstructed_face(values, dx):
    gradient = (np.roll(values, -1, axis=-1)-np.roll(values, 1, axis=-1))/(2*dx)
    left = np.roll(values, 1, axis=-1)+dx/2*np.roll(gradient, 1, axis=-1)
    right = values-dx/2*gradient
    first = np.stack((right[0], .5*(left[1]+right[1]), left[2]))
    slope = np.stack((gradient[0], .5*(np.roll(gradient[1], 1, axis=-1)+gradient[1]),
                      np.roll(gradient[2], 1, axis=-1)))
    return first, slope


def equilibrium_derivatives(first, slope, theta=.25):
    rho, momentum = moments(first)
    rho_x, momentum_x = moments(slope)
    epsilon = 1.e-30
    rho_complex = rho+1j*epsilon*rho_x
    momentum_complex = momentum+1j*epsilon*momentum_x
    gradient = np.imag(equilibrium(rho_complex, momentum_complex, theta))/epsilon
    pressure_gradient = (theta*rho_complex+momentum_complex**2/rho_complex).imag/epsilon
    rho_t, momentum_t = -momentum_x, -pressure_gradient
    temporal = np.imag(equilibrium(rho+1j*epsilon*rho_t, momentum+1j*epsilon*momentum_t, theta))/epsilon
    return equilibrium(rho, momentum, theta), gradient, temporal


def quadrature_step(values, h, dx, tau=.03, theta=.25, *, nodes=14,
                    temporal_equilibrium=True, flux_duration=None, transported_equilibrium=True):
    duration = h if flux_duration is None else flux_duration
    roots, weights = np.polynomial.legendre.leggauss(nodes)
    roots, weights = (roots+1)/2, weights/2
    first, slope = reconstructed_face(values, dx)
    geq, gx, gt = equilibrium_derivatives(first, slope, theta)
    if not temporal_equilibrium:
        gt = np.zeros_like(gt)
    flux = np.zeros_like(values)
    for component, velocity in enumerate(VELOCITIES):
        for t, wt in zip(duration*roots, duration*weights, strict=True):
            characteristic = np.exp(-t/tau)*(first[component]-velocity*t*slope[component])
            for s, ws in zip(t*roots, t*weights, strict=True):
                source = geq[component]-velocity*(t-s)*gx[component]+s*gt[component]
                characteristic += ws/tau*np.exp(-(t-s)/tau)*source
            flux[component] += wt*velocity*characteristic/duration
    transported = values+h/dx*(flux-np.roll(flux, -1, axis=-1))
    old_equilibrium = equilibrium(*moments(values), theta)
    next_equilibrium = equilibrium(*moments(transported), theta) if transported_equilibrium else old_equilibrium
    # Solve the trapezoidal local collision relation directly after transport.
    z = h/(2*tau)
    return (transported+z*(old_equilibrium-values+next_equilibrium))/(1+z)


def reference(values, h, dx, steps, tau=.03, theta=.25, **mutant):
    result = values.copy()
    for _ in range(steps):
        result = quadrature_step(result, h, dx, tau, theta, **mutant)
    return result
