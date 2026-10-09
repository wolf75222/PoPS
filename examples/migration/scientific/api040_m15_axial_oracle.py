"""Independent NumPy finite-volume oracle for the fixed five-moment B.1 variant.

No PoPS import, expression interpreter, generated Jacobian, or native solver.
Arrays use canonical order M0..M4 on the leading axis.
"""
from __future__ import annotations

import math
import numpy as np


RESOLUTIONS = (32, 64, 128)
T_END = .02
ORDERS = ((0, 1, 2, 3, 4), (4, 1, 3, 0, 2))
CRITERIA = {
    "initial_max_error": 2.e-14,
    "state_max_error": 3.e-11,
    "moment_integral_error": 2.e-12,
    "time_error": 2.e-14,
    "permutation_max_error": 3.e-12,
    "spectral_imaginary_tolerance": 2.e-11,
    "preflight_face_courant_max": .1,
}


def normal_moments(mean, variance):
    return np.array([sum(math.comb(k, 2*j) * math.factorial(2*j)
                        / (2**j * math.factorial(j)) * variance**j * mean**(k-2*j)
                        for j in range(k//2+1)) for k in range(5)])


def initial_averages(cells):
    if type(cells) is not int or cells < 4:
        raise ValueError("cells must be an integer >=4")
    phase = 2*np.pi*(np.arange(cells)+.5)/cells
    attenuation = np.sinc(1/cells)
    left = .6 + .08*attenuation*np.cos(phase)
    right = .4 + .06*attenuation*np.sin(phase)
    return (normal_moments(-.4, .3)[:, None]*left
            + normal_moments(.7, .2)[:, None]*right)


def domain_values(raw):
    rho, m1, m2, m3, m4 = np.asarray(raw)
    h2 = rho*m2-m1*m1
    h3 = rho*(m2*m4-m3*m3)-m1*(m1*m4-m2*m3)+m2*(m1*m3-m2*m2)
    return rho, h2, h3


def require_admissible(raw):
    if np.shape(raw)[0] != 5 or not np.isfinite(raw).all():
        raise ValueError("five finite axial moments required")
    rho, h2, h3 = domain_values(raw)
    if np.any(rho <= 0) or np.any(h2 <= 0) or np.any(h3 < 0):
        raise ValueError("outside positive density/variance/Hankel moment domain")


def flux(raw):
    # Algebraically rational central-moment B.1 formula, independent of the
    # production sqrt/standardized expression and its automatic differentiation.
    rho, m1, m2, m3, m4 = np.asarray(raw)
    u = m1/rho
    c2 = m2-rho*u*u
    c3 = m3-3*u*m2+2*rho*u**3
    c4 = m4-4*u*m3+6*u*u*m2-3*rho*u**4
    c5 = .5*c3*(5*c4/c2-3*(c3/c2)**2-c2/rho)
    m5 = c5+5*u*c4+10*u*u*c3+10*u**3*c2+rho*u**5
    return np.stack((m1, m2, m3, m4, m5))


def jacobian(raw):
    state = np.asarray(raw)
    columns = []
    for column in range(5):
        shifted = state.astype(complex)
        shifted[column] += 1.e-30j
        columns.append(flux(shifted).imag/1.e-30)
    return np.moveaxis(np.stack(columns, axis=1), (0, 1), (-2, -1))


def speeds(raw):
    require_admissible(raw)
    spectrum = np.linalg.eigvals(jacobian(raw))
    if not np.isfinite(spectrum).all() or np.max(np.abs(spectrum.imag)) > CRITERIA["spectral_imaginary_tolerance"]:
        raise ValueError("nonreal or nonfinite axial spectrum")
    return spectrum.real.min(axis=-1), spectrum.real.max(axis=-1)


def rhs(raw):
    lower, upper = speeds(raw)
    right = np.roll(raw, -1, axis=1)
    sminus = np.minimum(lower, np.roll(lower, -1))
    splus = np.maximum(upper, np.roll(upper, -1))
    left_flux, right_flux = flux(raw), flux(right)
    # All selected mixtures have sminus<0<splus. The complete upwind branches
    # remain explicit so this function does not silently change the HLL method.
    middle = (splus*left_flux-sminus*right_flux+sminus*splus*(right-raw))/(splus-sminus)
    face = np.where(sminus >= 0, left_flux, np.where(splus <= 0, right_flux, middle))
    frequency = raw.shape[1]*np.maximum(np.abs(sminus), np.abs(splus)).max()
    return -raw.shape[1]*(face-np.roll(face, 1, axis=1)), float(frequency)


def trajectory(cells, *, steps=None):
    dt = 1/(100*cells)
    count = 2*cells if steps is None else steps
    raw = initial_averages(cells)
    maximum_courant = 0.
    for _ in range(count):
        rate, frequency = rhs(raw)
        maximum_courant = max(maximum_courant, dt*frequency)
        raw = raw+dt*rate
        require_admissible(raw)
    if maximum_courant > CRITERIA["preflight_face_courant_max"]:
        raise ValueError("predeclared axial face-Courant budget exceeded")
    return raw, maximum_courant


def invalid_states(cells):
    return {
        "negative_density": np.repeat(np.array([-1., 0., -1., 0., 0.])[:, None], cells, axis=1),
        "zero_variance": np.repeat(np.array([1., 0., 0., 0., 1.])[:, None], cells, axis=1),
        "negative_hankel": np.repeat(np.array([1., 0., 1., 0., .5])[:, None], cells, axis=1),
    }
