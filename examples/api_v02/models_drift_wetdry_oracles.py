"""Independent finite-volume receivers; never called by a PoPS Program."""
from __future__ import annotations

import numpy as np


def wet_dry_rhs(state, spacing, *, source_mode="both"):
    """Hydrostatic FV from pressure-corrected fluxes seen by each cell.

    The left/right cell sees its own hydrostatic pressure correction. This
    spelling does not call FaceBalance or the production equation callback.
    """
    h, discharge, bed = np.asarray(state)
    hr, qr, zr = (np.roll(item, -1, axis=-1) for item in (h, discharge, bed))
    ul = np.divide(discharge, h, out=np.zeros_like(h), where=h > 0)
    ur = np.divide(qr, hr, out=np.zeros_like(hr), where=hr > 0)
    crest = np.maximum(bed, zr)
    hl_star = np.maximum(h+bed-crest, 0.)
    hr_star = np.maximum(hr+zr-crest, 0.)
    speed = np.maximum(abs(ul)+np.sqrt(h), abs(ur)+np.sqrt(hr))
    face_mass = .5*(hl_star*ul+hr_star*ur-speed*(hr_star-hl_star))
    face_momentum = .5*(hl_star*ul**2+hl_star**2/2
                         +hr_star*ur**2+hr_star**2/2
                         -speed*(hr_star*ur-hl_star*ul))
    left_correction = (h*h-hl_star*hl_star)/2
    right_correction = (hr*hr-hr_star*hr_star)/2
    if source_mode == "none":
        left_correction = np.zeros_like(h)
        right_correction = np.zeros_like(h)
    elif source_mode == "omit_left":
        left_correction = np.zeros_like(h)
    elif source_mode == "omit_right":
        right_correction = np.zeros_like(h)
    elif source_mode == "swap":
        left_correction, right_correction = right_correction, left_correction
    elif source_mode != "both":
        raise ValueError("unknown hydrostatic source counterexample")
    outgoing = face_momentum+left_correction
    incoming = np.roll(face_momentum+right_correction, 1, axis=-1)
    return np.stack((-(face_mass-np.roll(face_mass, 1, axis=-1))/spacing,
                     -(outgoing-incoming)/spacing, np.zeros_like(h)))


def ssprk2(initial, rhs, *, dt, steps=2):
    value = np.asarray(initial).copy()
    for _ in range(steps):
        predictor = value+dt*rhs(value)
        value = .5*(value+predictor+dt*rhs(predictor))
    return value


def periodic_potential(density, lengths):
    ny, nx = density.shape
    eigen = ((2*nx/lengths[0]*np.sin(np.pi*np.fft.fftfreq(nx)))**2)[None, :]
    eigen = eigen+((2*ny/lengths[1]*np.sin(np.pi*np.fft.fftfreq(ny)))**2)[:, None]
    spectrum = np.fft.fft2(density-density.mean())
    return np.fft.ifft2(np.divide(spectrum, eigen, out=np.zeros_like(spectrum),
                                 where=eigen != 0)).real


def bernoulli_quadrature(jump):
    """B(z) = 1/integral_0^1 exp(z*s) ds, independent Gauss quadrature.

    This receiver covers the bounded potentials in the PDE witness; it is not
    the production stable-asymptotic implementation of Bernoulli.
    """
    jump = np.asarray(jump, dtype=float)
    if np.max(abs(jump), initial=0.) > 4:
        raise ValueError("quadrature oracle is qualified only for abs(jump)<=4")
    nodes, weights = np.polynomial.legendre.leggauss(24)
    return 1/sum(.5*w*np.exp(.5*(x+1)*jump)
                 for x, w in zip(nodes, weights, strict=True))


def drift_diffusion_rhs(state, lengths, *, diffusion=.1, mobility=.1,
                        potential=None):
    n = np.asarray(state)[0]
    phi = periodic_potential(n, lengths) if potential is None else potential
    result = np.zeros_like(n)
    for axis in (0, 1):
        spacing = lengths[1-axis]/n.shape[axis]
        nr = np.roll(n, -1, axis=axis)
        jump = mobility/diffusion*(np.roll(phi, -1, axis=axis)-phi)
        # Solve the constant-field cell-edge ODE by its exponential integral.
        current = diffusion/spacing*(n*bernoulli_quadrature(jump)
                                    -nr*bernoulli_quadrature(-jump))
        result -= (current-np.roll(current, 1, axis=axis))/spacing
    return result[None]
