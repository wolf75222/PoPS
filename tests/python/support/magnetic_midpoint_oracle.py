"""Independent NumPy discretization, used only to receive native M09 results.

The midpoint reference solves the ORIGINAL 3N by 3N matrix, with np.linalg.solve.
It does not call the Program, GMRES, Schur implementation or any PoPS evaluator.
"""
from __future__ import annotations

import math
import numpy as np


def initial(cells, *, lengths=(2*math.pi, 2*math.pi), alpha=1., omega=4.):
    nx, ny = cells
    x, y = np.meshgrid(2*np.pi*(np.arange(nx)+.5)/nx,
                       2*np.pi*(np.arange(ny)+.5)/ny)
    rho = 1+.08*np.cos(x)+.04*np.sin(y)
    vx, vy = .12+.03*np.sin(x+y), .07-.02*np.cos(x-y)
    eigen = tuple(4*np.sin(np.pi/n)**2*(n/L)**2 for n, L in zip(cells, lengths, strict=True))
    return np.stack((rho, rho*vx, rho*vy, 2.5+.5*rho*(vx*vx+vy*vy),
        alpha*(.08*np.cos(x)/eigen[0]+.04*np.sin(y)/eigen[1]),
        omega*(1+.2*np.cos(y))))


def operators(cells, lengths):
    nx, ny = cells
    difference = []
    for n, length in zip(cells, lengths, strict=True):
        one = np.roll(np.eye(n), 1, axis=1)-np.eye(n)
        difference.append(one*n/length)
    gx = np.kron(np.eye(ny), difference[0])
    gy = np.kron(difference[1], np.eye(nx))
    return gx, gy, gx.T@gx+gy.T@gy


def source(state, h, *, lengths, alpha):
    cells = (state.shape[2], state.shape[1])
    n = math.prod(cells)
    rho, mx, my, energy, phi, omega = state.reshape(6, n)
    gx, gy, k = operators(cells, lengths)
    s = h/2
    identity = np.eye(n)
    density = np.diag(rho)
    rotation = np.diag(s*omega)
    gauge = np.ones((n, n))/n
    matrix = np.block([[identity, -rotation, s*gx],
                       [rotation, identity, s*gy],
                       [-s*alpha*gx.T@density, -s*alpha*gy.T@density, k+gauge]])
    rhs = np.concatenate((mx/rho, my/rho, k@phi+np.mean(phi)))
    solution = np.linalg.solve(matrix, rhs)
    vx, vy, mid_phi = solution.reshape(3, n)
    new_vx, new_vy = 2*vx-mx/rho, 2*vy-my/rho
    updated = state.copy().reshape(6, n)
    updated[1] = rho*new_vx
    updated[2] = rho*new_vy
    updated[3] = energy+.5*rho*(new_vx**2+new_vy**2)-.5*(mx**2+my**2)/rho
    updated[4] = 2*mid_phi-phi
    residual = float(np.max(np.abs(matrix@solution-rhs)))
    return updated.reshape(state.shape), residual


def hyperbolic_rhs(state, lengths, gamma):
    rho, mx, my, energy = state[:4]
    pressure = (gamma-1)*(energy-.5*(mx*mx+my*my)/rho)
    out = np.zeros_like(state)
    for axis, length in enumerate(lengths):
        np_axis = -1-axis
        normal = (mx, my)[axis]/rho
        flux = np.stack((rho*normal, mx*normal, my*normal, (energy+pressure)*normal))
        flux[axis+1] += pressure
        speed = np.abs(normal)+np.sqrt(gamma*pressure/rho)
        shared = np.maximum(speed, np.roll(speed, -1, axis=np_axis))
        faces = .5*(flux+np.roll(flux, -1, axis=np_axis)
                    -shared*(np.roll(state[:4], -1, axis=np_axis)-state[:4]))
        out[:4] -= (faces-np.roll(faces, 1, axis=np_axis))*state.shape[np_axis]/length
    return out


def hyperbolic(state, h, lengths, gamma):
    first = state+h*hyperbolic_rhs(state, lengths, gamma)
    return .5*(state+first+h*hyperbolic_rhs(first, lengths, gamma))


def evolve(state, h, steps, *, lengths=(2*math.pi, 2*math.pi), alpha=1., gamma=1.4,
           source_only=False):
    state = state.copy()
    residuals = []
    for _ in range(steps):
        if not source_only:
            state = hyperbolic(state, h/2, lengths, gamma)
        state, residual = source(state, h, lengths=lengths, alpha=alpha)
        residuals.append(residual)
        if not source_only:
            state = hyperbolic(state, h/2, lengths, gamma)
    return state, residuals


def energy(state, lengths=(2*math.pi, 2*math.pi), alpha=1.):
    cells = (state.shape[2], state.shape[1])
    _, _, k = operators(cells, lengths)
    phi = state[4].reshape(-1)
    return math.prod(lengths)/math.prod(cells)*(state[3].sum()+phi@k@phi/(2*alpha))


def gauss(state, lengths=(2*math.pi, 2*math.pi), alpha=1.):
    cells = (state.shape[2], state.shape[1])
    _, _, k = operators(cells, lengths)
    return k@state[4].reshape(-1)-alpha*(state[0].reshape(-1)-1)
