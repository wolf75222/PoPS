"""Independent 1D axial B.1 five-moment FV oracle for the M15 bounded variant.

Only NumPy and Python arithmetic are used. The source's expression builder,
native flux, compiler, and runtime are never imported. This is not a
Fox--Laurent multiorder hierarchy or an exact PDE solution.
"""

from __future__ import annotations

from math import comb, factorial, pi

import numpy as np


COMPONENTS = 5
FINAL_TIME = .02
INITIAL_ATOL = 2.e-14
STATE_ATOL = 3.e-11
INTEGRAL_ATOL = 2.e-12
TIME_ATOL = 2.e-14
PERMUTATION_ATOL = 3.e-12
NON_GAUSSIAN_GAP = 1.e-2


def normal_raw(order: int, mean: float, variance: float) -> float:
    return sum(comb(order, 2*j)*factorial(2*j)/(2**j*factorial(j))
               *variance**j*mean**(order-2*j) for j in range(order//2+1))


def exact_cell_means(cells: int) -> np.ndarray:
    """Integrate the authored positive two-Gaussian mixture over each cell."""
    if type(cells) is not int or cells < 4:
        raise ValueError("cells must be an integer >= 4")
    centers = (np.arange(cells, dtype=float)+.5)/cells
    sinc = np.sinc(1/cells)
    weight_left = .6+.08*sinc*np.cos(2*pi*centers)
    weight_right = .4+.06*sinc*np.sin(2*pi*centers)
    left = np.array([normal_raw(order,-.4,.3) for order in range(COMPONENTS)])
    right = np.array([normal_raw(order,.7,.2) for order in range(COMPONENTS)])
    return left[:,None]*weight_left[None,:]+right[:,None]*weight_right[None,:]


def hankel_minimum(raw: np.ndarray) -> float:
    """Smallest eigenvalue of the order-two truncated Hamburger Gram matrix."""
    values = np.asarray(raw,dtype=float)
    if values.ndim != 2 or values.shape[0] != COMPONENTS:
        raise ValueError("expected [M0..M4, cells]")
    minimum = float("inf")
    for cell in range(values.shape[1]):
        gram = np.array([[values[i+j,cell] for j in range(3)] for i in range(3)])
        minimum = min(minimum,float(np.linalg.eigvalsh(gram)[0]))
    return minimum


def b1_flux(raw: np.ndarray) -> np.ndarray:
    """B.1 flux via centered-moment algebra, independent of the PoPS IR formula."""
    density,m1,m2,m3,m4 = np.asarray(raw)
    mean=m1/density
    c2=m2-2*mean*m1+mean*mean*density
    c3=m3-3*mean*m2+3*mean*mean*m1-mean**3*density
    c4=m4-4*mean*m3+6*mean*mean*m2-4*mean**3*m1+mean**4*density
    # Algebraically equal to rho*sigma**5*S30*(5*S40-3*S30**2-1)/2.
    c5=.5*c3*(5*c4/c2-3*c3*c3/(c2*c2)-c2/density)
    m5=c5+5*mean*c4+10*mean**2*c3+10*mean**3*c2+density*mean**5
    return np.asarray((m1,m2,m3,m4,m5))


def gaussian_fifth(raw: np.ndarray) -> np.ndarray:
    density,m1,m2,_,_ = np.asarray(raw)
    mean=m1/density
    variance=m2/density-mean*mean
    return density*(mean**5+10*mean**3*variance+15*mean*variance**2)


def signed_speeds(raw: np.ndarray) -> tuple[float,float]:
    """Independent complex-step Jacobian and nonsymmetric characteristic roots."""
    point=np.asarray(raw,dtype=float)
    if point.shape != (COMPONENTS,):
        raise ValueError("signed speeds require one five-moment state")
    step=1.e-25
    jac=np.empty((COMPONENTS,COMPONENTS),dtype=float)
    for column in range(COMPONENTS):
        perturbed=point.astype(complex)
        perturbed[column]+=1j*step
        jac[:,column]=np.imag(b1_flux(perturbed))/step
    eigenvalues=np.linalg.eigvals(jac)
    if max(abs(eigenvalues.imag))>1.e-9:
        raise ValueError("axial B.1 state has non-real characteristic speeds")
    return float(min(eigenvalues.real)),float(max(eigenvalues.real))


def hll_face(left: np.ndarray,right: np.ndarray) -> np.ndarray:
    left_min,left_max=signed_speeds(left)
    right_min,right_max=signed_speeds(right)
    low,high=min(left_min,right_min),max(left_max,right_max)
    fleft,fright=b1_flux(left),b1_flux(right)
    if low>=0:
        return fleft
    if high<=0:
        return fright
    return (high*fleft-low*fright+low*high*(right-left))/(high-low)


def forward_euler(initial: np.ndarray,steps: int) -> np.ndarray:
    """Periodic first-order HLL finite-volume trajectory with dt=1/(100*N)."""
    state=np.asarray(initial,dtype=float).copy()
    if state.ndim != 2 or state.shape[0] != COMPONENTS:
        raise ValueError("expected [five moments, cells]")
    cells=state.shape[1]
    if type(steps) is not int or steps<0:
        raise ValueError("steps must be a non-negative integer")
    dt=1/(100*cells)
    for _ in range(steps):
        faces=np.stack([hll_face(state[:,i],state[:,(i+1)%cells])
                        for i in range(cells)],axis=1)
        state+=dt*cells*(np.roll(faces,1,axis=1)-faces)
        if not np.isfinite(state).all() or hankel_minimum(state)<=0:
            raise ValueError("B.1 trajectory left the realizable interior")
    return state


def planned_steps(cells: int) -> int:
    steps=2*cells
    assert abs(steps/(100*cells)-FINAL_TIME)<1.e-15
    return steps
