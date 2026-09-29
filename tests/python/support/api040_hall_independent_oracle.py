"""Independent linear Hall oracle; NumPy only, no PoPS authoring or codegen.

Normative equation: w_t = (eta_ohmic I + eta_hall J) w_xx,
J = [[0,-1],[1,0]], on [0,2*pi). The reversible part is separate
from optional nonnegative physical Ohmic dissipation. Real rows are (w1,w2).
"""
import numpy as np

WAVENUMBER = 2
ETA_HALL = .3
FINAL_TIME = .2
INITIAL_VECTOR = np.array((1., .2))


def cell_averages(cells, *, wavenumber=WAVENUMBER):
    h = 2*np.pi/cells
    centers = (np.arange(cells)+.5)*h
    # Analytic integral, evaluated without subtracting nearby sine endpoints.
    average = np.cos(wavenumber*centers)*np.sinc(wavenumber/cells)
    return INITIAL_VECTOR[:, None]*average


def laplacian_symbol(cells, *, wavenumber=WAVENUMBER):
    h = 2*np.pi/cells
    return 4*np.sin(wavenumber*h/2)**2/h**2


def exact_amplitude(time=FINAL_TIME, *, eta_hall=ETA_HALL, eta_ohmic=0., symbol=None):
    if eta_ohmic < 0:
        raise ValueError("Ohmic dissipation must be nonnegative")
    k2 = WAVENUMBER**2 if symbol is None else symbol
    return np.exp(-time*k2*(eta_ohmic + 1j*eta_hall))


def apply_amplitude(initial, amplitude):
    result = amplitude*(initial[0]+1j*initial[1])
    return np.stack((result.real, result.imag))


def periodic_rhs(state, *, eta_hall=ETA_HALL, eta_ohmic=0.):
    h = 2*np.pi/state.shape[1]
    laplace = (np.roll(state, -1, axis=1)-2*state+np.roll(state, 1, axis=1))/h**2
    return eta_ohmic*laplace + eta_hall*np.stack((-laplace[1], laplace[0]))


def theta_amplitude(cells, steps, *, theta=.5, eta_hall=ETA_HALL, eta_ohmic=0.):
    dt = FINAL_TIME/steps
    z = -dt*laplacian_symbol(cells)*(eta_ohmic + 1j*eta_hall)
    return ((1+(1-theta)*z)/(1-theta*z))**steps


def ssprk2_amplitude(cells, steps, *, eta_hall=ETA_HALL):
    z = -1j*(FINAL_TIME/steps)*laplacian_symbol(cells)*eta_hall
    return (1+z+z*z/2)**steps


def observed_amplitude(initial, final):
    a = initial[0]+1j*initial[1]
    b = final[0]+1j*final[1]
    return np.vdot(a, b)/np.vdot(a, a)
