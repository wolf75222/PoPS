"""Independent dense/Fourier references for authored coupled methods."""
import numpy as np


def matrices(n):
    backward = (np.eye(n)-np.roll(np.eye(n), -1, axis=1))*n
    lap = (2*np.eye(n)-np.roll(np.eye(n), 1, axis=1)-np.roll(np.eye(n), -1, axis=1))*n*n
    return lap, backward


def block_matrix(n):
    k, up = matrices(n)
    angle = 2*np.pi*np.arange(n)/n
    a = 2*np.eye(n)+.015*k+.15*up
    d = 1.5*np.eye(n)+.03*k+.04*up
    b = np.diag(.2*(1+.1*np.cos(angle)))
    c = np.diag(-.3*(1+.2*np.sin(angle)))
    return np.block([[a, b], [c, d]])


def block_solution(n, steps):
    angle = 2*np.pi*np.arange(n)/n
    answer = np.concatenate((1+.2*np.cos(angle), .3+.15*np.sin(angle)))
    matrix = block_matrix(n)
    for _ in range(steps):
        answer = np.linalg.solve(matrix, answer)
    return answer.reshape(2, n)


def relation_data(n):
    angle = 2*np.pi*np.arange(n)/n
    return 2+.1*np.sin(angle), 1+.2*np.cos(angle)


def selected_root(n, mode, branch):
    frozen, forcing = relation_data(n)
    k, up = matrices(n)
    spatial = .002*k+.01*up
    if mode == "frozen":
        return np.linalg.solve(np.diag(frozen)+spatial, forcing)
    x = branch*np.sqrt(forcing)
    for _ in range(30):
        r = x*x+spatial@x-forcing
        if np.max(abs(r)) < 2.e-14:
            return x
        x -= np.linalg.solve(np.diag(2*x)+spatial, r)
        assert np.all(branch*x > 0)
    raise AssertionError("dense reference Newton did not converge")


def imex_reference(n, dt, steps):
    k, up = matrices(n)
    explicit = -.2*np.eye(n)-.3*up
    implicit = -.05*k
    gamma = 1-1/np.sqrt(2)
    delta = 1-1/(2*gamma)
    left = np.eye(n)-dt*gamma*implicit
    value = 1+.2*np.cos(2*np.pi*np.arange(n)/n)
    for _ in range(steps):
        first = np.linalg.solve(left, value+dt*gamma*explicit@value)
        right = value+dt*delta*explicit@value+dt*(1-delta)*explicit@first+dt*(1-gamma)*implicit@first
        value = np.linalg.solve(left, right)
    return value


def imex_exact(n, time):
    angle = 2*np.pi*np.arange(n)/n
    eigen = -.2-.3*n*(1-np.exp(-1j*angle))-.05*4*n*n*np.sin(angle/2)**2
    initial = 1+.2*np.cos(angle)
    return np.fft.ifft(np.exp(time*eigen)*np.fft.fft(initial)).real


def directed_rates(values, circulation):
    n = len(values)
    out = np.zeros((n, n))  # [receiver, donor]
    for donor in range(n):
        out[(donor+1)%n, donor] += .3+.03*donor
        out[(donor-1)%n, donor] += .15+.02*((donor-1)%n)
        for receiver in range(n):
            if receiver != donor:
                out[receiver, donor] += circulation*values[receiver]
    return out


def generator(rates):
    return rates-np.diag(rates.sum(axis=0))


def patankar_step(old, h, circulation):
    q1 = directed_rates(old, circulation)
    stage = np.linalg.solve(np.eye(len(old))-h*generator(q1), old)
    q2 = directed_rates(stage, circulation)
    production = .5*(q1*old[None, :]+q2*stage[None, :])
    assert np.all(stage > 0), "selected reference network is strongly connected"
    weighted = production/stage[None, :]
    return np.linalg.solve(np.eye(len(old))-h*generator(weighted), old)


def patankar_reference(n, h, steps, circulation=0., empty=False):
    values = 1+.2*np.arange(n)
    if empty:
        values[0] = 0.
    for _ in range(steps):
        values = patankar_step(values, h, circulation)
    return values


def network_exact(n, time, empty=False):
    initial = 1+.2*np.arange(n)
    if empty:
        initial[0] = 0.
    # The explicitly symmetric nonlinear process pairs cancel for every state.
    matrix = generator(directed_rates(initial, 0.))
    eigen, vectors = np.linalg.eig(matrix)
    return (vectors@(np.exp(time*eigen)*np.linalg.solve(vectors, initial))).real
