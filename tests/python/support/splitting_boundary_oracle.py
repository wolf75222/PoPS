"""Independent dense Dirichlet matrices and exact semidiscrete evolution."""
import numpy as np


def operators(n, diffusion=.08, reaction=1.3):
    x = np.arange(1, n+1)/(n+1)
    lap = (np.diag(np.full(n, -2.))+np.diag(np.ones(n-1), 1)
           +np.diag(np.ones(n-1), -1))*(n+1)**2
    return x, diffusion*lap, -reaction*np.diag(x)


def exact(n, time):
    x, a, b = operators(n)
    eigen, vectors = np.linalg.eigh(a+b)
    w = vectors@(np.exp(time*eigen)*(vectors.T@(.2*np.sin(np.pi*x))))
    return w+np.exp(time)*(1+x)


def discrete(n, h, steps):
    x, a, b = operators(n)
    propagator = np.linalg.solve(np.eye(n)-h*a/4, np.eye(n)+h*a/4)
    reaction = np.diag(np.exp(h*np.diag(b)))
    w = .2*np.sin(np.pi*x)
    for _ in range(steps):
        w = propagator@reaction@propagator@w
    return w+np.exp(h*steps)*(1+x)


def original_residual(n, time):
    """Independent original nonhomogeneous semidiscrete RHS and lifted RHS."""
    x, a, b = operators(n)
    u = exact(n, time)
    trace = np.exp(time)*np.array([1., 2.])
    physical = (a+b)@u+(1+1.3*x)*np.exp(time)*(1+x)
    physical[0] += .08*(n+1)**2*trace[0]
    physical[-1] += .08*(n+1)**2*trace[1]
    lifted = (a+b)@(u-np.exp(time)*(1+x))+np.exp(time)*(1+x)
    return physical, lifted
