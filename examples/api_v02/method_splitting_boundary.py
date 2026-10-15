"""Extended Strang on a genuine Dirichlet spatial domain, in graph algebra.

u_t = kappa*u_xx - lambda*x*u + (1+lambda*x)*exp(t)*(1+x), 0<x<1,
u(t,0)=exp(t), u(t,1)=2*exp(t). Writing u=w+b(t,x), b=exp(t)*(1+x),
gives w_t=kappa*w_xx-lambda*x*w with homogeneous Dirichlet traces.
Both split subproblems preserve their declared trace domain. A advances the
physical clock; B freezes it. The sum of algorithmic durations is 2h, while
the reached physical duration is h. No boundary provider or solve is physical
code in the engine: traces, lifting, matrix actions, and GMRES are authored here.
"""
from __future__ import annotations

import math
from pops import math as pm

from .methods_coupled_nonsymmetric import _context, amap, at, axpy, solve_linear


def dirichlet_laplacian(p, values):
    """N interior unknowns; the two prescribed homogeneous traces are separate."""
    n = p.size(p.domain(values))
    scale = p.float64(n+1)**2
    def cell(b, i):
        j = b.index_value(i)
        left = b.branch(j == 0, lambda c: c.float64(0),
                        lambda c: at(c, values, j-1))
        right = b.branch(j+1 == n, lambda c: c.float64(0),
                         lambda c: at(c, values, j+1))
        return scale*(left-2*b.at(values, i)+right)
    return amap(p, values, cell)


def lift(p, template, time):
    n = p.size(p.domain(template))
    return amap(p, template, lambda b, i:
        pm.exp(time)*(1+b.float64(b.index_value(i)+1)/b.float64(n+1)))


def lifted_strang(p, old, time, h, *, capacity, diffusion=.08, reaction=1.3):
    """CN for each diffusion half-flow; exact local reaction at frozen time.

    CN is a second-order approximation to the diffusion subflow. The two
    spatial operators do not commute, so this is a nontrivial composition.
    """
    w = axpy(p, old, -1, lift(p, old, time))
    def diffusion_half(b, value):
        rhs = axpy(b, value, h*diffusion/4, dirichlet_laplacian(b, value))
        return solve_linear(b, rhs,
            lambda c, trial: axpy(c, trial, -h*diffusion/4, dirichlet_laplacian(c, trial)),
            capacity=capacity)
    first, r0, _ = diffusion_half(p, w)
    midpoint = time+h/2
    n = p.size(p.domain(w))
    middle = amap(p, first, lambda b, i: b.at(first, i)*pm.exp(
        -reaction*h*b.float64(b.index_value(i)+1)/b.float64(n+1)))
    final, r1, _ = diffusion_half(p, middle)
    reached = midpoint+h/2
    return axpy(p, final, 1, lift(p, final, reached)), reached, r0+r1


def build_t03_dirichlet(*, count=7, dt=.05, steps=4):
    """Spatially varying Dirichlet heat/reaction, with a boundary-domain witness."""
    def initialize(p, domain):
        u = p.map(domain, lambda b, i:
            1+b.float64(b.index_value(i)+1)/(count+1)
            +.2*pm.sin(math.pi*b.float64(b.index_value(i)+1)/(count+1)))
        return p.product(u=u, physical_time=p.float64(0), midpoint=p.float64(0),
            left_trace=p.float64(1), right_trace=p.float64(2),
            mid_left=p.float64(1), mid_right=p.float64(2),
            algorithmic_duration=p.float64(0), reached_duration=p.float64(0),
            naive_B_left_residual=p.float64(1), naive_B_right_residual=p.float64(2))
    def step(p, old, h):
        u, time, residual = lifted_strang(p, old["u"], old["physical_time"], h,
                                          capacity=count)
        midpoint = old["physical_time"]+h/2
        # On the boundary the naive unsplit reaction/forcing vector field is
        # b, whereas a frozen-time trace requires zero tangent. This nonzero
        # witness is derived from the original PDE, not a failed solve proxy.
        mid_left, mid_right = pm.exp(midpoint), 2*pm.exp(midpoint)
        values = p.product(u=u, physical_time=time, midpoint=midpoint,
            left_trace=pm.exp(time), right_trace=2*pm.exp(time),
            mid_left=mid_left, mid_right=mid_right,
            algorithmic_duration=2*h, reached_duration=h,
            naive_B_left_residual=mid_left, naive_B_right_residual=mid_right)
        return values, residual
    return _context("T03_non_autonomous_Dirichlet_Strang", count, dt, steps, initialize, step)
