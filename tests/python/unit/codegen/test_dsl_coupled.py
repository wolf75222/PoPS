"""Shared Euler--Poisson DSL model for final native-package integration tests.

The historical prototype-loader scenario that lived in this file was retired. The
builder remains the single fixture used by the production
native-loader, AMR, SSPRK3, and WENO5 acceptance tests.
"""
from pops.math import sqrt
from pops.physics._model import HyperbolicModel

GAMMA = 1.4
from tests.python.support.requirements import repo_include
INCLUDE = repo_include()


def build_euler_poisson():
    """euler_poisson en formules : Euler compressible + force de gravite (g = -grad phi) + couplage
    self-consistant f = -(rho - 1) (GravityCoupling sign=-1, 4piG=1, rho0=1)."""
    e = HyperbolicModel("euler_poisson")
    rho, rhou, rhov, E = e.conservative_vars(
        "rho", "rho_u", "rho_v", "E",
        roles=["Density", "MomentumX", "MomentumY", "Energy"])
    u = e.primitive("u", rhou / rho)
    v = e.primitive("v", rhov / rho)
    p = e.primitive("p", (GAMMA - 1.0) * (E - 0.5 * rho * (u * u + v * v)))
    H = (E + p) / rho
    c = sqrt(GAMMA * p / rho)
    e.set_flux(x=[rhou, rhou * u + p, rhou * v, rho * H * u],
               y=[rhov, rhov * u, rhov * v + p, rho * H * v])
    e.set_eigenvalues(x=[u - c, u, u + c], y=[v - c, v, v + c])
    e.set_primitive_state(rho, u, v, p)
    e.set_conservative_from([rho, rho * u, rho * v, p / (GAMMA - 1.0) + 0.5 * rho * (u * u + v * v)])
    # source de gravite : g = -grad phi ; S = (0, rho gx, rho gy, rho_u gx + rho_v gy) = GravityForce
    gx = e.aux("grad_x")
    gy = e.aux("grad_y")
    e.set_source([0.0, -rho * gx, -rho * gy, -(rhou * gx + rhov * gy)])
    # couplage : f = sign 4piG (rho - rho0), sign=-1 4piG=1 rho0=1 (GravityCoupling)
    e.set_elliptic_rhs(-1.0 * (rho - 1.0))
    # ADC-590 : hllc/roe generiques exigent la capability EMISE (plus de fallback Euler implicite) ;
    # les tests de parite aval (production/aot) exercent riemann='hllc'/'roe' sur ce modele.
    e.enable_hllc()
    e.enable_roe()
    return e
