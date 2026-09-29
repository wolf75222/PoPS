"""M16: full Appendix B.1 HyQMOM15 and its correlated oblique obstruction.

The source model uses the public 2V moment generator. The diagnostic
differentiate its actual public symbolic flux body by complex step at a
strictly realizable correlated Gaussian. The exact polynomial is the pinned
rational witness from HYQMOM15_LIMITATION.md. No real-part projection, covariance
floor, Gaussian fifth-moment replacement, or invented speed bound is applied.
"""
from __future__ import annotations

import numpy as np
import pops
from pops._ir.expr import Var
from pops.domain import Rectangle
from pops.frames import Cartesian2D
from pops.layouts import Uniform
from pops.lib.time import ForwardEuler
from pops.mesh import CartesianGrid, PeriodicAxes
from pops.moments import (CartesianVelocityMoments, HyQMOM15Closure,
                          moment_flux_expressions, moment_names)
from pops.numerics import DiscretizationPlan, reconstruction, riemann, variables
from pops.numerics.spatial import FiniteVolume
from pops.time import AdaptiveCFL


# q-outer order through total degree four, as in the pinned source audit.
CORRELATED_GAUSSIAN = np.array(
    [1, 0, 1, 0, 3, 0, .5, 0, 1.5, 1, 0, 1.5, 0, 1.5, 3], dtype=float,
)
EXPECTED_POLYNOMIAL = np.array(
    [1, 0, -67 / 2, 0, 5641 / 16, 0, -1422, 0, 9747 / 4,
     0, -1593, 0, 4131 / 16, 0, 0, 0], dtype=float,
)


class _ExpressionAuthor:
    def primitive(self, _name, expression):
        return expression


def oblique_witness():
    """Differentiate the public B.1 flux at the exact correlated Gaussian.

    Complex step is a numerical cross-check of the rational characteristic
    polynomial, not a replacement for its exact discriminant argument.
    """
    names = tuple(moment_names(4))
    variables_ = tuple(Var(name, "cons") for name in names)
    flux = moment_flux_expressions(
        _ExpressionAuthor(), variables_, 4, HyQMOM15Closure(), robust=False,
    )
    h = 1e-24
    jx, jy = np.empty((15, 15)), np.empty((15, 15))
    for column in range(15):
        perturbed = CORRELATED_GAUSSIAN.astype(complex)
        perturbed[column] += 1j * h
        environment = dict(zip(names, perturbed, strict=True))
        jx[:, column] = np.imag([value.eval(environment) for value in flux.x]) / h
        jy[:, column] = np.imag([value.eval(environment) for value in flux.y]) / h
    a = jx - jy
    # Discriminant of 4*z^3 - 91*z^2 + 384*z - 459, z=lambda^2.
    cubic = (4, -91, 384, -459)
    aa, bb, cc, dd = cubic
    discriminant = (bb * bb * cc * cc - 4 * aa * cc**3 - 4 * bb**3 * dd
                    - 27 * aa * aa * dd * dd + 18 * aa * bb * cc * dd)
    return {
        "jx": jx, "jy": jy, "oblique": a,
        "polynomial": np.poly(a), "expected_polynomial": EXPECTED_POLYNOMIAL.copy(),
        "discriminant": discriminant,
        "max_imag_x": float(np.max(abs(np.linalg.eigvals(jx).imag))),
        "max_imag_y": float(np.max(abs(np.linalg.eigvals(jy).imag))),
        "max_imag_unit_oblique": float(np.max(abs(np.linalg.eigvals(a / np.sqrt(2)).imag))),
    }


def build_case(cells: int = 8):
    """Author the full fifteen-equation B.1 transport with true Jacobian speeds."""
    if type(cells) is not int or cells < 4:
        raise ValueError("M16 cells must be an integer >= 4")
    frame = Rectangle("b1_oblique_square", (0., 0.), (1., 1.)).frame(Cartesian2D())
    model = CartesianVelocityMoments(
        4, closure=HyQMOM15Closure(), robust=False, exact_speeds=True,
    ).build("hyqmom15_B1", frame=frame)
    state, flux, rate = (model.states["U"], model.fluxes["transport"],
                         model.operators["transport"])
    case = pops.Case("M16_HyQMOM15_B1_oblique")
    block = case.block("moments", model)
    plan = DiscretizationPlan()
    plan.rates.add(rate, FiniteVolume(
        flux=flux, variables=variables.Conservative(state),
        reconstruction=reconstruction.FirstOrder(), riemann=riemann.HLL(),
    ))
    case.numerics(plan, block=block)
    program = ForwardEuler(block[state], rate=rate)
    program.step_strategy(AdaptiveCFL(cfl=0.1))
    case.program(program)
    layout = Uniform(CartesianGrid(
        frame=frame, cells=(cells, cells), periodic=PeriodicAxes(frame.axes),
    ))
    return case, layout, state


__all__ = ["CORRELATED_GAUSSIAN", "EXPECTED_POLYNOMIAL", "build_case", "oblique_witness"]
