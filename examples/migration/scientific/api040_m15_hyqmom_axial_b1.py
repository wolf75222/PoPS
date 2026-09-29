"""M15 bounded variant: five one-velocity moments with Appendix B.1 S50.

This is the order-four axial marginal of the available fifteen-moment B.1
closure. It does not claim the unspecified Fox--Laurent recurrence at other
orders. The physical flux is an ordinary Python expression body passed to the
generic Model/FiniteVolume/Program construction; no Gaussian fifth moment is
substituted for the nonlinear B.1 relation.
"""
from __future__ import annotations

import pops
from pops.domain import CartesianDomain
from pops.frames import Cartesian1D
from pops.layouts import Uniform
from pops.lib.time import ForwardEuler
from pops.math import ddt, div
from pops.mesh import CartesianGrid, PeriodicAxes
from pops.moments import hyqmom_b1_axial_flux
from pops.numerics import DiscretizationPlan, reconstruction, riemann, variables
from pops.numerics.spatial import FiniteVolume
from pops.time import AdaptiveCFL


def build_case(cells: int = 16):
    """Author a periodic order-four axial B.1 test with exact Jacobian speeds."""
    if type(cells) is not int or cells < 4:
        raise ValueError("M15 axial cells must be an integer >= 4")
    frame = CartesianDomain("axial_b1_interval", (0.,), (1.,)).frame(Cartesian1D())
    model = pops.Model("hyqmom_b1_axial_order4", frame=frame)
    state = model.state("M", components=("M0", "M1", "M2", "M3", "M4"))
    # The B.1 formula divides by density and variance. Its admissible truncated
    # moment domain is not implied by finite fluxes or real characteristic roots.
    rho, m1, m2, m3, m4 = state
    hankel2 = rho * m2 - m1 * m1
    hankel3 = (rho * (m2 * m4 - m3 * m3)
               - m1 * (m1 * m4 - m2 * m3) + m2 * (m1 * m3 - m2 * m2))
    model.primitive_state(*state, conservative=tuple(state))
    model.recovery_admissibility(M0=rho > 0, M2=hankel2 > 0, M4=hankel3 >= 0)
    flux = model.flux(
        "axial_b1_transport", frame=frame, state=state,
        components={frame.axes[0]: hyqmom_b1_axial_flux(state)},
    )
    model.wave_speeds_from_jacobian()
    rate = model.rate("moment_balance", equation=ddt(state) == -div(flux))
    case = pops.Case("M15_axial_B1_order4")
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
        frame=frame, cells=(cells,), periodic=PeriodicAxes(frame.axes),
    ))
    return case, layout, state


__all__ = ["build_case"]
