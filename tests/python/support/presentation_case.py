"""A two-block, two-model case whose owner names are parameters (N11 lot 2b).

``build_named_case`` builds the same transport problem under any choice of case, block, model and
layout names, so a test can rename every owner (reversing their alphabetical order) and compare
identities. The names of the kept kinds (quantities, terms, the Program) are fixed unless a test
passes them, which is how the C52 direction of the boundary tests changes one of them.
"""

import pops
from pops.domain import Rectangle
from pops.frames import Cartesian2D
from pops.initial import InitialCondition
from pops.layouts import Uniform
from pops.lib.initial import BindArray
from pops.lib.time import RungeKutta, RungeKuttaRoute, SSPRK2_TABLEAU
from pops.math import ddt, div
from pops.mesh import CartesianGrid, PeriodicAxes
from pops.numerics import DiscretizationPlan, FiniteVolume
from pops.numerics.reconstruction import MUSCL
from pops.numerics.reconstruction.limiters import VanLeer
from pops.numerics.riemann import ScalarUpwind
from pops.numerics.variables import Conservative
from pops.projection import ConservativeCellAverage


FIRST = {"case": "advection", "blocks": ("alpha", "beta"), "models": ("alpha_model", "beta_model")}
# Every owner renamed so that the alphabetical order of the blocks and of the models reverses.
RENAMED = {"case": "transport_pair", "blocks": ("zeta", "eta"), "models": ("zeta_model", "eta_model")}


def _transport_model(name, frame, *, speed, loss, state_name="U", rate_name="balance"):
    model = pops.Model(name, frame=frame)
    state = model.state(state_name, components=("q",))
    (q,) = state
    flux = model.flux("F", frame=frame, state=state,
                      components={frame.x: (speed * q,), frame.y: (0 * q,)},
                      waves={frame.x: (speed,), frame.y: (0,)})
    velocity = model.vector("a", frame=frame, components={frame.x: speed, frame.y: 0})
    rate = model.rate(rate_name, equation=ddt(state) == -div(flux) - loss * q ** 2)
    return model, state, flux, velocity, rate


def build_named_case(names=FIRST, *, state_name="U", rate_name="balance", cells=(8, 8),
                     speeds=(1, 2)):
    """Return ``(case, layout)`` for the two-block transport case under ``names``.

    Block ``k`` advects at ``speeds[k]`` with the loss ``0.5 / speeds[k]``.
    """
    frame = Rectangle("box", lower=(0., 0.), upper=(1., 1.)).frame(Cartesian2D())
    case = pops.Case(names["case"])
    routes = []
    for index, (block_name, model_name) in enumerate(zip(names["blocks"], names["models"])):
        model, state, flux, velocity, rate = _transport_model(
            model_name, frame, speed=speeds[index], loss=0.5 / speeds[index],
            state_name=state_name, rate_name=rate_name)
        block = case.block(block_name, model)
        plan = DiscretizationPlan()
        plan.rates.add(rate, FiniteVolume(flux=flux, variables=Conservative(state),
                                          reconstruction=MUSCL(VanLeer()),
                                          riemann=ScalarUpwind(velocity=velocity)))
        case.numerics(plan, block=block)
        case.initials.add(InitialCondition(state=block[state], value=BindArray(),
                                          projection=ConservativeCellAverage()))
        routes.append(RungeKuttaRoute(block[state], rate))
    case.program(RungeKutta(routes=tuple(routes), tableau=SSPRK2_TABLEAU))
    layout = Uniform(CartesianGrid(frame=frame, cells=cells, periodic=PeriodicAxes(frame.axes)))
    return case, layout


def resolve_named_case(names=FIRST, **options):
    case, layout = build_named_case(names, **options)
    return pops.resolve(pops.validate(case), layout=layout)
