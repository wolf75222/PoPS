"""Independently authored frozen-donor Field publication recipe."""

from fractions import Fraction
import pops
from pops import math
from pops.domain import Rectangle
from pops.frames import Cartesian2D
from pops.model import Handle, OwnerPath
from pops.fields import (
    FieldProblem,
    FieldDiscretization,
    CellCenteredSecondOrder,
    FieldBoundary,
    bcs,
)
from pops.solvers import CG
from pops.time import Program, FailRun, FixedDt
from pops.numerics import Diffusion, DiscretizationPlan


def author_frozen_coefficient_case(
    *, guard_driver=False, guard_field_input=False, suffix=""
):
    frame = Rectangle("audit-domain" + suffix, (0, 0), (2, 3)).frame(Cartesian2D())
    donor = pops.Model("forcing-library" + suffix, frame=frame)
    driver = donor.state("unchanged-material" + suffix, components=("material",))
    receiver = pops.Model("conducting-library" + suffix, frame=frame)
    state = receiver.state("density" + suffix, components=("density",))
    coefficient = receiver.aux("mobility" + suffix)
    flux = receiver.diffusive_flux(
        "constitutive" + suffix,
        state=state,
        value=(Fraction(2, 7) + coefficient**2) * math.grad(state[0]),
    )
    rate = receiver.rate(
        "conservation" + suffix, equation=math.ddt(state) == math.div(flux)
    )
    case = pops.Case("independent-mathematical-composition" + suffix)
    left = case.block("fixed-owner" + suffix, donor)
    right = case.block("evolving-owner" + suffix, receiver)
    plan = DiscretizationPlan()
    plan.rates.add(rate, Diffusion(flux=flux))
    case.numerics(plan, block=right)
    phi = Handle(
        "elliptic-unknown" + suffix,
        kind="field",
        owner=OwnerPath.model("equation-library" + suffix),
    )
    problem = FieldProblem(
        "screened-equation" + suffix,
        unknowns=(phi,),
        equations=(-math.laplacian(phi) + math.Reaction(phi, 3) == driver[0],),
        boundaries=(
            FieldBoundary(
                phi, bcs.BoundaryCondition(bcs.AllPhysicalBoundaries(), bcs.Periodic())
            ),
        ),
    )
    field = case.field(
        problem,
        FieldDiscretization(
            method=CellCenteredSecondOrder(), boundaries=(), solver=CG(max_iter=64)
        ),
    )
    p = Program("two-owner-method" + suffix)
    d = p.state(left[driver])
    u = p.state(right[state])
    point = p.stage("field-read" + suffix, c=0)
    consumed_driver = (
        p.guard(
            "donor-value-with-independent-condition" + suffix,
            d.n,
            p.norm2(u.n) >= 0,
            action=FailRun(),
        )
        if guard_field_input
        else d.n
    )
    outcome = p.solve(field, values={left[driver]: consumed_driver}, at=point)
    solution = field.observe(outcome.consume(action=FailRun()))
    target = right[
        receiver.module.field_handle(receiver.module.field_spaces()["fields"])
    ]
    published = solution.publish(
        {(target, "mobility" + suffix): solution[field[phi]]},
        states={right[state]: u.n},
    )
    increment = rate(u.n, published)
    next_u = p.value(
        "actual-accepted-density" + suffix, u.n + p.dt * increment, at=u.next.point
    )
    next_d = p.value("exact-unchanged-forcing" + suffix, 1 * d.n, at=d.next.point)
    if guard_driver:
        next_d = p.guard(
            "finite-forcing" + suffix, next_d, p.norm2(next_d) >= 0, action=FailRun()
        )
    p.commit(d.next, next_d)
    p.commit(u.next, next_u)
    p.step_strategy(FixedDt(1e-4))
    case.program(p)
    return p, case
