"""Public two-owner guarded elliptic publication and physical diffusion recipe."""

from fractions import Fraction
import numpy as np
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
    CompositeHierarchySolve,
    bcs,
)
from pops.numerics import Diffusion, DiscretizationPlan
from pops.physics.diffusion import DiffusiveBoundary
from pops.time import Program, FailRun, FixedDt, every
from pops.params import RuntimeParam
from pops.layouts import AMR
from pops.mesh import CartesianGrid, PeriodicAxes
from pops.amr import (
    AMRExecution,
    AMRHierarchy,
    AMRRegrid,
    AMRTagging,
    AMRTransfer,
    Buffer,
    ConflictPolicy,
    EqualityPolicy,
    Hysteresis,
    Tag,
)
from pops.lib.amr import StateTransfer
from pops.initial import InitialCondition
from pops.projection import ConservativeCellAverage
from pops.lib.initial import BindArray


def author_case(
    *,
    solver,
    cells=(16, 12),
    dt=Fraction(1, 10000),
    guard=True,
    suffix="",
    reaction=3,
    diffusion_offset=Fraction(2, 7),
):
    """Keep the equations identical for every explicitly supplied solver descriptor."""
    frame = Rectangle("independent-elliptic-domain" + suffix, (0, 0), (2, 3)).frame(
        Cartesian2D()
    )
    donor = pops.Model("fixed-forcing" + suffix, frame=frame)
    driving = donor.state("material" + suffix, components=("material",))
    receiver = pops.Model("conducting-material" + suffix, frame=frame)
    density = receiver.state("density" + suffix, components=("density",))
    mobility = receiver.aux("mobility" + suffix)
    flux = receiver.diffusive_flux(
        "physical-conduction" + suffix,
        state=density,
        value=(diffusion_offset + mobility**2) * math.grad(density[0]),
        boundaries=tuple(
            DiffusiveBoundary(axis, side, "periodic")
            for axis in range(len(frame.axes))
            for side in ("lower", "upper")
        ),
    )
    rate = receiver.rate(
        "conservation" + suffix, equation=math.ddt(density) == math.div(flux)
    )
    case = pops.Case("independent-guarded-elliptic-diffusion" + suffix)
    left = case.block("donor" + suffix, donor)
    right = case.block("receiver" + suffix, receiver)
    methods = DiscretizationPlan()
    methods.rates.add(rate, Diffusion(flux=flux))
    case.numerics(methods, block=right)
    phi = Handle(
        "screened-unknown" + suffix,
        kind="field",
        owner=OwnerPath.model("physical-equation" + suffix),
    )
    problem = FieldProblem(
        "shifted-periodic-elliptic" + suffix,
        unknowns=(phi,),
        equations=(-math.laplacian(phi) + math.Reaction(phi, reaction) == driving[0],),
        boundaries=(
            FieldBoundary(
                phi, bcs.BoundaryCondition(bcs.AllPhysicalBoundaries(), bcs.Periodic())
            ),
        ),
    )
    field = case.field(
        problem,
        FieldDiscretization(
            method=CellCenteredSecondOrder(),
            boundaries=(),
            solver=solver,
            hierarchy_policy=CompositeHierarchySolve(),
        ),
    )
    p = Program("guarded-method" + suffix)
    d = p.state(left[driving])
    q = p.state(right[density])
    point = p.stage("consume-at-start" + suffix, c=0)
    consumed_driver = (
        p.guard(
            "donor-input-acceptance" + suffix, d.n, p.norm2(q.n) >= 0, action=FailRun()
        )
        if guard
        else d.n
    )
    solution = field.observe(
        p.solve(field, values={left[driving]: consumed_driver}, at=point).consume(
            action=FailRun()
        )
    )
    observed_phi = solution[field[phi]]
    target = right[
        receiver.module.field_handle(receiver.module.field_spaces()["fields"])
    ]
    fields = solution.publish(
        {(target, "mobility" + suffix): observed_phi}, states={right[density]: q.n}
    )
    endpoint = p.value(
        "actual-density-endpoint" + suffix,
        q.n + p.dt * rate(q.n, fields),
        at=q.next.point,
    )
    donor_endpoint = p.value("actual-donor-endpoint" + suffix, 1 * d.n, at=d.next.point)
    if guard:
        donor_endpoint = p.guard(
            "donor-commit-acceptance" + suffix,
            donor_endpoint,
            p.norm2(donor_endpoint) >= 0,
            action=FailRun(),
        )
    p.store_history(
        "actual-solved-phi" + suffix, observed_phi, depth=1, owner_block=right
    )
    p.commit(d.next, donor_endpoint)
    p.commit(q.next, endpoint)
    p.step_strategy(FixedDt(float(dt)))
    case.program(p)
    for reference in (left[driving], right[density]):
        case.initials.add(
            InitialCondition(
                state=reference, value=BindArray(), projection=ConservativeCellAverage()
            )
        )
    grid = CartesianGrid(frame=frame, cells=cells, periodic=PeriodicAxes(frame.axes))
    transfer = AMRTransfer()
    for reference in (left[driving], right[density]):
        transfer.state(reference, StateTransfer())
    threshold = case.param(RuntimeParam("tag-threshold" + suffix, default=100.0))
    layout = AMR(
        grid=grid,
        hierarchy=AMRHierarchy(max_levels=1, ratios=()),
        tagging=AMRTagging(
            rules=(
                Tag(math.ValueExpr(right[density])["density"] > case.value(threshold)),
                Buffer(cells=0),
            ),
            hysteresis=Hysteresis(0, EqualityPolicy.HOLD),
            conflict_policy=ConflictPolicy.REFINE_WINS,
        ),
        regrid=AMRRegrid(schedule=every(100, clock=p.clock)),
        transfer=transfer,
        execution=AMRExecution.synchronous(),
    )
    arrays, expected = independent_oracle(
        cells=cells, dt=dt, reaction=reaction, diffusion_offset=diffusion_offset
    )
    return (
        case,
        layout,
        {"donor" + suffix: arrays["donor"], "receiver" + suffix: arrays["receiver"]},
        expected,
    )


def independent_oracle(
    *,
    cells=(16, 12),
    dt=Fraction(1, 10000),
    forcing=Fraction(3, 5),
    reaction=3,
    diffusion_offset=Fraction(2, 7),
):
    """Independent periodic finite-volume difference equations for this one-step witness.

    Constant d makes every centered periodic discrete Laplacian of phi vanish;
    (-L_h+reaction I)phi=d therefore has the unique phi=d/reaction. The receiver uses cell
    averages of two separable cosine modes, then the explicit periodic FV stencil.
    No production lowering, face ledger, operator code or recognized model data is used.
    """
    nx, ny = cells
    dx, dy = 2 / nx, 3 / ny
    x = (np.arange(nx) + 0.5) * dx
    y = (np.arange(ny) + 0.5) * dy
    q0 = 1 + 0.08 * np.sinc(1 / nx) * np.cos(np.pi * x)[None, :]
    q0 = q0 + 0.03 * np.sinc(1 / ny) * np.cos(2 * np.pi * y / 3)[:, None]
    donor = np.full((ny, nx), float(forcing))
    phi = np.full((ny, nx), float(forcing / reaction))
    coefficient = float(diffusion_offset + (forcing / reaction) ** 2)
    laplacian = (np.roll(q0, -1, axis=1) - 2 * q0 + np.roll(q0, 1, axis=1)) / dx**2
    laplacian += (np.roll(q0, -1, axis=0) - 2 * q0 + np.roll(q0, 1, axis=0)) / dy**2
    expected = q0 + float(dt) * coefficient * laplacian
    frequency = 2 * coefficient * (dx**-2 + dy**-2)
    assert float(dt) * frequency < 1  # An explicit mathematical sufficient FE bound.
    arrays = {
        "donor": np.ascontiguousarray(donor[None]),
        "receiver": np.ascontiguousarray(q0[None]),
    }
    return arrays, {
        "donor": donor,
        "receiver": expected,
        "phi": phi,
        "dt": float(dt),
        "reaction": float(reaction),
        "diffusion_offset": float(diffusion_offset),
        "coefficient": coefficient,
        "forward_euler_frequency": frequency,
    }
