"""Two screened potentials drive a three-stage receiver; foreign donor stays frozen.

Physics and time composition use public Python declarations only. No compiler/core override.
"""

from dataclasses import dataclass
from fractions import Fraction as F

import pops
from pops import math
from pops.domain import Rectangle
from pops.fields import FieldBoundary, FieldDiscretization, FieldProblem, bcs
from pops.fields.methods import CellCenteredSecondOrder
from pops.frames import Cartesian2D
from pops.layouts import Uniform
from pops.mesh import CartesianGrid, PeriodicAxes
from pops.model import Handle, OwnerPath
from pops.numerics import Diffusion, DiscretizationPlan, StateStorage
from pops.solvers import CG
from pops.time import FailRun, FixedDt

CELLS = (16, 12)
LENGTHS = (F(2), F(3))
DT = F(1, 10000)
DIFFUSIVITY = F(3, 20)
DECAY = F(4, 5)
PHI_WEIGHT = F(1, 5)
PSI_WEIGHT = F(2, 5)
SCREENINGS = (F(3), F(5))
FOREIGN_PSI_WEIGHT = F(-3, 4)
INITIAL_MEANS = {"receiver": F(2), "donor": F(3, 5)}
INITIAL_AMPLITUDES = {"receiver": F(1, 8), "donor": F(7, 100)}


@dataclass(frozen=True)
class StageFieldsCase:
    case: object
    layout: object
    program: object
    receiver: object
    donor: object
    field: object
    unknowns: tuple
    states: tuple
    publications: tuple
    rhs: tuple
    observed_fields: tuple


def author_case(*, epsilon=F(0), beta=F(0), labels=None):
    """Compose a linear coefficient change or a cubic cell-mean source.

    The unchanged periodic Fields and SSPRK3 stages use public PoPS operations.
    This witness is restricted to epsilon in [0, 1/20], beta in [0, 1/20].
    """
    import math as scalar_math

    for name, value in (("epsilon", epsilon), ("beta", beta)):
        if not scalar_math.isfinite(float(value)) or not 0 <= value <= F(1, 20):
            raise ValueError(name + " must be finite and in [0, 1/20]")
    labels = {} if labels is None else dict(labels)

    def label(role, default):
        return labels.get(role, default)

    frame = Rectangle(
        label("domain", "screened-stage-domain"), lower=(0.0, 0.0), upper=(2.0, 3.0)
    ).frame(Cartesian2D())
    donor_model = pops.Model(label("donor_model", "frozen-reservoir"), frame=frame)
    d = donor_model.state(label("donor_state", "D"), components=("d",))
    zero = donor_model.source("zero-balance", on=d, value=(0 * d[0],))
    frozen_rate = donor_model.rate("frozen-rate", equation=math.ddt(d) == zero)
    receiver_model = pops.Model(label("receiver_model", "two-potential-receiver"), frame=frame)
    q = receiver_model.state(label("receiver_state", "Q"), components=("q",))
    phi_read = receiver_model.aux(label("phi_input", "screened_first"))
    psi_read = receiver_model.aux(label("psi_input", "screened_second"))
    flux = receiver_model.diffusive_flux(
        "diffusive-transport", state=q, value=math.CoeffGradient(q[0], float(DIFFUSIVITY))
    )
    forcing = (
        float(PHI_WEIGHT + epsilon) * phi_read
        + float(PSI_WEIGHT - epsilon) * psi_read
        - float(DECAY) * q[0]
    )
    # The finite-volume convention is (stored cell mean)**3, not mean(q**3).
    if beta:
        forcing = forcing - float(beta) * q[0] * q[0] * q[0]
    source = receiver_model.source("two-potential-forcing", on=q, value=(forcing,))
    rate = receiver_model.rate("coupled-balance", equation=math.ddt(q) == math.div(flux) + source)
    case = pops.Case(label("case", "stage-dependent-two-screened-fields"))
    donor = case.block(label("donor_block", "reservoir"), donor_model)
    receiver = case.block(label("receiver_block", "receiver"), receiver_model)
    dn = DiscretizationPlan()
    dn.rates.add(frozen_rate, StateStorage())
    case.numerics(dn, block=donor)
    qn = DiscretizationPlan()
    qn.rates.add(rate, Diffusion(flux=flux))
    case.numerics(qn, block=receiver)
    owner = OwnerPath.model(label("field_owner", "joint-screened-physics"))
    phi = Handle(label("phi", "phi"), kind="field", owner=owner)
    psi = Handle(label("psi", "psi"), kind="field", owner=owner)
    problem = FieldProblem(
        label("field_problem", "two-screened-potentials"),
        unknowns=(phi, psi),
        equations=(
            -math.laplacian(phi) + math.Reaction(phi, float(SCREENINGS[0])) == q[0] + d[0],
            -math.laplacian(psi) + math.Reaction(psi, float(SCREENINGS[1]))
            == q[0] + float(FOREIGN_PSI_WEIGHT) * d[0],
        ),
        boundaries=tuple(
            FieldBoundary(v, bcs.BoundaryCondition(bcs.AllPhysicalBoundaries(), bcs.Periodic()))
            for v in (phi, psi)
        ),
    )
    field = case.field(
        problem,
        FieldDiscretization(
            method=CellCenteredSecondOrder(),
            boundaries=(),
            solver=CG(max_iter=2000, rel_tol=1e-12, abs_tol=1e-14),
        ),
    )

    program = pops.Program(label("method", "explicit-SSPRK3-two-fields"))
    current = program.state(receiver[q])
    foreign = program.state(donor[d])
    carrier = receiver[
        receiver_model.module.field_handle(receiver_model.module.field_spaces()["fields"])
    ]
    publications = []
    rates = []
    observed_fields = []

    def stage_rate(value):
        solution = field.observe(
            program.solve(
                field, values={receiver[q]: value, donor[d]: foreign.n}, at=value.point
            ).consume(action=FailRun())
        )
        observed = (solution[field[phi]], solution[field[psi]])
        stage = len(publications)
        for role, sample in zip(("phi", "psi"), observed, strict=True):
            program.store_history("observed-stage-%d-%s" % (stage, role), sample, depth=1)
        context = solution.publish(
            {
                (carrier, label("phi_input", "screened_first")): observed[0],
                (carrier, label("psi_input", "screened_second")): observed[1],
            },
            states={receiver[q]: value},
        )
        rhs = rate(value, context)
        publications.append(context)
        rates.append(rhs)
        observed_fields.append(observed)
        return rhs

    first = stage_rate(current.n)
    stage1 = program.value(
        "full-step-stage", current.n + program.dt * first, at=program.stage("SSP-second", c=1)
    )
    second = stage_rate(stage1)
    stage2 = program.value(
        "half-step-stage",
        F(3, 4) * current.n + F(1, 4) * stage1 + F(1, 4) * program.dt * second,
        at=program.stage("SSP-third", c=F(1, 2)),
    )
    third = stage_rate(stage2)
    accepted = program.value(
        "accepted-SSP3",
        F(1, 3) * current.n + F(2, 3) * stage2 + F(2, 3) * program.dt * third,
        at=current.next.point,
    )
    donor_accepted = program.value("accepted-frozen-donor", 1 * foreign.n, at=foreign.next.point)
    program.store_history("receiver-stage-0", current.n, depth=1)
    program.store_history("receiver-stage-1", stage1, depth=1)
    program.store_history("receiver-stage-2", stage2, depth=1)
    program.commit_many({current.next: accepted, foreign.next: donor_accepted})
    # The field histories record the actual consumed Fields at their own stage point.
    # They are diagnostics and do not change the accepted equations.
    program.step_strategy(FixedDt(float(DT)))
    case.program(program)
    layout = Uniform(CartesianGrid(frame=frame, cells=CELLS, periodic=PeriodicAxes(frame.axes)))
    return StageFieldsCase(
        case,
        layout,
        program,
        receiver[q],
        donor[d],
        field,
        (phi, psi),
        (current.n, stage1, stage2),
        tuple(publications),
        tuple(rates),
        tuple(observed_fields),
    )


def literal_inputs():
    """Exact cell averages of one Fourier product, y-major arrays for public bind."""
    import numpy as np

    nx, ny = CELLS
    x = (np.arange(nx) + 0.5) / nx
    y = (np.arange(ny) + 0.5) / ny
    mode = (
        np.sinc(1 / nx)
        * np.sinc(1 / ny)
        * np.cos(2 * np.pi * x)[None, :]
        * np.cos(2 * np.pi * y)[:, None]
    )
    return {
        key: np.ascontiguousarray(
            (float(INITIAL_MEANS[key]) + float(INITIAL_AMPLITUDES[key]) * mode)[None]
        )
        for key in ("receiver", "donor")
    }
