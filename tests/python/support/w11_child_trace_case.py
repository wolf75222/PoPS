"""W11 finite FV current/capacity witness; not a kinetic sheath model."""
from fractions import Fraction
import numpy as np
import pops
from pops._ir.quantity import PhysicalDimension
from pops.boundary import TransportBoundarySet
from pops.boundary.transport import Outflow
from pops.domain import Rectangle
from pops.frames import Cartesian2D
from pops.initial import InitialCondition
from pops.layouts import Uniform
from pops.lib.initial import BindArray
from pops.math import ddt, div
from pops.mesh import CartesianGrid, PeriodicAxes
from pops.numerics import DiscretizationPlan, reconstruction, riemann, variables
from pops.numerics.spatial import FiniteVolume
from pops.projection import ConservativeCellAverage
from pops.time import Clock, TimePoint, StagePoint, SampleAndHold, FixedDt, RejectAttempt

CELLS = 8
Q0 = .7
OUTER_WEIGHT = Fraction(1, 2)


def build_case(*, ssprk2=False, guard_parent=False):
    frame = Rectangle("unit_domain", lower=(0., 0.), upper=(1., 1.)).frame(Cartesian2D())
    model = pops.Model("conservative_transport", frame=frame)
    state = model.state("u", components=("density",))
    x, y = frame.axes
    flux = model.flux("advection", frame=frame, state=state,
                      components={x: tuple(state), y: (0 * state[0],)},
                      waves={x: (1.,), y: (0.,)})
    rate = model.rate("balance", equation=ddt(state) == -div(flux))
    case = pops.Case("two_child_ticks")
    block = case.block("fluid", model)
    numerical = DiscretizationPlan()
    numerical.rates.add(rate, FiniteVolume(
        flux=flux, variables=variables.Conservative(state),
        reconstruction=reconstruction.FirstOrder(), riemann=riemann.Rusanov()))
    numerical.boundaries.add(TransportBoundarySet({
        frame.boundaries.x_min: Outflow(state=block[state]),
        frame.boundaries.x_max: Outflow(state=block[state]),
    }, periodic=PeriodicAxes((y,))))
    case.numerics(numerical, block=block)
    program = pops.Program("parent_blended_child_transport")
    temporal = program.state(block[state])
    child = Clock("child", owner=program.owner_path)
    on_child = program.synchronize(
        temporal.n, at=TimePoint(child), relation=SampleAndHold(), name="enter_child")
    quantity = program.integral_state("outflow_reservoir", initial=Q0, units=PhysicalDimension())
    child_state = program.state(block[state], clock=child)
    program.keep_history(child_state, depth=2)

    def tick(builder, previous):
        first = rate(previous)
        builder.accept_external_trace(quantity, rate=first, axis=0, side=1, component=0, scale=-1.)
        if not ssprk2:
            accepted_child = builder.value("child_euler", previous + builder.dt * first,
                                           at=child_state.next.point)
            return accepted_child
        stage = StagePoint("child_predictor", {"main": TimePoint(child, 1)})
        predictor = builder.value("child_predictor", previous + builder.dt * first, at=stage)
        second = rate(predictor)
        builder.accept_external_trace(quantity, rate=second, axis=0, side=1, component=0, scale=-1.)
        accepted_child = builder.value("child_ssprk2", Fraction(1, 2) * previous + Fraction(1, 2) * predictor
                                        + Fraction(1, 2) * builder.dt * second, at=child_state.next.point)
        return accepted_child

    advanced = program.subcycle(on_child, clock=child, within=program.clock, count=2,
                                body_fn=tick, name="two_ticks")
    returned = program.synchronize(advanced, at=temporal.next.point,
                                   relation=SampleAndHold(), name="return_parent")
    accepted = program.value("parent_blend", (1 - OUTER_WEIGHT) * temporal.n
                             + OUTER_WEIGHT * returned, at=temporal.next.point)
    if guard_parent:
        candidate_quantity = program.integral_value(quantity, at=temporal.next.point, scope="candidate").value
        accepted = program.guard("provisional_current_observed", accepted,
                                 candidate_quantity > Q0, action=RejectAttempt())
        increment = program.value("parent_increment", accepted - temporal.n, at=temporal.next.point)
        accepted = program.guard("parent_reject_after_child_current", accepted,
                                 program.norm_inf(increment) <= .009,
                                 action=RejectAttempt())
    program.commit(temporal.next, accepted)
    program.step_strategy(FixedDt(.1))
    case.program(program)
    case.initials.add(InitialCondition(state=block[state], value=BindArray(),
                                      projection=ConservativeCellAverage()))
    layout = Uniform(CartesianGrid(frame=frame, cells=(CELLS, CELLS), periodic=PeriodicAxes((y,))))
    values = 1 + .2 * (np.arange(CELLS) + .5) / CELLS
    initial = np.broadcast_to(values[None, None, :], (1, CELLS, CELLS)).copy()
    return case, layout, initial, quantity


def oracle(initial, dt, *, ssprk2=False):
    previous = np.asarray(initial).copy()
    current = previous.copy()
    charge = Q0
    child_dt = dt / 2
    for _ in range(2):
        left = np.concatenate((current[..., :1], current[..., :-1]), axis=-1)
        first = -CELLS * (current - left)
        predictor = current + child_dt * first
        if ssprk2:
            predictor_left = np.concatenate((predictor[..., :1], predictor[..., :-1]), axis=-1)
            second = -CELLS * (predictor - predictor_left)
            charge += float(OUTER_WEIGHT) * child_dt * .5 * float(np.mean(current[..., -1] + predictor[..., -1]))
            current = .5 * current + .5 * predictor + .5 * child_dt * second
        else:
            charge += float(OUTER_WEIGHT) * child_dt * float(np.mean(current[..., -1]))
            current = predictor
    return (1 - float(OUTER_WEIGHT)) * previous + float(OUTER_WEIGHT) * current, charge
