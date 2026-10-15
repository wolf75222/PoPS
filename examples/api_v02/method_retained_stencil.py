"""AB2 face equations retained by native History or complete Continuation values.

The reference recurrence is only an independent receiver; executed equations are the Program.
"""
from __future__ import annotations

from fractions import Fraction
import numpy as np
from pops.continuations import Advance, Assess, ContinuationSchema, advance_once
from pops.graph_functions import graph
from pops.numerics import DiscretizationPlan, StateStorage
from pops.time import FixedDt, GuardRole, RejectAttempt
from pops.time.expressions import ProgramScalar
from pops.time.variant_types import Bool, Failure, Product, Result, type_of

from .common import CaseSpec, non_owner_metrics, state_array
from .method_scoped_stencil import face_equation
from .methods_T01_T12 import _storage
from .models_M01_M14 import _initial


def initial_state(cells=(16, 4)):
    x, y = np.meshgrid((np.arange(cells[0])+.5)/cells[0],
                       (np.arange(cells[1])+.5)/cells[1])
    return (1+.2*np.cos(2*np.pi*x)+.1*np.sin(2*np.pi*y))[None]


def reference(initial, *, dt=.01, steps=2):
    current = np.asarray(initial, dtype=float).copy()
    previous_faces = None
    increments = []
    face_terms = []
    for _ in range(steps):
        faces = np.roll(current, 1, axis=-1)
        terms = ((dt, faces),) if previous_faces is None else (
            (1.5*dt, faces), (-.5*dt, previous_faces))
        integrated = sum(weight*value for weight, value in terms)
        change = -current.shape[-1]*(np.roll(integrated, -1, axis=-1)-integrated)
        increments.append(change)
        face_terms.append(terms)
        previous_faces = faces
        current += change
    return current, increments, face_terms


def build_retained_stencil(*, retention="history", steps=2, reject_after=None,
                           basis_capacity=None, cells=(16, 4), dt=.01):
    if retention not in ("history", "continuation"):
        raise ValueError("retention must be history or continuation")
    case, layout, _model, block, state, p, q, _data = _storage(
        "retained-face-AB2", ("q",), (1.,), cells=cells)
    plan = DiscretizationPlan()
    plan.states.add(state, StateStorage(ghost_depth=1))
    case.numerics(plan, block=block)
    # This is a resource contract, not a change to the equations or their stability gate.
    if basis_capacity is not None:
        p.flux_expression_budget = (basis_capacity, 1)

    if retention == "history":
        rate = p.stencil_flux_divergence("current_faces", q.n, flux=face_equation(1.))
        p.store_history("retained_rhs", rate, depth=1)
        interval = p.interval(duration=p.requested_dt())

        def mature(builder):
            old = builder.history("retained_rhs", 1, space=q.n.space,
                                  block=q.n.block, state_ref=q.n.state_ref)
            old = builder.history_interval(old).bind(builder, old, name="authentic_old_rhs")
            candidate = builder.value("AB2", q.n+p.dt*(Fraction(3, 2)*rate-Fraction(1, 2)*old),
                                      at=q.next.point)
            return builder.guard("energy_admission", candidate,
                builder.dot(candidate, candidate) <= builder.dot(q.n, q.n),
                role=GuardRole.NUMERICAL_STABILITY, action=RejectAttempt())

        candidate = p.branch(interval.begin > p.float64(0.), mature,
            lambda builder: builder.value("Euler", q.n+p.dt*rate, at=q.next.point))
        if reject_after is not None:
            candidate = p.guard("parent_rejection", candidate,
                interval.begin < p.float64(reject_after*dt), action=RejectAttempt())
        p.commit(q.next, candidate)
    else:
        zero_rate = p.rate("initial_rate", (0.,), of=q.n)
        initial_alias = p.value("initial_alias", 1*q.n, at=q.next.point)
        initial = p.product(state=q.n, alias=initial_alias, rate=zero_rate,
                            time=p.float64(0.), step=p.int64(0))
        schema = ContinuationSchema(initial, fields=((q.next, ("state",)),),
                                    retained_fields=(("alias",), ("rate",)),
                                    time=("time",), step=("step",))
        base = p.continuation_open(initial, schema)
        proposal = p.product(h=p.requested_dt())
        envelope_type = Product(continuation=schema.write_type, increment=Product(),
                                observations=Product(), checks=Product())
        outcome = Result(envelope_type, Failure())

        @graph(inputs=Product(base=schema.read_type, proposal=type_of(proposal)), output=outcome)
        def trial(g, args):
            old, h = args["base"], args["proposal"]["h"]
            current = g.stencil_flux_divergence("current_faces", old["state"], flux=face_equation(1.))
            def mature(b):
                value = b.value("AB2", (old["state"][0]+ProgramScalar(h)*(
                    Fraction(3, 2)*current[0]-Fraction(1, 2)*old["rate"][0]),), at=q.next.point)
                return b.guard("energy_admission", value,
                    b.dot(value, value) <= b.dot(old["state"], old["state"]),
                    role=GuardRole.NUMERICAL_STABILITY, action=RejectAttempt())
            value = g.branch(old["step"] > g.int64(0), mature,
                lambda b: b.value("Euler", (old["state"][0]+ProgramScalar(h)*current[0],), at=q.next.point))
            next_value = g.product(state=value, alias=value, rate=current,
                                  time=old["time"]+h, step=old["step"]+g.int64(1))
            envelope = g.product(continuation=next_value, increment=g.product(),
                                 observations=g.product(), checks=g.product())
            return g.variant(outcome, "ok", envelope)

        assessment_type = Product(accepted=Bool, checks=Product())
        @graph(inputs=Product(base=schema.read_type, proposal=type_of(proposal), candidate=envelope_type),
               output=assessment_type)
        def assessment(g, args):
            valid = args["candidate"]["continuation"]["time"] == args["base"]["time"]+args["proposal"]["h"]
            if reject_after is not None:
                valid = valid & (args["base"]["step"] < g.int64(reject_after))
            return g.product(accepted=valid, checks=g.product())
        accepted = p.require_success(advance_once(p, Advance(trial, Assess(assessment)), base=base, proposal=proposal))
        p.continuation_publish(accepted)
    p.step_strategy(FixedDt(dt))
    case.program(p)
    initial_data = initial_state(cells)
    expected, _, _ = reference(initial_data, dt=dt, steps=steps)
    def check(runtime):
        actual = state_array(runtime, "method", initial_data.shape)
        if actual is None:
            return non_owner_metrics()
        error = float(np.max(abs(actual-expected)))
        return dict(passed=error < 4.e-14, metrics_owner=True, state_error=error,
                    no_op_error=float(np.max(abs(expected-initial_data))))
    return CaseSpec(case, layout, _initial(case, block[state], initial_data), {"method": block[state]},
                    steps*dt, dt, check, "AB2 retained face source via "+retention)
