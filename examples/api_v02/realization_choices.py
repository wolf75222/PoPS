"""Multiple provisional evaluations and checked scalar failures in real PoPS."""
import numpy as np
import pops

from pops import math as pm
from pops.domain import Rectangle
from pops.frames import Cartesian2D
from pops.initial import InitialCondition
from pops.layouts import Uniform
from pops.lib.initial import BindArray
from pops.mesh import CartesianGrid, PeriodicAxes
from pops.numerics import DiscretizationPlan, ProgramRealization, StateStorage
from pops.projection import ConservativeCellAverage
from pops.realization import Realize, Use
from pops.time import FixedDt

from .common import CaseSpec, state_array, non_owner_metrics


def build_choice(kind="ancestral", *, dt=.02):
    frame = Rectangle("choice_domain", (0., 0.), (1., 1.)).frame(Cartesian2D())
    model = pops.Model("authored_exchange", frame=frame)
    state = model.state("U", components=("a", "b"), sampling="cell_average")
    if kind in {"ancestral", "discarded", "branch"}:
        source = model.source("exchange", on=state, value=(state[1]-state[0], state[0]-state[1]))
        balance = model.rate("exchange_balance", equation=pm.ddt(state) == source)
        use = Use(balance.occurrences[0], "cell", "step_average", "rate")
        numerics = DiscretizationPlan()
        numerics.rates.add(balance, ProgramRealization(uses=(use,)))
    elif kind in {"scalar_capture", "inactive_division", "scalar_reject",
                  "scalar_elementary", "scalar_domain_capture"}:
        numerics = DiscretizationPlan()
        numerics.states.add(state, StateStorage())
    else:
        raise ValueError("unknown realization choice")
    case = pops.Case("realization_" + kind)
    block = case.block("material", model)
    case.numerics(numerics, block=block)
    p = pops.Program("provisional_choice_" + kind)
    q = p.state(block[state])
    old = q.n
    initial = np.empty((2, 4, 8))
    initial[0] = .6 + .1*np.cos(2*np.pi*(np.arange(8)+.5)/8)[None, :]
    initial[1] = 1.1 + .03*np.sin(2*np.pi*(np.arange(8)+.5)/8)[None, :]
    if kind in {"ancestral", "discarded", "branch"}:
        operator = model.module.operator_handle("exchange")
        def evaluation(name, value):
            return Realize(name, owns=(use,), bindings={"old": value},
                           body=lambda t, old: t.source(operator, old),
                           outputs={use: ()}).evaluate(p)[use]
        first_input = p.value("trial_input", 2*old) if kind == "discarded" else old
        first = evaluation("trial", first_input)
        predictor = (old if kind == "discarded" else
                     p.value("midpoint_predictor", old + .5*p.dt*first, at=old.point))
        second = evaluation("selected", predictor)
        retained = (p.branch(p.requested_dt() > .015, lambda b: second, lambda b: first)
                    if kind == "branch" else second)
        candidate = p.value("candidate", old+p.dt*retained, at=q.next.point)
        rate = np.stack((initial[1]-initial[0], initial[0]-initial[1]))
        midpoint = initial + .5*dt*rate
        expected = (initial+dt*rate if kind == "discarded" or (kind == "branch" and dt <= .015)
                    else initial+dt*np.stack((midpoint[1]-midpoint[0], midpoint[0]-midpoint[1])))
    else:
        def singular(t):
            h = t.requested_dt()
            return h / (h-h)
        expected_scalar = 7.
        if kind == "scalar_elementary":
            x = p.requested_dt() + 1.2
            value = (pm.sin(x)+pm.cos(x)+pm.log(x)+pm.exp(x/5.)+pm.expm1(x)
                     +pm.sqrt(x)+pm.minimum(x, .5)+pm.maximum(x, 2.)+abs(-x)+x**2.3)
            import math
            a = dt+1.2
            expected_scalar = (math.sin(a)+math.cos(a)+math.log(a)+math.exp(a/5.)+math.expm1(a)
                               +math.sqrt(a)+min(a,.5)+max(a,2.)+abs(-a)+a**2.3)
        elif kind == "inactive_division":
            value = p.branch(p.isfinite(p.requested_dt()),
                             lambda t: t.requested_dt()*0.+7., singular)
        else:
            attempt = p.capture(lambda t: pm.log(t.float64(-1.))) if kind == "scalar_domain_capture" else p.capture(singular)
            value = (p.require_success(attempt) if kind == "scalar_reject" else
                     p.match(attempt, ok=lambda t, x: x,
                             err=lambda t, error: t.requested_dt()*0.+7.))
        candidate = p.value("candidate", (old[0]+value, old[1]-value), at=q.next.point)
        expected = initial + np.array((expected_scalar, -expected_scalar))[:, None, None]
    p.commit(q.next, candidate)
    p.step_strategy(FixedDt(dt))
    case.program(p)
    subject = block[state]
    case.initials.add(InitialCondition(state=subject, value=BindArray(),
                                       projection=ConservativeCellAverage()))
    layout = Uniform(CartesianGrid(frame=frame, cells=(8, 4), periodic=PeriodicAxes(frame.axes)))
    def invariants(runtime):
        values = state_array(runtime, "material", initial.shape)
        if values is None:
            return non_owner_metrics()
        error = float(np.max(np.abs(values-expected)))
        inventory = float(np.max(np.abs(values.sum(axis=0)-initial.sum(axis=0))))
        return {"passed": bool(error < 3e-14 and inventory < 3e-14),
                "max_error": error, "inventory_error": inventory, "metrics_owner": True}
    return CaseSpec(case, layout, {subject: initial}, {"material": subject}, dt, dt,
                    invariants, "native provisional realization / checked scalar: " + kind)
