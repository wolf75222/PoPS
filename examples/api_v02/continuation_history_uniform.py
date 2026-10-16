"""A Python AB2 update consumes History and publishes an assessed Continuation."""
from __future__ import annotations

import numpy as np
import pops
from pops.continuations import ContinuationSchema
from pops.time import FixedDt

from .common import CaseSpec, non_owner_metrics
from .continuation_forest_amr import _accepted
from .models_M01_M14 import _domain, _initial


def build_continuation_history_uniform(*, cells=(8, 4), dt=.1, steps=3, reject=False):
    frame, layout = _domain("continuation_history", cells)
    model = pops.Model("continuation_history", frame=frame)
    state = model.state("q", components=("amount",))
    case = pops.Case("continuation_history")
    block = case.block("method", model, states=(state,))
    p = pops.Program("continuation_history_ab2",
        history_field_metadata_budget=(8*1024*1024, 8*1024*1024))
    q = p.state(block[state])
    # Store before reading gives a defined cold start. Slot one still contains
    # the previous accepted sample after subsequent stores to slot zero.
    p.store_history("prior", q.n, depth=1)
    prior = p.history("prior", lag=1, space=q.n.space, block=block, state_ref=block[state])
    initial = p.product(state=q.n, previous=q.n, time=p.float64(0), step=p.int64(0))
    schema = ContinuationSchema(initial, fields=((q.next, ("state",)),),
        retained_fields=(("previous",),), time=("time",), step=("step",))
    base = p.continuation_open(initial, schema)
    old = p.continuation_read(base)
    proposal = p.product(h=p.requested_dt())
    candidate = p.product(
        state=p.value("ab2", old["state"]+p.dt*(1.5*old["state"]-.5*prior), at=q.next.point),
        previous=prior, time=old["time"]+proposal["h"], step=old["step"]+p.int64(1))
    p.continuation_publish(_accepted(p, base, proposal, candidate, p.product(), reject=reject))
    p.step_strategy(FixedDt(dt))
    case.program(p)
    x = (np.arange(cells[0])+.5)/cells[0]
    data = np.broadcast_to(1.+.2*np.sin(2*np.pi*x), (1, cells[1], cells[0])).copy()

    def invariants(runtime):
        actual = runtime.state_global("method")
        if actual is None:
            return non_owner_metrics()
        previous, current = 1., 1.
        for _ in range(runtime.macro_step()):
            previous, current = current, (1.+1.5*dt)*current-.5*dt*previous
        error = float(np.max(abs(np.asarray(actual).reshape(data.shape)-current*data)))
        return dict(passed=error < 5.e-13, metrics_owner=True, state_error=error)

    return CaseSpec(case, layout, _initial(case, block[state], data), {"method": block[state]},
                    steps*dt, dt, invariants,
                    "History and structural Continuation share native acceptance, rollback and restart")
