"""Independent elliptic observations retained and read by the Python History algebra."""

import numpy as np
import pops
from pops._ir.elliptic import DivCoeffGrad, Reaction
from pops.fields import FieldBoundary, FieldDiscretization, FieldProblem, bcs
from pops.fields.methods import CellCenteredSecondOrder
from pops.time.expressions import ProgramScalar
from pops.solvers import CG
from pops.time import FailRun, FixedDt
from .common import CaseSpec, non_owner_metrics
from .methods_T01_T12 import _storage
from .models_M01_M14 import _initial


def build_independent_history(*, cells=(8, 4), dt=0.02):
    case, layout, model, block, state, p, q, data = _storage(
        "independent_history", ("amount", "time", "history_sum"), (1.0, 0.0, 0.0), cells=cells
    )
    # The budget is authored on construction, not mutated into an existing Program.
    p = pops.Program(
        "independent_history", history_field_metadata_budget=(8 * 1024 * 1024, 16 * 1024 * 1024)
    )
    q = p.state(block[state])
    phi = model.field("potential")
    problem = FieldProblem(
        "screened_history",
        unknowns=(phi,),
        equations=(Reaction(phi, 2) - DivCoeffGrad(phi, 0.04) == state[0],),
        boundaries=(
            FieldBoundary(phi, bcs.BoundaryCondition(bcs.AllPhysicalBoundaries(), bcs.Periodic())),
        ),
    )
    field = case.field(
        problem,
        FieldDiscretization(
            method=CellCenteredSecondOrder(),
            boundaries=(),
            solver=CG(max_iter=200, rel_tol=1.0e-12, abs_tol=1.0e-13),
        ),
    )
    observed = field.observe(
        p.solve(field, values={block[state]: q.n}, at=q.n.point).consume(action=FailRun())
    )[field[phi]]
    p.store_history("potential", observed, depth=2, owner_block=block)
    retained = p.history("potential", lag=1)
    total = p.sum(retained)
    p.commit(
        q.next,
        p.value(
            "history_consumer",
            (q.n[0] + p.dt, q.n[1] + p.dt, ProgramScalar(total)),
            at=q.next.point,
        ),
    )
    p.step_strategy(FixedDt(dt))
    case.program(p)

    def check(runtime):
        values = runtime.state_global("method")
        if values is None:
            return non_owner_metrics()
        a = np.asarray(values).reshape(data.shape)
        step = runtime.macro_step()
        expected = np.empty_like(data)
        expected[0] = 1 + step * dt
        expected[1] = step * dt
        expected[2] = np.prod(cells) * (1 + max(0, step - 2) * dt) / 2
        error = float(np.max(abs(a - expected)))
        return dict(passed=error < 3.0e-10, metrics_owner=True, history_consumer_error=error)

    return CaseSpec(
        case,
        layout,
        _initial(case, block[state], data),
        {"method": block[state]},
        4 * dt,
        dt,
        check,
        "Screened PDE history retains an independent provider through native reads/restart",
    )
