"""A solved FieldProblem value belongs to its own provider, not to the stored State."""
import numpy as np
from pops._ir.elliptic import DivCoeffGrad, Reaction
from pops.continuations import ContinuationSchema
from pops.fields import FieldBoundary, FieldDiscretization, FieldProblem, bcs
from pops.fields.methods import CellCenteredSecondOrder
from pops.solvers import CG
from pops.time import FailRun, FixedDt
from examples.api_v02.common import CaseSpec, non_owner_metrics
from examples.api_v02.methods_T01_T12 import _storage
from examples.api_v02.models_M01_M14 import _initial
from examples.api_v02.continuation_forest_amr import _accepted


def build_independent_uniform(*, cells=(8,8), dt=.002, steps=2, reject=False):
    parts = _storage("independent_field_continuation", ("amount",), (1.,), cells=cells)
    case, layout, model, block, state, p, q, data = parts
    data[0] = 1.+.1*np.sin(2*np.pi*(np.arange(cells[0])+.5)/cells[0])[None, :]
    phi = model.field("potential")
    problem = FieldProblem("screened_observation", unknowns=(phi,),
        equations=(Reaction(phi,1)-DivCoeffGrad(phi,.04) == state[0],),
        boundaries=(FieldBoundary(phi,bcs.BoundaryCondition(bcs.AllPhysicalBoundaries(),bcs.Periodic())),))
    solver = CG(max_iter=200, rel_tol=1.e-12, abs_tol=1.e-13)
    field = case.field(problem,FieldDiscretization(method=CellCenteredSecondOrder(),boundaries=(),solver=solver))
    seed = field.observe(p.solve(field,values={block[state]:q.n},at=q.n.point).consume(action=FailRun()))
    initial = p.product(state=q.n, time=p.float64(0), step=p.int64(0),
        retained=seed[field[phi]], gradient=seed.gradient(field[phi],dimension=2), previous=p.float64(0))
    schema = ContinuationSchema(initial,fields=((q.next,("state",)),),
        retained_fields=(("retained",),("gradient",)),time=("time",),step=("step",))
    base = p.continuation_open(initial,schema)
    old = p.continuation_read(base)
    previous = p.sum(old["retained"])
    def observe(b):
        source = b.value("scaled_source", old["state"]*(1+50*old["time"]), at=q.n.point)
        solved = field.observe(b.solve(field,values={block[state]:source},at=q.n.point).consume(action=FailRun()))
        return b.product(retained=solved[field[phi]], gradient=solved.gradient(field[phi],dimension=2))
    selected = p.branch(old["step"] == p.int64(0), observe, observe)
    proposal = p.product(h=p.requested_dt())
    candidate = p.product(state=p.value("unchanged_state",old["state"],at=q.next.point),
        time=old["time"]+proposal["h"], step=old["step"]+p.int64(1),
        retained=selected["retained"], gradient=selected["gradient"],previous=previous)
    accepted = _accepted(p,base,proposal,candidate,p.product(),reject=reject)
    p.continuation_publish(accepted)
    p.step_strategy(FixedDt(dt))
    case.program(p)
    def check(runtime):
        values=runtime.state_global("method")
        if values is None:
            return non_owner_metrics()
        error=float(np.max(abs(np.asarray(values).reshape(data.shape)-data)))
        return {"passed":error<1.e-14,"metrics_owner":True,"unchanged_error":error}
    return CaseSpec(case,layout,_initial(case,block[state],data),{"method":block[state]},
        steps*dt,dt,check,"Independent scalar and gradient FieldProblem observations in a complete Uniform Continuation")
