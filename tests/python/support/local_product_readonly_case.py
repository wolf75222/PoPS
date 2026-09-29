"""One algebraic unknown and one independently owned, read-only State capture."""

import pops

from pops.domain import Rectangle
from pops.frames import Cartesian2D
from pops.initial import InitialCondition
from pops.layouts import Uniform
from pops.lib.initial import BindArray
from pops.mesh import CartesianGrid, PeriodicAxes
from pops.projection import ConservativeCellAverage
from pops.solvers.nonlinear import LocalNewton
from pops.time import FailRun, FixedDt, LocalResidual


def make_case(*, reverse=False):
    frame = Rectangle("capture_box", (0., 0.), (1., 1.)).frame(Cartesian2D())
    case = pops.Case("one_unknown_frozen_target")
    subjects = {}
    for name in (("target", "dual") if reverse else ("dual", "target")):
        model = pops.Model(name, frame=frame)
        state = model.state("U", components=("a", "b", "c"))
        subjects[name] = case.block(name, model)[state]
    program = pops.Program("one_unknown")
    dual, target = (program.state(subjects[name]) for name in ("dual", "target"))
    seed = program.value("seed", dual.n, at=dual.next.point)

    def residual(P, unknowns, *, frozen):
        return {"dual": tuple(unknowns["dual"][i]**2-frozen[i] for i in range(3))}

    solved = program.solve(LocalResidual(residual, {"dual": seed},
        captures={"frozen": target.n}), solver=LocalNewton(tolerance=1e-12)
        ).consume(action=FailRun())
    program.commit(dual.next, solved[subjects["dual"].block_ref])
    program.step_strategy(FixedDt(.01))
    case.program(program)
    for subject in subjects.values():
        case.initials.add(InitialCondition(state=subject, value=BindArray(),
                                          projection=ConservativeCellAverage()))
    layout = Uniform(CartesianGrid(frame=frame, cells=(4, 4),
                                  periodic=PeriodicAxes(frame.axes)))
    return case, layout
