"""Public local source/apply equation with heterogeneous, coupled unknowns."""
import pops
from pops.domain import Rectangle
from pops.frames import Cartesian2D
from pops.initial import InitialCondition
from pops.layouts import Uniform
from pops.lib.initial import BindArray
from pops.mesh import CartesianGrid, PeriodicAxes
from pops.params import RuntimeParam
from pops.projection import ConservativeCellAverage
from pops.solvers.nonlinear import LocalNewton
from pops.time import LocalResidual, FixedDt, FailRun


def make_case(widths=(2, 3), *, reverse=False, derived=False, return_parameters=False):
    frame = Rectangle("local_operators_box", (0., 0.), (1., 1.)).frame(Cartesian2D())
    models, states, sources, operators, parameters = [], [], [], [], []
    for i, width in enumerate(widths):
        model = pops.Model("physics_%d" % i, frame=frame)
        state = model.state("U", components=tuple("v%d" % c for c in range(width)))
        parameter = model.param(RuntimeParam("gain", default=.5 + .25*i))
        parameters.append(parameter)
        gain = model.value(parameter)
        model.source("quadratic", on=state,
                     value=tuple(gain*state[c]*state[c] for c in range(width)))
        operators.append(model.operator("diagonal", returns=model.local_linear_operator(
            "diagonal", on=state,
            matrix=tuple(tuple(float(c+1) if c == j else 0. for j in range(width))
                         for c in range(width)))))
        models.append(model)
        states.append(state)
        sources.append(model.module.operator_handle("quadratic"))
    case = pops.Case("product_operators")
    order = tuple(reversed(range(len(widths)))) if reverse else range(len(widths))
    blocks = {i: case.block("block_%d" % i, models[i]) for i in order}
    subjects = tuple(blocks[i][state] for i, state in enumerate(states))
    program = pops.Program("product_operators_step")
    handles = tuple(program.state(subject) for subject in subjects)
    keys = tuple("unknown_%d" % i for i in range(len(widths)))
    seeds = {key: program.value("seed_%d" % i,
        tuple(1+0*handles[i].n[c] for c in range(widths[i])), at=handles[i].next.point)
        for i, key in enumerate(keys)}

    def residual(P, unknowns, **captures):
        joint = sum(unknowns[key][c] for i, key in enumerate(keys) for c in range(widths[i]))
        rows = {}
        for i, key in enumerate(keys):
            argument = (P.value("twice_"+key, 2*unknowns[key], at=unknowns[key].point)
                        if derived else unknowns[key])
            source = P.source(sources[i], state=argument)
            applied = P.apply(operators[i], state=unknowns[key])
            rows[key] = tuple(source[c] + applied[c] + .1*joint - captures["old_%d" % i][c]
                              for c in range(widths[i]))
        return rows

    solved = program.solve(LocalResidual(residual,
        dict(reversed(tuple(seeds.items()))) if reverse else seeds,
        captures={"old_%d" % i: h.n for i, h in enumerate(handles)}),
        solver=LocalNewton(tolerance=1e-11)).consume(action=FailRun())
    program.commit_many({h.next: solved[blocks[i]] for i, h in enumerate(handles)})
    program.step_strategy(FixedDt(.01))
    case.program(program)
    for subject in subjects:
        case.initials.add(InitialCondition(state=subject, value=BindArray(),
                                          projection=ConservativeCellAverage()))
    layout = Uniform(CartesianGrid(frame=frame, cells=(4, 4), periodic=PeriodicAxes(frame.axes)))
    if return_parameters:
        return case, layout, subjects, tuple(blocks[i][p] for i, p in enumerate(parameters))
    return case, layout, subjects
