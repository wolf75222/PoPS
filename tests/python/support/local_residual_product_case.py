"""Public heterogeneous original equation shared by source and installed witnesses."""
import pops
from pops.domain import Rectangle
from pops.frames import Cartesian2D
from pops.initial import InitialCondition
from pops.layouts import Uniform
from pops.lib.initial import BindArray
from pops.mesh import CartesianGrid, PeriodicAxes
from pops.projection import ConservativeCellAverage
from pops.solvers.nonlinear import LocalNewton
from pops.time import LocalResidual, FixedDt, FailRun


def make_case(*, reverse=False, invalid=False, block_names=("left", "right")):
    frame = Rectangle("local_box", (0., 0.), (1., 1.)).frame(Cartesian2D())
    models, states = [], []
    for name, components in (("pair", ("x", "y")), ("triple", ("p", "q", "r"))):
        model = pops.Model(name, frame=frame)
        states.append(model.state("U", components=components))
        models.append(model)
    case = pops.Case("original_product")
    order = (1, 0) if reverse else (0, 1)
    blocks = {i: case.block(block_names[i], models[i]) for i in order}
    subjects = tuple(blocks[i][states[i]] for i in range(2))
    program = pops.Program("local_product_step")
    a, b = tuple(program.state(subject) for subject in subjects)
    a_seed = program.value("seed_a", 0*a.n, at=a.next.point)
    b_seed = program.value("seed_b", tuple(1+0*b.n[i] for i in range(3)), at=b.next.point)

    def residual(P, z, *, old_a, old_b):
        # A full invertible cross-block equation, no invertible leading 2x2 block.
        return {"a": (z["b"][0]-old_a[0], z["b"][1]-old_a[1]),
                "b": (z["a"][0]*z["b"][0]-old_b[0],
                      z["a"][1]+z["b"][1]-old_b[1],
                      1 if invalid else z["b"][2]-old_b[2])}

    initial = {"b": b_seed, "a": a_seed} if reverse else {"a": a_seed, "b": b_seed}
    solved = program.solve(LocalResidual(residual, initial,
        captures={"old_b": b.n, "old_a": a.n}),
        solver=LocalNewton(tolerance=1e-11)).consume(action=FailRun())
    program.commit_many({a.next: solved[blocks[0]], b.next: solved[blocks[1]]})
    program.step_strategy(FixedDt(.01))
    case.program(program)
    for subject in subjects:
        case.initials.add(InitialCondition(state=subject, value=BindArray(),
                                          projection=ConservativeCellAverage()))
    layout = Uniform(CartesianGrid(frame=frame, cells=(4,4), periodic=PeriodicAxes(frame.axes)))
    return case, layout, subjects
