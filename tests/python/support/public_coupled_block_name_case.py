"""Public coupled exchange with exact arbitrary Block labels; Source fixture."""
import pops
from pops.domain import Rectangle
from pops.frames import Cartesian2D
from pops.layouts import Uniform
from pops.mesh import CartesianGrid, PeriodicAxes
from pops.time import FixedDt


def build(labels=("z-first", "a-second"), *, reverse=False):
    frame = Rectangle("alias collision domain", (0, 0), (1, 1)).frame(Cartesian2D())
    model = pops.Model("physical exchange alias authority", frame=frame)
    a = model.species("first space", state=("pops_input_0_component_0", "balance"))
    b = model.species("second space", state=("pops_input_1_component_0", "pops input 0 component 0"))
    q = b[0] - a[0]
    operator = model.coupled_rate("same conservative exchange", inputs=(a, b),
                                 outputs={a: (q, -q), b: (-q, q)})
    case = pops.Case("internal-looking authored names")
    spaces = (a, b)
    blocks = {i: case.block(labels[i], model, states=(spaces[i],))
              for i in ((1, 0) if reverse else (0, 1))}
    program = pops.Program("explicit exchange")
    states = tuple(program.state(blocks[i][spaces[i]]) for i in range(2))
    rates = operator(*(state.n for state in states))
    for i in ((1, 0) if reverse else (0, 1)):
        state, block = states[i], blocks[i]
        program.commit(state.next, program.value("next " + block.local_id,
            state.n + program.dt * rates[block], at=state.next.point))
    program.step_strategy(FixedDt(1/64))
    case.program(program)
    return pops.resolve(pops.validate(case), layout=Uniform(CartesianGrid(
        frame=frame, cells=(3, 5), periodic=PeriodicAxes(frame.axes))))
