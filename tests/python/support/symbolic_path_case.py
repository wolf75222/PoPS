"""A two-state transport system unrelated to the moment-library design cases."""
import pops
from pops.domain import Rectangle
from pops.frames import Cartesian2D
from pops.layouts import Uniform
from pops.lib.time import ForwardEuler
from pops.math import ddt, div, maximum
from pops.mesh import CartesianGrid, PeriodicAxes
from pops.numerics import (DiscretizationPlan, PathConservativeFiniteVolume, SymbolicPath,
                           reconstruction, riemann, variables)
from pops.time import AdaptiveCFL


SIMPSON = ((0., 1./6.), (.5, 4./6.), (1., 1./6.))


def declarations(*, scale=1., reverse=False):
    frame = Rectangle("path_box", lower=(0., 0.), upper=(1., 1.)).frame(Cartesian2D())
    x, y = frame.axes
    model = pops.Model("authored_path", frame=frame)
    state = model.state("U", components=("v", "u") if reverse else ("u", "v"))
    u_index, v_index = (1, 0) if reverse else (0, 1)
    u, v = state[u_index], state[v_index]
    flux = model.flux("transport", state=state, frame=frame,
                      components={x: tuple(0.5*q for q in state), y: (0., 0.)})
    matrix = [[0., 0.], [0., 0.]]
    matrix[u_index][u_index] = scale * v * v
    product = model.nonconservative_product("variable_velocity", state=state,
        matrices={x: tuple(tuple(row) for row in matrix), y: ((0., 0.), (0., 0.))},
        conservative_components=("v",))

    def speed(left, right, axis):
        return (0.5 + abs(scale) * maximum(left[v_index]*left[v_index], right[v_index]*right[v_index])
                if axis == 0 else 0.)

    path = SymbolicPath(product, frame=frame, quadrature=SIMPSON, speed=speed)
    return model, state, flux, product, path


def make_case(*, scale=1., reverse=False, cells=8, dt=1.e-3):
    model, state, flux, product, path = declarations(scale=scale, reverse=reverse)
    rate = model.rate("full_balance", equation=ddt(state) == -div(flux) - product)
    method = PathConservativeFiniteVolume(flux=flux, path=path,
        variables=variables.Conservative(state), reconstruction=reconstruction.FirstOrder(),
        riemann=riemann.Rusanov())
    numerics = DiscretizationPlan()
    numerics.rates.add(rate, method)
    case = pops.Case("symbolic_path_case")
    block = case.block("transport", model)
    case.numerics(numerics, block=block)
    program = ForwardEuler(block[state], rate=rate)
    program.step_strategy(AdaptiveCFL(cfl=0.25, max_dt=dt))
    case.program(program)
    layout = Uniform(CartesianGrid(frame=path.frame, cells=(cells, cells),
                                   periodic=PeriodicAxes(path.frame.axes)))
    return case, layout
