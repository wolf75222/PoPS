"""Four-state, two-direction nonlinear path with an independent polynomial oracle."""

from functools import reduce

import numpy as np
import pytest

import pops
from pops.domain import Rectangle
from pops.frames import Cartesian2D
from pops.layouts import Uniform
from pops.lib.time import ForwardEuler
from pops.math import ddt, div, maximum
from pops.mesh import CartesianGrid, PeriodicAxes
from pops.numerics import (DiscretizationPlan, PathConservativeFiniteVolume,
                           SymbolicPath, reconstruction, riemann, variables)
from pops.time import AdaptiveCFL
from tests.python.support.native_execution_context import artifact_execution_context


pytestmark = [pytest.mark.compiler, pytest.mark.native_loader]
_X_FLUX = {"a": .25, "b": .15, "c": -.2, "d": .1}
_Y_FLUX = {"a": .05, "b": -.1, "c": .3, "d": .2}


def _gauss_rule(points):
    nodes, weights = np.polynomial.legendre.leggauss(points)
    return tuple((float((node+1)/2), float(weight/2))
                 for node, weight in zip(nodes, weights, strict=True))


def _case(order, cells=8):
    frame = Rectangle("noncommuting_box", lower=(0., 0.), upper=(1., 1.)).frame(Cartesian2D())
    x_axis, y_axis = frame.axes
    model = pops.Model("four_state_nonlinear_path", frame=frame)
    state = model.state("Q", components=order)
    q = dict(zip(order, state, strict=True))
    flux = model.flux("two_direction_flux", state=state, frame=frame,
        components={x_axis: tuple(_X_FLUX[name]*q[name] for name in order),
                    y_axis: tuple(_Y_FLUX[name]*q[name] for name in order)})

    def matrix(entries):
        rows = [[0. for _ in order] for _ in order]
        for row, column, expression in entries:
            rows[order.index(row)][order.index(column)] = expression
        return tuple(tuple(row) for row in rows)

    bx = matrix((("a", "b", .4*q["c"]), ("b", "c", -.3*q["a"]),
                 ("c", "a", .2*q["b"])))
    by = matrix((("a", "c", .35*q["b"]), ("b", "a", .25*q["c"]),
                 ("c", "b", -.15*q["a"])))
    product = model.nonconservative_product("cyclic_product", state=state,
        matrices={x_axis: bx, y_axis: by}, conservative_components=("d",))

    def bend(left, right, s, axis):
        del left, right
        rows = [[0. for _ in order] for _ in order]
        if axis == 0:
            rows[order.index("b")][order.index("a")] = .5*s
        else:
            rows[order.index("c")][order.index("b")] = -.4*(1-s)
        return tuple(tuple(row) for row in rows)

    def speed(left, right, axis):
        # ||Psi||_inf <= M + max|K|*max(s(1-s))*D <= M + D/8.
        # Infinity row-sum bounds for DF+B are .25+.4||Psi||_inf (x)
        # and .3+.35||Psi||_inf (y), since each row has one B entry.
        magnitude = reduce(maximum, (abs(value) for value in (*left, *right)))
        jump = reduce(maximum, (abs(b-a) for a, b in zip(left, right, strict=True)))
        return ((.25 if axis == 0 else .3)
                + (.4 if axis == 0 else .35)*(magnitude + jump/8))

    path = SymbolicPath(product, frame=frame, quadrature=_gauss_rule(4),
                        speed=speed, bend=bend)
    rate = model.rate("balance", equation=ddt(state) == -div(flux) - product)
    method = PathConservativeFiniteVolume(flux=flux, path=path,
        variables=variables.Conservative(state), reconstruction=reconstruction.FirstOrder(),
        riemann=riemann.Rusanov())
    plan = DiscretizationPlan()
    plan.rates.add(rate, method)
    case = pops.Case("four_state_noncommuting_case")
    block = case.block("four_state", model)
    case.numerics(plan, block=block)
    program = ForwardEuler(block[state], rate=rate)
    program.step_strategy(AdaptiveCFL(cfl=.25, max_dt=1.e-4))
    case.program(program)
    layout = Uniform(CartesianGrid(frame=frame, cells=(cells, cells),
                                   periodic=PeriodicAxes(frame.axes)))
    return case, layout, path


def _initial(order, cells):
    x = (np.arange(cells) + .5) / cells
    X, Y = np.meshgrid(x, x, indexing="xy")
    values = {"a": 2. + .08*np.sin(2*np.pi*X) + .04*np.cos(2*np.pi*Y),
              "b": 1. + .07*np.cos(2*np.pi*X-.3) + .03*np.sin(2*np.pi*Y),
              "c": 1.5 + .06*np.sin(2*np.pi*(X+Y)),
              "d": .8 + .05*np.cos(2*np.pi*(X-2*Y))}
    return np.ascontiguousarray(np.stack(tuple(values[name] for name in order)))


def _axis_oracle(initial, order, axis):
    spatial_axis = 2 if axis == 0 else 1
    right = np.roll(initial, -1, axis=spatial_axis)
    L, R = ({name: initial[order.index(name)] for name in order},
            {name: right[order.index(name)] for name in order})
    delta = {name: R[name]-L[name] for name in order}
    magnitude = np.maximum(np.max(np.abs(initial), axis=0),
                           np.max(np.abs(right), axis=0))
    jump_magnitude = np.max(np.abs(right-initial), axis=0)
    speed = ((.25 if axis == 0 else .3)
             + (.4 if axis == 0 else .35)*(magnitude + jump_magnitude/8))
    coefficients = _X_FLUX if axis == 0 else _Y_FLUX
    flux = np.stack(tuple(.5*coefficients[name]*(L[name]+R[name])
                          - .5*speed*delta[name] for name in order))

    # Separate 12-point oracle: the authored four-point rule is exact through
    # degree 7; B(Psi)Psi' has degree at most 5 for these cubic curves.
    terms = {name: np.zeros_like(L[name]) for name in order}
    for s, weight in _gauss_rule(12):
        state = {name: L[name] + s*delta[name] for name in order}
        tangent = dict(delta)
        if axis == 0:
            state["b"] = state["b"] + .5*s*s*(1-s)*delta["a"]
            tangent["b"] = tangent["b"] + .5*(2*s-3*s*s)*delta["a"]
            action = {"a": .4*state["c"]*tangent["b"],
                      "b": -.3*state["a"]*tangent["c"],
                      "c": .2*state["b"]*tangent["a"]}
        else:
            state["c"] = state["c"] - .4*s*(1-s)**2*delta["b"]
            tangent["c"] = tangent["c"] - .4*(1-4*s+3*s*s)*delta["b"]
            action = {"a": .35*state["b"]*tangent["c"],
                      "b": .25*state["c"]*tangent["a"],
                      "c": -.15*state["a"]*tangent["b"]}
        for name, value in action.items():
            terms[name] += weight*value
    integral = np.stack(tuple(terms[name] for name in order))
    rhs = np.roll(flux, 1, axis=spatial_axis)-flux
    rhs -= .5*(integral+np.roll(integral, 1, axis=spatial_axis))
    return rhs


@pytest.mark.parametrize("order", [("a", "b", "c", "d"), ("d", "c", "a", "b")])
def test_four_state_noncommuting_curved_path_matches_native_2d_step(
        isolated_native_cache, native_cxx, kokkos_root, order):
    del isolated_native_cache, native_cxx, kokkos_root
    cells, dt = 8, 1.e-4
    case, layout, _ = _case(order, cells)
    artifact = pops.compile(pops.resolve(pops.validate(case), layout=layout))
    initial = _initial(order, cells)
    expected = initial + dt*cells*(_axis_oracle(initial, order, 0)
                                   + _axis_oracle(initial, order, 1))
    runtime = pops.bind(artifact, initial_state={"four_state": initial},
                        resources={"execution_context": artifact_execution_context(artifact)})
    report = pops.run(runtime, t_end=dt, max_steps=1)
    assert report.accepted_steps == 1
    actual = np.asarray(runtime.state_global("four_state")).reshape(initial.shape)
    np.testing.assert_allclose(actual, expected, rtol=0., atol=1.e-13)
    conservative = order.index("d")
    assert actual[conservative].sum() == pytest.approx(initial[conservative].sum(), abs=1.e-13)
    assert not np.allclose(actual, initial, rtol=0., atol=1.e-10)
