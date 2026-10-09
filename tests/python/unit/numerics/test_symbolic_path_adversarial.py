"""Independent closure and model-generic checks for authored path kernels."""

from functools import reduce

import pytest
import pops
from pops.codegen.module_codegen import _emit_bricks
from pops.codegen.module_lowering import lower_and_validate
from pops.domain import Rectangle
from pops.frames import Cartesian2D
from pops.layouts import Uniform
from pops.lib.time import ForwardEuler
from pops.math import ddt, div, maximum
from pops.mesh import CartesianGrid, PeriodicAxes
from pops.numerics import (DiscretizationPlan, PathConservativeFiniteVolume,
                           SymbolicPath, reconstruction, riemann, variables)
from pops.time import AdaptiveCFL
from pops._ir.expr import Var
from tests.python.support.symbolic_path_case import declarations


def _three_component_case(*, bent=False):
    """A triangular 3-state product with two different directional matrices."""
    frame = Rectangle("triangular_box", lower=(0., 0.), upper=(1., 1.)).frame(Cartesian2D())
    x, y = frame.axes
    model = pops.Model("triangular_transport", frame=frame)
    state = model.state("Q", components=("alpha", "beta", "gamma"))
    alpha, beta, gamma = state
    flux = model.flux("advection", state=state, frame=frame,
                      components={x: tuple(0.2 * q for q in state),
                                  y: tuple(0.1 * q for q in state)})
    product = model.nonconservative_product("triangular_coupling", state=state,
        matrices={x: ((1.7 * beta, 0., 0.), (0., 0., -0.4 * gamma),
                      (0., 0.3 * alpha, 0.)),
                  y: ((0., 0., 0.), (0., 0., 0.), (2.1 * alpha, 0., 0.))})

    def speed(left, right, axis):
        del axis
        magnitude = reduce(maximum, (abs(q) for q in (*left, *right)))
        return 10 + 3 * magnitude

    def bend(left, right, s, axis):
        del left, right, s
        # K_beta,alpha=2 gives deviation 2*(alpha_R-alpha_L)=6 at this probe.
        # The bend vanishes automatically for a zero jump.
        return (((0., 0., 0.), (2., 0., 0.), (0., 0., 0.)) if axis == 0
                else ((0., 0., 0.), (0., 0., 0.), (0., 0., 0.)))

    rule = ((0., 1./6.), (.5, 4./6.), (1., 1./6.)) if bent else ((0., .5), (1., .5))
    path = SymbolicPath(product, frame=frame, quadrature=rule, speed=speed,
                        bend=bend if bent else None)
    rate = model.rate("balance", equation=ddt(state) == -div(flux) - product)
    method = PathConservativeFiniteVolume(flux=flux, path=path,
        variables=variables.Conservative(state), reconstruction=reconstruction.FirstOrder(),
        riemann=riemann.Rusanov())
    plan = DiscretizationPlan()
    plan.rates.add(rate, method)
    case = pops.Case("triangular_generic_path")
    block = case.block("triangle", model)
    case.numerics(plan, block=block)
    program = ForwardEuler(block[state], rate=rate)
    program.step_strategy(AdaptiveCFL(cfl=0.25, max_dt=1.e-3))
    case.program(program)
    layout = Uniform(CartesianGrid(frame=frame, cells=(8, 8),
                                   periodic=PeriodicAxes(frame.axes)))
    return case, layout, path


def test_three_component_two_axis_product_has_independent_oracle_and_native_source():
    case, layout, path = _three_component_case()
    left, right = (2., 3., 4.), (5., 7., 9.)
    env = {symbol.name: value for symbols, values in
           ((path.left_symbols, left), (path.right_symbols, right))
           for symbol, value in zip(symbols, values, strict=True)}
    x_integral = tuple(value.eval(env) for value in path.integrals[0])
    y_integral = tuple(value.eval(env) for value in path.integrals[1])
    assert x_integral == pytest.approx((1.7 * (3. + 7.) / 2 * (5. - 2.),
                                       -0.4 * (4. + 9.) / 2 * (9. - 4.),
                                       0.3 * (2. + 5.) / 2 * (7. - 3.)))
    assert y_integral == pytest.approx((0., 0., 2.1 * (2. + 5.) / 2 * (5. - 2.)))
    assert all(bound.eval(env) >= 0. for bound in path.speeds)

    resolved = pops.resolve(pops.validate(case), layout=layout)
    block = resolved.blocks[0]
    emitter, _ = lower_and_validate(block.model, state_space=block.state_spaces[0],
        resolved_operations=block.resolved_operations, numerics=block.numerics)
    source = _emit_bricks(emitter._m)[1]
    assert "PathIntegralResult<n_vars>" in source
    assert "pops_path_left_2" in source
    assert "direction[1]" in source
    assert "normalized_moment_path" not in source


def test_curved_path_integrates_declared_product_with_exact_tangent():
    _, _, path = _three_component_case(bent=True)
    left, right = (2., 3., 4.), (5., 7., 9.)
    env = {symbol.name: value for symbols, values in
           ((path.left_symbols, left), (path.right_symbols, right))
           for symbol, value in zip(symbols, values, strict=True)}
    x_integral = tuple(value.eval(env) for value in path.integrals[0])
    # beta(s) = 3 + 4s + 6s(1-s); d(alpha)/ds = 3, so mean beta = 6.
    # gamma d(gamma) is exact. The alpha d(beta) term probes the bend tangent:
    # integral alpha(s)*(4 + 6*(1-2s)) ds = 3.5*4 - 3.
    assert x_integral == pytest.approx((1.7 * 3. * (3. + 4./2. + 6./6.),
                                       -0.4 * (9.**2 - 4.**2) / 2.,
                                       0.3 * ((2. + 5.) / 2 * 4. - 3.)))
    equal_env = {symbol.name: value for symbols in
                 (path.left_symbols, path.right_symbols)
                 for symbol, value in zip(symbols, left, strict=True)}
    assert tuple(value.eval(equal_env) for value in path.integrals[0]) == pytest.approx((0., 0., 0.))


def test_foreign_free_variable_in_authored_integral_is_rejected():
    _, _, _, product, path = declarations()
    with pytest.raises(ValueError, match="free variable"):
        SymbolicPath(product, frame=path.frame,
            quadrature=((0., 1.),),
            bend=lambda left, right, s, axis: ((Var(left[0].name, "path_left"), 0.),
                                               (0., 0.)),
            speed=lambda left, right, axis: 1.)


def test_free_integral_callback_is_not_part_of_public_api():
    _, _, _, product, path = declarations()
    with pytest.raises(TypeError, match="integral"):
        SymbolicPath(product, frame=path.frame,
            integral=lambda action, left, right, axis: (0., 0.),
            speed=lambda left, right, axis: 1.)


@pytest.mark.parametrize("rule", [(), ((-0.1, 1.),), ((0.5, 0.8),),
                                  ((float("nan"), 1.),), ((0.5, float("inf")),)])
def test_invalid_quadrature_rejected_before_native_emission(rule):
    _, _, _, product, path = declarations()
    with pytest.raises(ValueError, match="quadrature"):
        SymbolicPath(product, frame=path.frame, quadrature=rule,
                     speed=lambda left, right, axis: 1.)


@pytest.mark.parametrize("bend", [lambda left, right, s, axis: ((1., 0.),),
                                  lambda left, right, s, axis: ((1.,), (0.,))])
def test_bend_must_preserve_the_full_state_shape(bend):
    _, _, _, product, path = declarations()
    with pytest.raises(ValueError, match="square matrix"):
        SymbolicPath(product, frame=path.frame, quadrature=((0., 1.),),
                     speed=lambda left, right, axis: 1., bend=bend)
