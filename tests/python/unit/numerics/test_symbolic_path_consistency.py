"""Independent path consistency and identity checks from the Astra review."""

import pytest
import pops
from pops.domain import Rectangle
from pops.frames import Cartesian2D
from pops.numerics import SymbolicPath


def constant_matrix_path():
    frame = Rectangle("consistency_box", lower=(0., 0.), upper=(1., 1.)).frame(Cartesian2D())
    model = pops.Model("constant_matrix", frame=frame)
    state = model.state("U", components=("u", "v"))
    product = model.nonconservative_product("mixed_matrix", state=state,
        matrices={frame.axes[0]: ((2., -1.), (0., 0.)),
                  frame.axes[1]: ((0., 0.), (0., 0.))},
        conservative_components=("v",))
    path = SymbolicPath(product, frame=frame, quadrature=((.5, 1.),),
                        speed=lambda left, right, axis: 3.,
                        bend=lambda left, right, s, axis: ((s, 2 * s), (0., s)))
    return path


def evaluate(path, left, right):
    env = {symbol.name: value for symbols, values in
           ((path.left_symbols, left), (path.right_symbols, right))
           for symbol, value in zip(symbols, values, strict=True)}
    return tuple(expression.eval(env) for expression in path.integrals[0])


@pytest.mark.parametrize("state", [(0., 0.), (-3., 7.), (1.e100, -1.e100)])
def test_equal_states_make_no_product_even_with_coordinate_dependent_bending(state):
    path = constant_matrix_path()
    assert evaluate(path, state, state) == (0., 0.)


@pytest.mark.parametrize("left,right", [((0., 0.), (1., 0.)), ((2., 3.), (5., 7.)),
                                        ((2., 3.), (2. + 1.e-6, 3. - 1.e-6))])
def test_quadrature_cannot_change_a_constant_physical_matrix(left, right):
    # Direct midpoint quadrature of the curved tangent gives a 25% error for
    # K(s)=s and one scalar jump. It therefore changes the PDE at first order.
    # Exact integration of B(left)*jump must retain the actual constant matrix.
    path = constant_matrix_path()
    expected = 2 * (right[0] - left[0]) - (right[1] - left[1])
    assert evaluate(path, left, right) == pytest.approx((expected, 0.), rel=1.e-14, abs=0.)


def test_coordinate_substitution_uses_identity_and_preserves_shared_nodes():
    from pops._ir.application import substitute_quantities
    from pops._ir.expr import Const, Var
    coordinate = Var("same", "coordinate")
    unrelated = Var("same", "coordinate")
    shared = coordinate + unrelated
    replacement = Const(.5)
    first, second = substitute_quantities((shared, shared), {},
                                         expression_bindings={id(coordinate): replacement})
    assert first is second
    assert first.a is replacement
    assert first.b is not replacement
    assert first.b.name == unrelated.name


def test_path_only_runtime_speed_parameter_reaches_native_member():
    from pops.codegen.module_codegen import _emit_bricks
    from pops.codegen.module_lowering import lower_and_validate
    from pops.layouts import Uniform
    from pops.lib.time import ForwardEuler
    from pops.math import ddt, div
    from pops.mesh import CartesianGrid, PeriodicAxes
    from pops.numerics import (DiscretizationPlan, PathConservativeFiniteVolume,
                               reconstruction, riemann, variables)
    from pops.params import RuntimeParam
    from pops.time import AdaptiveCFL
    from tests.python.support.symbolic_path_case import declarations, SIMPSON
    model, state, flux, product, original = declarations()
    coefficient = model.value(model.param(RuntimeParam("path_speed_only", default=100.)))
    path = SymbolicPath(product, frame=original.frame, quadrature=SIMPSON,
                        speed=lambda left, right, axis: coefficient)
    rate = model.rate("balance", equation=ddt(state) == -div(flux) - product)
    method = PathConservativeFiniteVolume(flux=flux, path=path,
        variables=variables.Conservative(state), reconstruction=reconstruction.FirstOrder(),
        riemann=riemann.Rusanov())
    plan = DiscretizationPlan()
    plan.rates.add(rate, method)
    case = pops.Case("runtime_speed_path")
    block = case.block("transport", model)
    case.numerics(plan, block=block)
    program = ForwardEuler(block[state], rate=rate)
    program.step_strategy(AdaptiveCFL(cfl=.25, max_dt=1.e-3))
    case.program(program)
    layout = Uniform(CartesianGrid(frame=path.frame, cells=(4, 4),
                                   periodic=PeriodicAxes(path.frame.axes)))
    resolved = pops.resolve(pops.validate(case), layout=layout)
    native_block = resolved.blocks[0]
    emitter, _ = lower_and_validate(native_block.model, state_space=native_block.state_spaces[0],
        resolved_operations=native_block.resolved_operations, numerics=native_block.numerics)
    source = _emit_bricks(emitter._m)[1]
    assert "params.get(" in source
    assert "RuntimeParams params" in source
