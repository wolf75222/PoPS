"""A path body uses the physical product and ordinary expression lowering."""
import pytest
import pops
from pops.codegen.module_codegen import _emit_bricks
from pops.codegen.module_lowering import lower_and_validate
from pops.numerics import SymbolicPath
from tests.python.support.symbolic_path_case import declarations, make_case, SIMPSON


@pytest.mark.parametrize("scale,reverse", [(1., False), (2.5, True)])
def test_path_body_integrates_the_declared_product_with_permuted_coordinates(scale, reverse):
    _, state, _, _, path = declarations(scale=scale, reverse=reverse)
    left, right = ((2., 3.), (5., 7.))
    if reverse:
        left, right = left[::-1], right[::-1]
    env = {symbol.name: value for symbols, values in
           ((path.left_symbols, left), (path.right_symbols, right))
           for symbol, value in zip(symbols, values, strict=True)}
    actual = [expression.eval(env) for expression in path.integrals[0]]
    expected = scale * (3.**2 + 3.*7. + 7.**2) / 3. * (5. - 2.)
    assert actual[state.components.index("u")] == pytest.approx(expected)
    assert actual[state.components.index("v")] == 0.
    assert path.speeds[0].eval(env) == pytest.approx(0.5 + abs(scale)*49.)


def test_foreign_quantity_cannot_be_captured_as_an_interface_value():
    model, _, _, product, path = declarations()
    foreign = pops.Model("different", frame=model.frame).state("U", components=("u", "v"))
    with pytest.raises(NotImplementedError, match="captured quantities"):
        SymbolicPath(product, frame=path.frame, quadrature=SIMPSON,
                     speed=lambda left, right, axis: foreign[0])


def test_public_path_reaches_the_existing_native_brick_without_a_model_recipe():
    case, layout = make_case()
    resolved = pops.resolve(pops.validate(case), layout=layout)
    block = resolved.blocks[0]
    emitter, _ = lower_and_validate(block.model, state_space=block.state_spaces[0],
        resolved_operations=block.resolved_operations, numerics=block.numerics)
    source = _emit_bricks(emitter._m)[1]
    assert "path_conservative = true" in source
    assert "pops_path_left_0" in source
    assert "PathIntegralResult<n_vars>" in source
    assert "normalized_moment_path" not in source
    assert "FanLi" not in source
