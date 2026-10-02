"""Non-author Source authority probes; no Native qualification."""
import pytest
from pops._ir.expr import Const
from pops.model import FluxWaveLaw
from pops.model.flux_waves import flux_waves, common_flux_waves
from tests.review.test_flux_wave_authority import make


def ops(model):
    return [op for op in model.module.operator_registry() if op.kind == 'grid_operator']


def test_foreign_registry_same_spelling_refused():
    left, *_ = make(); right, *_ = make()
    assert ops(left)[0].name == ops(right)[0].name
    with pytest.raises(ValueError, match='registered'):
        flux_waves(left.module, ops(right)[0])


def test_axes_are_canonical_and_input_alias_detached():
    model, state, _ = make()
    source = {'y': (Const(2.),)*2, 'x': (Const(2.),)*2}
    law = FluxWaveLaw(state.space, source)
    expected = ops(model)[0].lowering['flux_wave_law'].to_data()
    assert law.to_data() == expected
    source['x'] = (Const(999.),)*2
    assert law.to_data() == expected
    with pytest.raises(TypeError): law.values['x'] = ()


def test_joint_authority_does_not_become_selected_state_authority():
    model, *_ = make(); names = [op.name for op in ops(model)]
    model.module.eigenvalues(x=(Const(11.),), y=(Const(11.),))
    for selected in (names, names[::-1]):
        with pytest.raises(ValueError, match='ambiguous'):
            common_flux_waves(model.module, selected)
        assert common_flux_waves(model.module, selected, global_authority=True) is model.module._eigenvalues
    assert flux_waves(model.module, ops(model)[0])['x'][0].value == 2.


def test_mixed_explicit_and_legacy_requires_exact_common_bound():
    model, *_ = make(); operators = ops(model)
    del operators[1].lowering['flux_wave_law']
    names = [op.name for op in operators]
    with pytest.raises(ValueError, match='ambiguous'):
        common_flux_waves(model.module, names)
    model.module.eigenvalues(x=(Const(2.),)*2, y=(Const(2.),)*2)
    assert common_flux_waves(model.module, names) == flux_waves(model.module, operators[0])
    model.module.eigenvalues(x=(Const(3.),)*2, y=(Const(3.),)*2)
    with pytest.raises(ValueError, match='ambiguous'):
        common_flux_waves(model.module, names)
