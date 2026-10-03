import ast
from pathlib import Path
import numpy as np
import pytest
from tests.python.support import m19_vlasov_poisson_oracle as oracle


def test_two_steps_discriminate_force_sign_and_stage_refresh():
    f=oracle.initial(); one,stage=oracle.step(f); two,_=oracle.step(one)
    wrong,_=oracle.step(f,force_sign=-1.); stale,_=oracle.step(f,stale_predictor=True)
    assert np.min(two)>0
    assert abs(two.sum()/16-1)<1e-14
    assert abs(stage['initial_electric']).max()>0
    assert abs(stage['predictor_electric']-stage['initial_electric']).max()>1e-8
    assert abs(one-wrong).max()>1e-6
    assert abs(one-stale).max()>1e-8
    assert abs(two-f).max()>1e-4
    assert abs(stage['initial_residual']).max()<1e-12
    assert abs(stage['predictor_residual']).max()<1e-12


def test_periodic_neutrality_not_projected():
    with pytest.raises(ValueError,match='not neutral'):
        oracle.electrostatics(oracle.initial()*1.01)


def test_reference_imports_no_pops():
    tree=ast.parse(Path(oracle.__file__).read_text())
    for node in ast.walk(tree):
        if isinstance(node,ast.ImportFrom):
            assert not node.module.startswith('pops')
        if isinstance(node,ast.Import):
            assert all(not alias.name.startswith('pops') for alias in node.names)
