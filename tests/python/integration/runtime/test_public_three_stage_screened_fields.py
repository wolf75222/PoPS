"""Three public stage-field witnesses; no mathematical recipe in compiler/core.

Tests use an unchanged non-author NumPy oracle. Passing these author-written
assertions is distinct from a separate non-author reception of retained outputs.
"""
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import numpy as np
import pytest
from tests.python.support.stage_fields_public_fixture import capture
from tests.python.support.stage_fields_independent_reference import Physics,check_arrays,tolerances

pytestmark=[pytest.mark.compiler,pytest.mark.native_loader]
BUDGET=Path(__file__).resolve().parents[2]/'support/stage_fields_pre_native_budget.json'
BUDGET_SHA='bed47b146393f5b94bb37ee6a57684985951b769773ee0f956ed34dd84129cd3'


def private_cache(patch,root):
    root.mkdir()
    patch.setenv('POPS_CACHE_DIR',str(root/'jit'))
    patch.setenv('POPS_NATIVE_CACHE_DIR',str(root/'jit'))
    patch.setenv('POPS_CODEGEN_DIR',str(root/'generated'))
    patch.setenv('POPS_KEEP_GENERATED','1')
    patch.delenv('POPS_INCLUDE',raising=False)


@pytest.fixture(scope='module')
def representative(tmp_path_factory,native_cxx,kokkos_root):
    del native_cxx,kokkos_root
    root=tmp_path_factory.mktemp('three-stage-screened-fields')
    assert hashlib.sha256(BUDGET.read_bytes()).hexdigest()==BUDGET_SHA
    frozen=json.loads(BUDGET.read_text())
    assert json.loads(json.dumps(tolerances(Physics())))==frozen
    (root/'pre-execution-independent-budget.json').write_bytes(BUDGET.read_bytes())
    with pytest.MonkeyPatch.context() as patch:
        private_cache(patch,root/'private')
        return capture('representative',root/'representative')


@pytest.mark.parametrize('variant',('representative','rename','decay'))
def test_public_three_stage_screened_fields(variant,representative,tmp_path,monkeypatch,record_property):
    assert hashlib.sha256(BUDGET.read_bytes()).hexdigest()==BUDGET_SHA
    frozen=json.loads(BUDGET.read_text())
    # Baseline budget remains exactly the pre-Native independent budget. Changed
    # decay uses the same independent derivation, fixed before its execution.
    assert json.loads(json.dumps(tolerances(Physics())))==frozen
    physics=replace(Physics(),decay=1.1) if variant=='decay' else Physics()
    budget=tolerances(physics)
    if variant=='representative':
        actual=representative
    else:
        private_cache(monkeypatch,tmp_path/'private')
        (tmp_path/'pre-execution-independent-budget.json').write_text(json.dumps(budget,indent=2)+'\n')
        actual=capture(variant,tmp_path/variant)
    comparisons=check_arrays(actual.arrays,physics)
    if variant=='rename':
        assert set(actual.arrays)==set(representative.arrays)
        for name in actual.arrays:
            left,right=actual.arrays[name],representative.arrays[name]
            assert left.dtype==right.dtype and left.shape==right.shape and left.tobytes()==right.tobytes(),name
    elif variant=='decay':
        assert actual.arrays['d_final'].tobytes()==representative.arrays['d_final'].tobytes()
        difference=float(np.max(np.abs(actual.arrays['qfinal']-representative.arrays['qfinal'])))
        assert difference>budget['absolute_bounds']['qfinal']+frozen['absolute_bounds']['qfinal']
    (actual.directory/'author-test-comparison.json').write_text(json.dumps({
        'comparison':comparisons,'pre_execution_budget':budget,'original_budget_sha256':BUDGET_SHA,
        'scope':'Author test using separately authored frozen oracle; separate non-author actual-output reception still required'},indent=2)+'\n')
    record_property('stage_fields_saved_outputs',str(actual.directory))
