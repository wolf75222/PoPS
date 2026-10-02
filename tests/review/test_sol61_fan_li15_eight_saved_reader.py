"""Offline synthetic probes, never receipts of Native runs."""
import sys
import numpy as np
import pytest
from tests.review import sol61_fan_li15_eight_saved_reader as r

def test_wick_gram_flux_independent_of_generating_function():
    for raw in r.initial()[:,::5].T:
        for direction in ((1.,0.),(0.,1.),(-.75,.5)):
            gram=r.gram_flux(raw,direction)
            primary=np.array(r.primary.derivative_gaussian_flux(raw,direction))
            np.testing.assert_allclose(gram,primary,rtol=0,atol=2e-12)
    theta=np.array([[.8,.17],[.17,1.2]])
    assert r.wick(theta,2,2)==pytest.approx(.8*1.2+2*.17**2)
    assert r.wick(theta,3,0)==0
    assert not any(name=='_pops' or name.startswith('pops.') for name in sys.modules)

def test_domain_and_terminal_support_mutants():
    q=r.initial()[:,0]
    for density in (0.,-1.,np.nan):
        bad=q.copy();bad[0]=density
        with pytest.raises(ValueError):r.domain(bad)
    bad=q.copy();bad[6]=100.
    with pytest.raises(ValueError,match='SPD'):r.domain(bad)
    product=r.path(q,r.initial()[:,1],24)
    assert np.array_equal(product[list(r.LOW)],np.zeros(10))
    assert np.max(abs(product[list(r.TOP)]))>1e-8

def fake_states():
    q=np.broadcast_to(r.initial()[:,None,:],(15,r.N,r.N)).copy()
    return [q.copy() for _ in range(9)],[{'time':float(k*r.DT),'tick':k} for k in range(9)]

@pytest.mark.parametrize('mutant',('missing','duplicate_order','bool_tick','wrong_time','y','nan','initial'))
def test_scope_representation_and_phase_mutants_fail_before_science(mutant):
    states,clocks=fake_states();order=r.INDICES
    if mutant=='missing':states.pop()
    elif mutant=='duplicate_order':order=(order[0],)+order[:-1]
    elif mutant=='bool_tick':clocks[0]['tick']=False
    elif mutant=='wrong_time':clocks[3]['time']=0.
    elif mutant=='y':states[2][0,1,0]+=1e-4
    elif mutant=='nan':states[2][0,0,0]=np.nan
    elif mutant=='initial':states[0][0,:,:]+=1e-4
    with pytest.raises(ValueError):r.audit_states(states,clocks,order)

def test_saved_sequence_cannot_replace_evolution_by_repeated_initial():
    states,clocks=fake_states()
    with pytest.raises(ValueError,match='SSPRK2'):r.audit_states(states,clocks,r.INDICES)

def test_nine_synthetic_states_from_independent_historical_flux_route(tmp_path):
    import importlib.util,json,hashlib
    from pathlib import Path
    path=Path(__file__).resolve().parents[2]/'examples/migration/scientific/api040_m17_oracle.py'
    spec=importlib.util.spec_from_file_location('source_only_historical_m17_math',path)
    oracle=importlib.util.module_from_spec(spec);spec.loader.exec_module(oracle)
    line=r.initial();states=[]
    for k in range(9):
        states.append(np.broadcast_to(line[:,None,:],(15,r.N,r.N)).copy())
        if k<8:
            k0=oracle.semidiscrete_rhs(0.,line.ravel()).reshape(15,r.N)
            first=line+r.DT*k0
            k1=oracle.semidiscrete_rhs(r.DT,first.ravel()).reshape(15,r.N)
            line=line+.5*r.DT*(k0+k1)
    clocks=[{'time':float(k*r.DT),'tick':k} for k in range(9)]
    result=r.audit_states(states,clocks,r.INDICES)
    assert len(result['metrics'])==8 and result['dop853_final_error']<3e-8
    assert result['native_authority'] is result['root_scientific_approval'] is False
    # Reordered arrays authenticate their explicit basis; no first-density guess.
    order=tuple(reversed(r.INDICES))
    reverse=[q[::-1].copy() for q in states]
    pins={};files=[]
    for k,(q,clock) in enumerate(zip(reverse,clocks)):
        state='state'+str(k)+'.npy';timing='clock'+str(k)+'.json'
        np.save(tmp_path/state,q);(tmp_path/timing).write_text(json.dumps([clock['time'],clock['tick']]))
        for name in (state,timing):pins[name]=hashlib.sha256((tmp_path/name).read_bytes()).hexdigest()
        files.append((state,timing))
    assert r.compare_permutation(states,r.INDICES,reverse,order)==[0.]*9
    poisoned=[q.copy() for q in reverse];poisoned[4][0,0,0]+=2e-12
    with pytest.raises(ValueError,match='permutation'):r.compare_permutation(states,r.INDICES,poisoned,order)
    received=r.receive(tmp_path,pins,order,tuple(files))
    assert received['whole_carrier_qualification'] is False
    bad=files[:-1]+[files[-2]]
    with pytest.raises(ValueError,match='duplicate'):r.receive(tmp_path,pins,order,tuple(bad))
    (tmp_path/files[2][1]).write_text('{}')
    with pytest.raises(ValueError,match='pin'):r.receive(tmp_path,pins,order,tuple(files))

def test_failure_domain_keeps_signed_zero_and_clock_authority():
    before=np.zeros((15,r.N,r.N),dtype=np.float64);after=before.copy()
    failure={'phase':'attempt','exception_type':'Source-only synthetic refusal','message':'not a Native witness'}
    assert r.audit_rejected_valid_rollback(before,after,(0.,0),(0.,0),failure)['full_storage_rollback_qualified'] is False
    after[0,0,0]=-0.
    with pytest.raises(ValueError,match='rollback'):r.audit_rejected_valid_rollback(before,after,(0.,0),(0.,0),failure)
    with pytest.raises(ValueError,match='attempted'):r.audit_rejected_valid_rollback(before,before,(0.,0),(0.,0),{**failure,'phase':'bind'})

def test_disabled_regularization_is_a_different_pde_not_an_oracle_fallback():
    seed=r.initial();active=r.step(seed,48);disabled=r.step(seed,48,active=False)
    assert np.max(abs(active-disabled))>3e-8
    states,clocks=fake_states()
    states[1]=np.broadcast_to(disabled[:,None,:],(15,r.N,r.N)).copy()
    with pytest.raises(ValueError,match='SSPRK2'):r.audit_states(states,clocks,r.INDICES)
