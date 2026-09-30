"""Actual Program/Kokkos source composition and FV-ledger feedback reception.

No Python loop evolves the runtime: this independent array oracle reads saved state.
"""
import numpy as np
import pops
import pytest
from tests.python.unit.codegen.test_integral_candidate_capture import build_feedback
from tests.python.integration.runtime.test_integral_state_public_restart import (
    _world, _root, _compile, _bind, _run, _assert_quantity,
)
from tests.python.support.collective_checks import collective_attempt, collective_call, collective_check
from tests.python.support.integral_state_receipts import (
    collective_directory, save_public_snapshot, selected_external_records,
)

pytestmark = [pytest.mark.compiler,pytest.mark.kokkos,pytest.mark.native_loader]
DT,GAMMA,Q0,TOL = .01,.3,.7,3e-13


def _case(cells,**kwargs):
    case,layout,_,quantity,_ = build_feedback(cells=cells,periodic=False,
                                            initial_condition=True,**kwargs)
    initial = 1.+.2*(np.arange(cells)+.5)/cells
    return case,layout,quantity,initial


def _oracle(previous,q,dt):
    reaction = previous*(1.-GAMMA*q*dt)
    left = np.concatenate((reaction[:1],reaction[:-1]))
    result = reaction-dt*len(previous)*(reaction-left)
    delivered = dt*reaction[-1]
    return result,q+delivered,delivered


def _receipt_check(ledgers,quantity,q,delivered):
    records = []
    keys = []
    for ledger in ledgers:
        assert ledger["quantities"][quantity.identity]==(Q0,q)
        records.extend(selected_external_records(ledger))
        keys.extend(ledger["consumed"])
    assert len(records)==1 and len(keys)==1
    record = records[0]
    assert record["orientation"]==-1 and record["multiplicity"]==1
    assert abs(record["measure"]*record["flux"]*record["weight"]-delivered)<TOL
    assert keys[0]==(record["operation"],record["occurrence"],record["context"],record["quadrature"])


@pytest.mark.parametrize("cells",(8,16))
def test_public_feedback_original_source_real_exterior_and_byte_exact_restart(
        isolated_native_cache,native_cxx,kokkos_root,tmp_path,record_property,cells):
    world = _world()
    case,layout,quantity,initial = _case(cells)
    artifact = _compile(world,case,layout)
    directory = collective_directory(world,tmp_path)
    runtime = _bind(world,artifact,initial)
    _,saved0 = save_public_snapshot(world,runtime,artifact,quantity,directory,"initial")
    _run(world,runtime,DT,1)
    checkpoint,saved1 = save_public_snapshot(world,runtime,artifact,quantity,directory,"accepted")
    expected,q1,delivery = _oracle(initial,Q0,DT)
    _assert_quantity(world,runtime,quantity,q1)
    with collective_check(world):
        if _root(world):
            np.testing.assert_allclose(saved1[0].reshape(-1),expected,atol=TOL,rtol=0)
            _receipt_check(saved1[2],quantity,q1,delivery)
            # The reaction loss is independently recomputed from saved pre-step U and q.
            before = saved0[0].reshape(-1)
            reaction_inventory = -GAMMA*Q0*DT*np.mean(before)
            face_change = DT*((1.-GAMMA*Q0*DT)*before[0]-(1.-GAMMA*Q0*DT)*before[-1])
            assert abs(np.mean(saved1[0])-np.mean(before)-reaction_inventory-face_change)<TOL
    _run(world,runtime,2*DT,1)
    _,continuous = save_public_snapshot(world,runtime,artifact,quantity,directory,"continuous")
    restarted = _bind(world,artifact,initial)
    collective_call(world,lambda:restarted.restart(checkpoint))
    _,restored = save_public_snapshot(world,restarted,artifact,quantity,directory,"restored")
    with collective_check(world):
        assert restarted.time()==DT and restarted.macro_step()==1
        if _root(world):
            np.testing.assert_array_equal(restored[0],saved1[0])
            assert restored[1]==saved1[1]
    _run(world,restarted,2*DT,1)
    _,replayed = save_public_snapshot(world,restarted,artifact,quantity,directory,"replayed")
    expected2,q2,delivery2 = _oracle(expected,q1,DT)
    _assert_quantity(world,restarted,quantity,q2)
    with collective_check(world):
        if _root(world):
            np.testing.assert_allclose(replayed[0].reshape(-1),expected2,atol=TOL,rtol=0)
            np.testing.assert_array_equal(replayed[0],continuous[0])
            assert replayed[1]==continuous[1]
            _receipt_check(replayed[2],quantity,q2,delivery2)
    record_property("feedback_receipts",str(directory))
    record_property("artifact_identity",artifact.artifact_identity.token)
    record_property("dim",2)
    record_property("rank",0 if world is None else int(world.rank))
    record_property("size",1 if world is None else int(world.size))


def test_feedback_rejected_transport_preserves_capture_integral_and_safe_retry(
        isolated_native_cache,native_cxx,kokkos_root,tmp_path):
    world = _world()
    case,layout,quantity,initial = _case(8,proposed_dt=.5)
    artifact = _compile(world,case,layout)
    directory = collective_directory(world,tmp_path)
    runtime = _bind(world,artifact,initial)
    _,before = save_public_snapshot(world,runtime,artifact,quantity,directory,"before")
    _,errors = collective_attempt(world,lambda:pops.run(runtime,t_end=.5,max_steps=1,console=False))
    with collective_check(world):
        assert all(error and error[2] and "user_face_numerical_stability" in error[1]
                   for error in errors),errors
        assert runtime.time()==0. and runtime.macro_step()==0
        assert runtime.integral_state(quantity)==Q0
    _,rejected = save_public_snapshot(world,runtime,artifact,quantity,directory,"rejected")
    with collective_check(world):
        if _root(world):
            np.testing.assert_array_equal(rejected[0],before[0])
            assert rejected[1]==before[1]
    _run(world,runtime,DT,1)
    _,retried = save_public_snapshot(world,runtime,artifact,quantity,directory,"retried")
    expected,q,_ = _oracle(initial,Q0,DT)
    _assert_quantity(world,runtime,quantity,q)
    with collective_check(world):
        if _root(world):
            np.testing.assert_allclose(retried[0].reshape(-1),expected,atol=TOL,rtol=0)
