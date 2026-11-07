"""Real finite-DOF operands/rebinds and owner-only failure with empty MPI ranks."""
from dataclasses import replace
import json

import numpy as np
import pops
import pytest
from tests.python.integration.runtime.test_user_numerical_bodies_runtime import _compile, _root_check
from tests.python.support.collective_checks import (
    collective_attempt, collective_call, collective_check, state_snapshots,
)
from tests.python.support.native_execution_context import artifact_execution_context
from tests.python.support.finite_m09_case import case_module, oracle

pytestmark=[pytest.mark.compiler,pytest.mark.kokkos,pytest.mark.native_loader]


def _operands(data,index):
    velocity=data.velocity_old*(1+.125*index)+np.linspace(-.2,.1,8)*index
    potential=data.potential_old*(1-.2*index)+np.linspace(.15,-.1,4)*index
    return replace(data,f=velocity, g=data.k@potential,velocity_old=velocity,potential_old=potential)


@pytest.mark.parametrize("condensed",(False,True))
@pytest.mark.parametrize("seed_scale",(0,.375))
def test_finite_maps_rebind_operands_and_reject_owner_nonfinite_collectively(
        isolated_native_cache,native_cxx,kokkos_root,tmp_path,record_property,condensed,seed_scale):
    del isolated_native_cache,native_cxx,kokkos_root
    data=oracle.witness()
    case,layout,subjects,(vo,po)=case_module.build_case(data,condensed=condensed,
                                                       permuted=True,seed_scale=seed_scale)
    artifact,world=_compile(case,layout,"finite-adversarial-%s-%s"%(condensed,seed_scale))
    context=collective_call(world,lambda:artifact_execution_context(artifact))
    with collective_check(world):
        identity=artifact.artifact_identity.token

    def bind(velocity,potential):
        with collective_check(world):
            initial={subjects[0]:velocity[list(vo)].reshape(8,1,1).copy(),
                     subjects[1]:potential[list(po)].reshape(4,1,1).copy()}
        return collective_call(world,lambda:pops.bind(artifact,initial_values=initial,
                                resources={"execution_context":context}))

    def accepted(index,label):
        with collective_check(world):
            sample=_operands(data,index)
        runtime=bind(sample.velocity_old,sample.potential_old)
        collective_call(world,lambda:pops.run(runtime,t_end=case_module.DT,max_steps=1,console=False))
        with collective_check(world):
            assert artifact.artifact_identity.token==identity
            assert runtime.time()==case_module.DT and runtime.macro_step()==1
        raw_v,raw_p=state_snapshots(runtime,world,("velocity","potential"))
        def original():
            values=raw_v.reshape(8)[np.argsort(vo)],raw_p.reshape(4)[np.argsort(po)]
            path=tmp_path/(label+".npz")
            np.savez_compressed(path,velocity=values[0],potential=values[1],
                                old_v=sample.velocity_old,old_p=sample.potential_old)
            with np.load(path) as saved:
                actual=saved["velocity"],saved["potential"]
                reference=oracle.monolithic(sample)
                residual=oracle.original_residual(sample,*actual)
                error=max(float(np.max(np.abs(a-b))) for a,b in zip(actual,reference,strict=True))
                assert max(residual,error)<oracle.MAX_ERROR
                record_property(label,json.dumps(dict(original_residual=residual,oracle_error=error)))
        _root_check(world,original)

    accepted(0,"first_capture")
    accepted(1,"second_capture_same_artifact")
    # All bound numbers are finite. K's diagonal times this finite potential
    # overflows during the ORIGINAL residual evaluation, only on the real owner.
    with collective_check(world):
        bad_v=np.zeros(8)
        bad_p=np.full(4,1.e308)
        assert np.isfinite(bad_v).all() and np.isfinite(bad_p).all()
        assert max(np.diag(data.k))>2
    failed=bind(bad_v,bad_p)
    local=collective_call(world,lambda:failed.local_boxes("velocity"))
    local_p=collective_call(world,lambda:failed.local_boxes("potential"))
    if world is None:
        owned=((local,local_p),)
    else:
        from pops._native_collectives import allgather_value
        owned=allgather_value(world,(local,local_p))
    with collective_check(world):
        assert all(a==b for a,b in owned),owned
        assert sum(len(a) for a,_ in owned)==1,owned
        if world is not None and world.size>1:
            assert any(not a for a,_ in owned),owned
        record_property("actual_rank_owned_boxes",repr(owned))
    before=state_snapshots(failed,world,("velocity","potential"))
    for attempt in range(2):
        _,failures=collective_attempt(world,lambda:pops.run(
            failed,t_end=case_module.DT,max_steps=1,console=False))
        with collective_check(world):
            assert all(row is not None and row[2] for row in failures),failures
            assert all("invalid_evaluation" in row[1] for row in failures),failures
            assert failed.time()==0 and failed.macro_step()==0
            record_property("owner_only_failure_%d"%attempt,repr(failures))
        after=state_snapshots(failed,world,("velocity","potential"))
        def unchanged(after=after):
            for old,new in zip(before,after,strict=True):
                np.testing.assert_array_equal(old,new)
        _root_check(world,unchanged)
    # This is a NEW bind of the same artifact, not a repaired failed runtime.
    accepted(2,"new_bind_after_two_refusals")
    with collective_check(world):
        record_property("mpi_ranks",1 if world is None else int(world.size))
