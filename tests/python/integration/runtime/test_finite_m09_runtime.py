"""Real loader reception: finite W06 DOFs in a singleton native batch, not a PDE."""
from dataclasses import replace
import json
import numpy as np
import pops
import pytest
from tests.python.integration.runtime.test_user_numerical_bodies_runtime import _compile, _root_check
from tests.python.support.collective_checks import (
    collective_call, collective_attempt, collective_check, state_snapshots,
)
from tests.python.support.native_execution_context import artifact_execution_context
from tests.python.support.finite_m09_case import case_module, oracle

pytestmark=[pytest.mark.compiler,pytest.mark.kokkos,pytest.mark.native_loader]


def _run(runtime,world):
    _, failures = collective_attempt(world, lambda: pops.run(
        runtime, t_end=case_module.DT, max_steps=1, console=False))
    return failures


@pytest.mark.parametrize("condensed",(False,True))
@pytest.mark.parametrize("permuted",(False,True))
def test_finite_w06_original_equations_and_reconstruction(isolated_native_cache,native_cxx,
        kokkos_root,tmp_path,record_property,condensed,permuted):
    del isolated_native_cache,native_cxx,kokkos_root
    data=oracle.witness()
    case,layout,subjects,(vo,po)=case_module.build_case(data,condensed=condensed,permuted=permuted)
    artifact,world=_compile(case,layout,"finite-w06-%d-%d"%(condensed,permuted))
    context=collective_call(world, lambda: artifact_execution_context(artifact))
    with collective_check(world):
        initial = {subjects[0]:data.velocity_old[list(vo)].reshape(8,1,1),
                   subjects[1]:data.potential_old[list(po)].reshape(4,1,1)}
    runtime=collective_call(world, lambda: pops.bind(artifact, initial_values=initial,
                           resources={"execution_context":context}))
    errors=_run(runtime,world)
    with collective_check(world):
        assert not any(errors),errors
        assert runtime.time()==case_module.DT and runtime.macro_step()==1
    raw_v,raw_p=state_snapshots(runtime,world,("velocity","potential"))
    def equations():
        velocity=raw_v.reshape(8)[np.argsort(vo)]
        potential=raw_p.reshape(4)[np.argsort(po)]
        path=tmp_path/("finite-w06-%d-%d.npz"%(condensed,permuted))
        np.savez_compressed(path,velocity=velocity,potential=potential,
                            velocity_old=data.velocity_old,potential_old=data.potential_old)
        with np.load(path) as saved:
            actual=saved["velocity"],saved["potential"]
            residual=oracle.original_residual(data,*actual)
            reference=oracle.monolithic(data)
            difference=max(np.max(np.abs(a-b)) for a,b in zip(actual,reference,strict=True))
            end=oracle.endpoint(data,actual)
            energy=abs(oracle.energy(data,*end)-oracle.energy(data,data.velocity_old,data.potential_old))
            assert max(residual,difference,energy)<oracle.MAX_ERROR
            record_property("finite_original_metrics",json.dumps(dict(residual=residual,
                monolithic_difference=float(difference),cn_energy=energy)))
    _root_check(world,equations)
    with collective_check(world):
        record_property("mpi_ranks",1 if world is None else int(world.size))


@pytest.mark.parametrize("condensed",(False,True))
def test_singular_elimination_pivot_is_not_a_monolithic_singularity(isolated_native_cache,
        native_cxx,kokkos_root,condensed,record_property):
    del isolated_native_cache,native_cxx,kokkos_root
    # Null(A) has dimension four, coupled by the full-rank B/C. The complete
    # 12x12 system is nonsingular, so rejecting it merely because A is singular
    # would make Schur an unjustified universal solver assumption.
    data=replace(oracle.witness(),a=np.diag([0.]*4+[1.]*4))
    assert np.linalg.matrix_rank(np.block([[data.a,data.b],[data.c,data.k]]))==12
    case,layout,subjects,_=case_module.build_case(data,condensed=condensed)
    artifact,world=_compile(case,layout,"finite-singular-pivot-%d"%condensed)
    context=collective_call(world, lambda: artifact_execution_context(artifact))
    with collective_check(world):
        initial={subjects[0]:data.velocity_old.reshape(8,1,1),
                 subjects[1]:data.potential_old.reshape(4,1,1)}
    runtime=collective_call(world, lambda: pops.bind(artifact, initial_values=initial,
                           resources={"execution_context":context}))
    before=state_snapshots(runtime,world,("velocity","potential"))
    errors=_run(runtime,world)
    with collective_check(world):
        if condensed:
            assert all(e is not None and e[2] for e in errors),errors
            assert runtime.time()==0 and runtime.macro_step()==0
        else:
            assert not any(errors),errors
            assert runtime.time()==case_module.DT and runtime.macro_step()==1
    after=state_snapshots(runtime,world,("velocity","potential"))
    if condensed:
        def unchanged():
            for a,b in zip(before,after,strict=True):
                np.testing.assert_array_equal(a,b)
        _root_check(world,unchanged)
        with collective_check(world):
            record_property("native_pivot_refusal",repr(errors))
    else:
        def original():
            assert oracle.original_residual(data,after[0].reshape(8),after[1].reshape(4))<oracle.MAX_ERROR
        _root_check(world,original)
