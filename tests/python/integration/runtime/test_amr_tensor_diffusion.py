"""Conservative composite tensor diffusion: interface, MMS and rejected continuation.

These properties are tested separately from the uniform periodic adjoint energy identity.
"""
import numpy as np
import pops
import pytest
from tests.python.integration.runtime.test_amr_implicit_diffusion import (
    DT, bind, mass, accepted_envelope, accepted_histories, assert_no_second_reflux,
    _conservative_reference_errors,
)
from tests.python.support.amr_snapshots import composite_active_mask

pytestmark = [pytest.mark.compiler, pytest.mark.native_loader]



def _assert_solve_residual(runtime):
    diagnostics = runtime.program_report().diagnostics
    names = [name for name in diagnostics if name.endswith(".residual_norm")]
    assert names, "accepted spatial SolveOutcome must publish its actual residual"
    for name in names:
        reference = diagnostics[name.removesuffix(".residual_norm")+".reference_residual_norm"]
        assert diagnostics[name] <= 1e-12*max(1.,reference)

def test_composite_variable_rotated_tensor_conserves_and_converges(isolated_native_cache, native_cxx, kokkos_root, record_property):
    del isolated_native_cache, native_cxx, kokkos_root
    errors = _conservative_reference_errors("tensor", False, periodic_witness=True)
    record_property("tensor_composite_uniform_reference_errors", errors)
    assert errors[1] < errors[0]/1.5 and errors[2] < errors[1]/1.5


def test_composite_tensor_stationary_mms_full_refinement_sequence(isolated_native_cache, native_cxx, kokkos_root, record_property):
    del isolated_native_cache, native_cxx, kokkos_root
    errors=[]
    for n in (16,32,64):
        runtime=bind(n,kind="tensor_mms",periodic_witness=True)
        mask0=composite_active_mask(runtime,0,refinement_ratio=2)
        assert runtime.n_levels()==2 and mask0.any() and (~mask0).any()
        report=pops.run(runtime,t_end=4*DT,max_steps=4,console=False)
        assert report.accepted_steps==4 and report.rejected_steps==0
        assert_no_second_reflux(runtime)
        _assert_solve_residual(runtime)
        squared=0.
        for level in range(runtime.n_levels()):
            count=n*2**level
            centers=(np.arange(count)+.5)/count
            exact=np.broadcast_to(1+.3*np.cos(2*np.pi*(centers-.5))*np.sinc(1/count),(count,count))
            actual=np.asarray(runtime.block_level_state_global("heat",level)).reshape(exact.shape)
            active=composite_active_mask(runtime,level,refinement_ratio=2)
            squared+=float(np.sum((actual[active]-exact[active])**2))/count**2
        errors.append(squared**.5)
    record_property("tensor_composite_stationary_mms_errors",errors)
    assert errors[1]<errors[0]/1.5 and errors[2]<errors[1]/1.5


def test_composite_tensor_rejection_and_exact_resumption(isolated_native_cache, native_cxx, kokkos_root):
    del isolated_native_cache, native_cxx, kokkos_root
    from pops.time import RejectAttempt
    options=dict(kind="tensor",periodic_witness=True,newton_iterations=1,
                 failure_action=RejectAttempt(statuses=("iteration_limit",)))
    runtime=bind(32,step_dt=16*DT,**options)
    before=accepted_envelope(runtime)
    before_diagnostics=runtime.program_report().diagnostics.copy()
    with pytest.raises(RuntimeError,match="(?i)(spatial|iteration|reject)"):
        pops.run(runtime,t_end=16*DT,max_steps=1,console=False)
    assert accepted_envelope(runtime)==before
    assert runtime.program_report().diagnostics==before_diagnostics
    assert runtime._executor._last_step_transaction_report.action=="reject_attempt"
    endpoint=DT/256
    reference=bind(32,step_dt=endpoint,**options)
    previous_mass=mass(runtime,32)
    for current in (runtime,reference):
        report=pops.run(current,t_end=endpoint,max_steps=1,console=False)
        assert report.accepted_steps==1 and report.rejected_steps==0
        assert_no_second_reflux(current)
        _assert_solve_residual(current)
    assert abs(mass(runtime,32)-previous_mass)<5e-11
    for level in range(runtime.n_levels()):
        np.testing.assert_array_equal(runtime.block_level_state_global("heat",level),
                                      reference.block_level_state_global("heat",level))
    assert runtime._executor._program_exchange_records()==reference._executor._program_exchange_records()
    assert accepted_histories(runtime)==accepted_histories(reference)

    assert runtime.program_report().diagnostics==reference.program_report().diagnostics


def test_composite_tensor_two_components_conserve_separately_across_actual_interfaces(isolated_native_cache, native_cxx, kokkos_root, record_property):
    del isolated_native_cache, native_cxx, kokkos_root
    errors=[]
    for n in (16,32,64):
        runtime=bind(n,kind="tensor",periodic_witness=True,components=2)
        reference=bind(2*n,kind="tensor",periodic_witness=True,components=2,refined=False)
        mask0=composite_active_mask(runtime,0,refinement_ratio=2)
        assert runtime.n_levels()==2 and mask0.any() and (~mask0).any()
        def inventories(current, n=n):
            result=np.zeros(2)
            for level in range(current.n_levels()):
                count=n*2**level
                values=np.asarray(current.block_level_state_global("heat",level)).reshape(2,count,count)
                active=composite_active_mask(current,level,refinement_ratio=2)
                result+=values[:,active].sum(axis=1)/count**2
            return result
        before=inventories(runtime)
        for current in (runtime,reference):
            report=pops.run(current,t_end=4*DT,max_steps=4,console=False)
            assert report.accepted_steps==4 and report.rejected_steps==0
            _assert_solve_residual(current)
        np.testing.assert_allclose(inventories(runtime),before,rtol=0,atol=5e-11)
        assert_no_second_reflux(runtime)
        fine=np.asarray(reference.state_global("heat")).reshape(2,2*n,2*n)
        coarse=fine.reshape(2,n,2,n,2).mean(axis=(2,4))
        squared=0.
        for level,expected in ((0,coarse),(1,fine)):
            active=composite_active_mask(runtime,level,refinement_ratio=2)
            values=np.asarray(runtime.block_level_state_global("heat",level)).reshape(expected.shape)
            squared+=float(np.sum((values[:,active]-expected[:,active])**2))/(n*2**level)**2
        errors.append(squared**.5)
    record_property("tensor_two_component_interface_reference_errors",errors)
    assert errors[1]<errors[0]/1.5 and errors[2]<errors[1]/1.5
