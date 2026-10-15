"""Actual spatial BDF2, independent Fourier oracle, AMR conservation and restart."""
import numpy as np
import pops
import pytest
from examples.api_v02.method_spatial_bdf2 import build_spatial_bdf2
from tests.python.support.api_v02_native import _bind, _compile
from tests.python.support.collective_checks import collective_call, collective_check
from tests.python.support.integral_state_receipts import collective_directory

pytestmark = [pytest.mark.compiler, pytest.mark.kokkos, pytest.mark.native_loader]
POPS_PROCESS_TIMEOUT = 1200


def fourier_reference(initial, h, steps):
    """Solve each independently derived periodic Laplacian eigenmode in time.

    This is only an oracle over gathered native data: the evolution under test
    is the generated Python residual executed by the real C++ spatial Newton.
    """
    ny,nx = initial.shape
    decay = .4*((nx*np.sin(np.pi*np.fft.fftfreq(nx)))**2)[None,:]
    decay = decay + .4*((ny*np.sin(np.pi*np.fft.fftfreq(ny)))**2)[:,None]
    prior = np.fft.fft2(initial)
    current = (3-h*decay)/(3+2*h*decay)*prior
    for _ in range(1, steps):
        prior,current = current,(4*current-prior)/(3+2*h*decay)
    exact = np.fft.fft2(initial)*np.exp(-steps*h*decay)
    return tuple(np.fft.ifft2(value).real for value in (current, prior, exact))


def _uniform_run(world, spec, artifact, steps):
    runtime = _bind(spec, artifact, world)
    initial = collective_call(world, lambda: runtime.state_global("method"))
    collective_call(world, lambda: pops.run(runtime, t_end=spec.t_end,
        max_steps=steps, console=False))
    actual = collective_call(world, lambda: runtime.state_global("method"))
    names = runtime.history_names()
    with collective_check(world): assert len(names) == 1
    retained = collective_call(world, lambda: runtime.history_global(names[0], 1))
    return runtime, initial, actual, retained


def test_global_spatial_bdf2_matches_equations_and_second_order(
        isolated_native_cache,native_cxx,kokkos_root,record_property):
    del isolated_native_cache,native_cxx,kokkos_root
    errors=[]; identities=[]
    for h,steps in ((.005,16),(.0025,32)):
        spec,artifact,world = _compile("T08_global_spatial_BDF2", builder=build_spatial_bdf2,
            dt=h, steps=steps)
        runtime,initial,actual,retained = _uniform_run(world,spec,artifact,steps)
        with collective_check(world):
            assert runtime.macro_step() == steps
            assert abs(runtime.time()-.08) < 2.e-13
            if actual is not None:
                a=np.asarray(actual).reshape(12,12); start=np.asarray(initial).reshape(a.shape)
                reference,previous,semidiscrete = fourier_reference(start,h,steps)
                np.testing.assert_allclose(a,reference,rtol=0,atol=3.e-10)
                np.testing.assert_allclose(np.asarray(retained).reshape(a.shape),previous,rtol=0,atol=3.e-10)
                assert abs(np.mean(a)-np.mean(start)) < 3.e-11
                errors.append(float(np.max(abs(a-semidiscrete))))
        identities.append(artifact.artifact_identity.hexdigest)
    with collective_check(world):
        if errors:
            assert errors[1] > 1.e-7
            assert 3.2 < errors[0]/errors[1] < 4.8, errors
    record_property("artifact_identities",str(identities))
    record_property("semidiscrete_errors",str(errors))


def test_spatial_bdf2_rejects_current_state_as_previous_lag(
        isolated_native_cache,native_cxx,kokkos_root):
    del isolated_native_cache,native_cxx,kokkos_root
    spec,artifact,world = _compile("T08_wrong_spatial_BDF2",builder=build_spatial_bdf2,
        history_control=True)
    _,initial,actual,_ = _uniform_run(world,spec,artifact,8)
    with collective_check(world):
        if actual is not None:
            a=np.asarray(actual).reshape(12,12)
            reference,_,_=fourier_reference(np.asarray(initial).reshape(a.shape),.01,8)
            assert np.max(abs(a-reference)) > .01


def test_partial_amr_spatial_bdf2_preserves_mass_and_full_restart(
        tmp_path,isolated_native_cache,native_cxx,kokkos_root,record_property):
    del isolated_native_cache,native_cxx,kokkos_root
    from tests.python.integration.runtime.test_api_v02_history_transfer_native import _image,_same
    spec,artifact,world = _compile("T08_partial_spatial_BDF2",builder=build_spatial_bdf2,
        hierarchy="partial",dt=.005,steps=8)
    runtime = _bind(spec,artifact,world)
    mass = collective_call(world,lambda:runtime.integral("method",component=0))
    collective_call(world,lambda:pops.run(runtime,t_end=.02,max_steps=4,console=False))
    image = _image(runtime,world)
    with collective_check(world):
        cells=sum((hi[0]-lo[0]+1)*(hi[1]-lo[1]+1) for level,lo,hi in image["patches"] if level==1)
        assert 0 < cells < 24**2, "the receiver must include actual coarse/fine interfaces"
    directory = collective_directory(world,tmp_path)
    checkpoint = collective_call(world,lambda:runtime.checkpoint(str(directory/"spatial_bdf2")))
    restarted = _bind(spec,artifact,world)
    collective_call(world,lambda:restarted.restart(checkpoint))
    _same(image,_image(restarted,world),world)
    for active in (runtime,restarted):
        collective_call(world,lambda active=active:pops.run(active,t_end=.04,max_steps=4,console=False))
        final_mass=collective_call(world,lambda active=active:active.integral("method",component=0))
        with collective_check(world): assert abs(final_mass-mass) < 3.e-11
    _same(_image(runtime,world),_image(restarted,world),world)
    record_property("artifact_identity",artifact.artifact_identity.hexdigest)
