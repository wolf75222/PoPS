"""Independent discrete Fourier oracle for a real retained FieldProblem solve."""
import numpy as np
import pops
import pytest
from examples.api_v02.continuation_independent_uniform import build_independent_uniform
from tests.python.support.api_v02_receipts import _decode, _image
from tests.python.support.api_v02_native import _bind, _compile
from tests.python.support.collective_checks import collective_attempt, collective_call, collective_check
from tests.python.support.integral_state_receipts import collective_directory

pytestmark = [pytest.mark.compiler, pytest.mark.kokkos, pytest.mark.native_loader]


def _verify(spec, runtime, world, step):
    image = collective_call(world, lambda: _image(runtime))
    decoded = collective_call(world, lambda: _decode(spec, image,
        rank=0 if world is None else int(world.rank)))
    scale = 1.+50.*spec.dt*(step-1)
    expected_previous = 64*(1.+50.*spec.dt*max(0, step-2))
    with collective_check(world):
        root = decoded["root"]
        assert root["step"] == step
        assert root["time"] == pytest.approx(step*spec.dt, rel=0, abs=2e-16)
        assert root["previous"] == pytest.approx(expected_previous, rel=2e-12, abs=2e-11)
        for key, components in (("retained", 1), ("gradient", 2)):
            sample = root[key]
            support = sample["field_support"]
            assert support["program_owner"] == -1 and support["producer_level"] == 0
            assert support["field_provider"] == sample["field_provider"]
            assert bool(sample["field_provider"])
            for ordinal, raw in sample["patches"].items():
                (lx, hx, _), (ly, hy, _) = sample["layout"][ordinal]
                gx, gy = (axis[0] for axis in sample["axes"])
                nx, ny = hx-lx+1, hy-ly+1
                values = raw.reshape(components, ny+2*gy, nx+2*gx)[:,gy:gy+ny,gx:gx+nx]
                x = (np.arange(lx,hx+1)+.5)/8
                amplitude = .1/(1.+.04*4*8**2*np.sin(np.pi/8)**2)
                if key == "retained":
                    expected = np.broadcast_to(scale*(1.+amplitude*np.sin(2*np.pi*x)), (1,ny,nx))
                else:
                    expected = np.zeros_like(values)
                    expected[0] = scale*amplitude*8*np.sin(2*np.pi/8)*np.cos(2*np.pi*x)
                    assert np.max(np.abs(expected[0])) > .01
                np.testing.assert_allclose(values,expected,rtol=2e-11,atol=2e-11)
    return image, decoded


def test_uniform_field_and_gradient_keep_prior_observation_and_restart(
        tmp_path, isolated_native_cache, native_cxx, kokkos_root):
    del isolated_native_cache, native_cxx, kokkos_root
    spec, artifact, world = _compile("independent Uniform Field", builder=build_independent_uniform)
    runtime = _bind(spec,artifact,world)
    collective_call(world,lambda:pops.step(runtime,t_bound=spec.t_end,console=False))
    before,_ = _verify(spec,runtime,world,1)
    directory = collective_directory(world,tmp_path)
    path = collective_call(world,lambda:runtime.checkpoint(str(directory/"provider")))
    replay = _bind(spec,artifact,world)
    collective_call(world,lambda:replay.restart(path))
    restored = collective_call(world,lambda:_image(replay))
    with collective_check(world):
        assert restored == before
    images=[]
    for candidate in (runtime,replay):
        collective_call(world,lambda candidate=candidate:pops.step(candidate,t_bound=spec.t_end,console=False))
        image,_ = _verify(spec,candidate,world,2)
        images.append(image)
    with collective_check(world):
        assert images[0] == images[1]


def test_uniform_rejected_assessment_restores_complete_provider_image(
        isolated_native_cache,native_cxx,kokkos_root):
    del isolated_native_cache,native_cxx,kokkos_root
    spec,artifact,world = _compile("rejected Uniform Field",builder=build_independent_uniform,reject=True)
    runtime = _bind(spec,artifact,world)
    collective_call(world,lambda:pops.step(runtime,t_bound=spec.t_end,console=False))
    before,_ = _verify(spec,runtime,world,1)
    state = collective_call(world,lambda:runtime.state_global("method"))
    _,errors = collective_attempt(world,lambda:pops.step(runtime,t_bound=spec.t_end,console=False))
    restored = collective_call(world,lambda:_image(runtime))
    after = collective_call(world,lambda:runtime.state_global("method"))
    with collective_check(world):
        assert errors and all(errors)
        assert restored == before
        assert runtime.time() == spec.dt and runtime.macro_step() == 1
        if state is not None:
            np.testing.assert_array_equal(state,after)
