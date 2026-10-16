"""Real generated PoPS kernels consume >32 bind values and restart unchanged."""
import numpy as np
import pops
import pytest

from examples.api_v02.runtime_parameter_family import build_runtime_parameter_family
from tests.python.support.api_v02_native import _compile
from tests.python.support.collective_checks import collective_call, collective_attempt, collective_check
from tests.python.support.integral_state_receipts import collective_directory
from tests.python.support.native_execution_context import artifact_execution_context

pytestmark = [pytest.mark.compiler, pytest.mark.kokkos, pytest.mark.native_loader]


@pytest.mark.parametrize("count", [33, 65, 129])
def test_runtime_parameter_family_executes_bind_overrides_and_restart(count, tmp_path,
        isolated_native_cache, native_cxx, kokkos_root):
    del isolated_native_cache, native_cxx, kokkos_root
    spec, artifact, world = _compile("runtime_parameter_family", builder=build_runtime_parameter_family,
                                     count=count)
    from pops import _pops
    capacity = _pops.runtime_parameter_storage_bytes(count)
    slots = artifact.bind_schema.runtime_slots
    assert len(slots) == count
    resources = {"execution_context": artifact_execution_context(artifact),
                 "runtime_parameter_byte_capacity": capacity}
    initial = np.asarray(next(iter(spec.initial_values.values())))
    def bind(scale):
        params = {slot.handle: scale*(int(slot.handle.local_id.rsplit("_", 1)[1])+1)
                  for slot in slots}
        return collective_call(world, lambda: pops.bind(artifact, params=params,
            initial_values={key: value.copy() for key, value in spec.initial_values.items()},
            resources=resources))
    _, errors = collective_attempt(world, lambda: pops.bind(artifact,
        initial_values=dict(spec.initial_values), resources={**resources,
            "runtime_parameter_byte_capacity": capacity-1}))
    with collective_check(world):
        assert all(errors), errors
        assert all("bound capacity" in str(error) for error in errors), errors
    slow, fast = bind(1.), bind(2.)
    with collective_check(world):
        report = slow.inspect().to_dict()["instance"]["runtime_parameter_storage"]
        assert report["byte_capacity"] == capacity
        assert report["initial_storage_bytes"] == sum(report["carrier_bytes"].values())
        assert report["scope"] == "immutable_incarnation"
    for runtime, scale in ((slow, 1.), (fast, 2.)):
        collective_call(world, lambda runtime=runtime: pops.step(runtime, t_bound=spec.dt, console=False))
        actual = collective_call(world, lambda runtime=runtime: runtime.state_global("method"))
        with collective_check(world):
            if actual is not None:
                np.testing.assert_allclose(np.asarray(actual).reshape(initial.shape),
                    initial*(1+scale*spec.dt), rtol=0, atol=3.e-13)
    directory = collective_directory(world, tmp_path)
    checkpoint = collective_call(world, lambda: fast.checkpoint(str(directory/"many_params")))
    restored = bind(2.)
    collective_call(world, lambda: restored.restart(checkpoint))
    for runtime in (fast, restored):
        collective_call(world, lambda runtime=runtime: pops.step(runtime, t_bound=2*spec.dt, console=False))
    a = collective_call(world, lambda: fast.state_global("method"))
    b = collective_call(world, lambda: restored.state_global("method"))
    with collective_check(world):
        if a is not None:
            np.testing.assert_array_equal(a, b)
            np.testing.assert_allclose(np.asarray(a).reshape(initial.shape),
                initial*(1+2*spec.dt)**2, rtol=0, atol=3.e-13)
        assert fast.time() == restored.time() == 2*spec.dt
