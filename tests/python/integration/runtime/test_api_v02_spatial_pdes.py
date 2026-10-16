"""Native reception of nonlocal M26 and nonlinear M27 on actual spatial cells."""
import pytest
import pops

from examples.api_v02.models_M15_M28 import build_m26, build_m27
from tests.python.support.public_native_profile import native_profile, compile_profile_plan_once
from tests.python.support.collective_checks import collective_call, collective_check
from tests.python.support.native_execution_context import artifact_execution_context


@pytest.mark.compiler
@pytest.mark.kokkos
@pytest.mark.native_loader
@pytest.mark.parametrize("builder,kwargs", (
    (build_m26, {}),
    (build_m26, {"perturbation": (8, .1)}),
    (build_m27, {}),
))
def test_native_multicell_nonlinear_pde(isolated_native_cache, builder, kwargs, record_property):
    del isolated_native_cache
    native, world = native_profile(scope="API v0.2 computed spatial field PDE")
    spec = collective_call(world, lambda: builder(cells=16, dimension=2, steps=4, **kwargs))
    plan = collective_call(world, lambda: pops.resolve(pops.validate(spec.case), layout=spec.layout))
    artifact = compile_profile_plan_once(world, plan, route=spec.profile, compile_artifact=pops.compile)
    context = collective_call(world, lambda: artifact_execution_context(artifact))
    runtime = collective_call(world, lambda: pops.bind(artifact, initial_values=spec.initial_values,
        resources={"execution_context": context}))
    collective_call(world, lambda: pops.run(runtime, t_end=spec.t_end, max_steps=4, console=False))
    metrics = collective_call(world, lambda: spec.invariants(runtime))
    with collective_check(world):
        assert metrics["passed"], metrics
    record_property("profile", spec.profile)
    record_property("native_path", str(native.__file__))
    record_property("metrics", str(metrics))
