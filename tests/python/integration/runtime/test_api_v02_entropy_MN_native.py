"""Real generated closure PDE; independent dense oracle checks every component."""
import pops
import pytest
from examples.api_v02.models_M18_entropy import build_m18_entropy
from tests.python.support.public_native_profile import native_profile, compile_profile_plan_once
from tests.python.support.collective_checks import collective_call, collective_check
from tests.python.support.native_execution_context import artifact_execution_context

pytestmark=[pytest.mark.compiler,pytest.mark.kokkos,pytest.mark.native_loader]


@pytest.mark.parametrize("order",(2,3))
def test_discrete_entropy_MN_native_Newton_spatial_RK4(isolated_native_cache,order,record_property):
    del isolated_native_cache
    native,world=native_profile(scope="API v0.2 entropy MN native Newton spatial PDE")
    spec=collective_call(world,lambda:build_m18_entropy(order=order))
    plan=collective_call(world,lambda:pops.resolve(pops.validate(spec.case),layout=spec.layout))
    artifact=compile_profile_plan_once(world,plan,route=spec.profile,compile_artifact=pops.compile)
    context=collective_call(world,lambda:artifact_execution_context(artifact))
    runtime=collective_call(world,lambda:pops.bind(artifact,initial_values=spec.initial_values,
        resources={"execution_context":context}))
    collective_call(world,lambda:pops.run(runtime,t_end=spec.t_end,max_steps=2,console=False))
    metrics=collective_call(world,lambda:spec.invariants(runtime))
    with collective_check(world):assert metrics["passed"],metrics
    record_property("profile",spec.profile)
    record_property("native_path",str(native.__file__))
    record_property("metrics",str(metrics))
