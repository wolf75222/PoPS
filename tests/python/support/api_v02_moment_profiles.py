"""Shared native moment-profile reception; equations and checks stay in the public builder."""
import pops

from examples.api_v02.models_M15_M28 import build_model, build_m19
from examples.api_v02.models_M20_M21 import build_m20, build_m21
from tests.python.support.public_native_profile import native_profile, compile_profile_plan_once
from tests.python.support.collective_checks import collective_call, collective_check
from tests.python.support.native_execution_context import artifact_execution_context


def _bind(world, spec):
    plan = collective_call(world, lambda: pops.resolve(pops.validate(spec.case), layout=spec.layout))
    artifact = compile_profile_plan_once(world, plan, route=spec.profile, compile_artifact=pops.compile)
    context = collective_call(world, lambda: artifact_execution_context(artifact))
    return collective_call(world, lambda: pops.bind(artifact, initial_values=spec.initial_values,
        resources={"execution_context": context}))


def run_native_moment_profile(model, record_property):
    native, world = native_profile(scope="API v0.2 moment and finite phase-space PDE")
    # Preserve the original reduced-dimensional regressions and their oracles.
    # The selected 3x3v and 3x2v profiles have dedicated full-state receivers;
    # their public builders do not accept this legacy one-dimensional mesh.
    legacy = {"M19": build_m19, "M20": build_m20, "M21": build_m21}
    spec = collective_call(world, lambda: legacy[model](cells=16, steps=4)
                           if model in legacy else build_model(model, cells=16, steps=4))
    runtime = _bind(world, spec)
    collective_call(world, lambda: pops.run(runtime, t_end=spec.t_end, max_steps=4, console=False))
    # This callback performs collective native gathers on all ranks.
    metrics = collective_call(world, lambda: spec.invariants(runtime))
    with collective_check(world):
        assert metrics["passed"], metrics
    record_property("profile", spec.profile)
    record_property("native_path", str(native.__file__))
    record_property("metrics", str(metrics))
