"""Compile and execute the declared PDE profiles; no equation-only passes."""
from __future__ import annotations

import pops
import pytest
import numpy as np

from examples.api_v02.common import MissingCapabilityError
from examples.api_v02.models_M01_M14 import READY, build_model
from tests.python.support.collective_checks import collective_call, collective_check
from tests.python.support.native_execution_context import artifact_execution_context
from tests.python.support.public_native_profile import (
    compile_profile_plan_once, native_profile, profile_gather_value,
)


@pytest.mark.compiler
@pytest.mark.kokkos
@pytest.mark.native_loader
@pytest.mark.parametrize("model", READY)
def test_api_v02_declared_pde_profile_native(model, isolated_native_cache,
                                            native_cxx, kokkos_root, record_property):
    del isolated_native_cache, native_cxx, kokkos_root
    _native, world = native_profile(dimension=2, scope=f"API v0.2 {model} PDE")
    spec = collective_call(world, lambda: build_model(model))
    resolved = collective_call(world, lambda: pops.resolve(pops.validate(spec.case), layout=spec.layout))
    artifact = compile_profile_plan_once(world, resolved, route=f"API v0.2 {model}",
                                         compile_artifact=pops.compile)
    runtime = collective_call(world, lambda: pops.bind(artifact,
        initial_values=dict(spec.initial_values),
        resources={"execution_context": artifact_execution_context(artifact)}))
    collective_call(world, lambda: pops.run(runtime, t_end=spec.t_end,
                                           max_steps=32, console=False))
    # All ranks enter the gathers inside the invariant callable.
    metrics = collective_call(world, lambda: spec.invariants(runtime))
    gathered = profile_gather_value(world, metrics)
    with collective_check(world):
        owners = [row for row in gathered if row.get("metrics_owner", True)]
        assert owners, "a gathered native state must actually be assessed"
        assert all(row["passed"] for row in owners), owners
        assert abs(runtime.time()-spec.t_end) <= 2.e-13
        assert runtime.macro_step() >= 2
        assert artifact.resolved_dimension == 2
    record_property("model_profile", spec.profile)
    record_property("native_metrics", str(gathered))


def test_api_v02_unknown_family_does_not_fall_back_to_a_witness():
    with pytest.raises(ValueError, match="unknown model"):
        build_model("M99")


@pytest.mark.compiler
@pytest.mark.kokkos
@pytest.mark.native_loader
def test_api_v02_ns_rotates_and_viscosity_changes_native_solution(
        isolated_native_cache, native_cxx, kokkos_root):
    """Full multi-cell two-dimensional momentum/energy transport, not scalar shear."""
    from examples.api_v02.models_M01_M14 import build_m05
    del isolated_native_cache, native_cxx, kokkos_root
    _native, world = native_profile(dimension=2, scope="API v0.2 NS rotation")
    def rotate(value):
        result = np.rot90(value, k=-1, axes=(1, 2)).copy()
        result[1], result[2] = -result[2].copy(), result[1].copy()
        return result
    def execute(spec, artifact, initial):
        runtime = collective_call(world, lambda: pops.bind(artifact,
            initial_values={spec.subjects["fluid"]: np.ascontiguousarray(initial)},
            resources={"execution_context": artifact_execution_context(artifact)}))
        collective_call(world, lambda: pops.run(runtime, t_end=spec.t_end, max_steps=4, console=False))
        result = collective_call(world, lambda: runtime.state_global("fluid"))
        return None if result is None else np.asarray(result).reshape(initial.shape)
    spec = collective_call(world, lambda: build_m05(cells=(12,12),two_dimensional=True))
    plan = collective_call(world, lambda: pops.resolve(pops.validate(spec.case),layout=spec.layout))
    artifact = compile_profile_plan_once(world,plan,route="NS 2D rotating",compile_artifact=pops.compile)
    initial = next(iter(spec.initial_values.values()))
    original = execute(spec,artifact,initial)
    rotated = execute(spec,artifact,rotate(initial))
    inviscid = collective_call(world, lambda: build_m05(cells=(12,12),viscosity=0.,two_dimensional=True))
    inviscid_plan = collective_call(world, lambda: pops.resolve(pops.validate(inviscid.case),layout=inviscid.layout))
    inviscid_artifact = compile_profile_plan_once(world,inviscid_plan,route="NS zero viscosity",compile_artifact=pops.compile)
    zero_mu = execute(inviscid,inviscid_artifact,initial)
    with collective_check(world):
        if original is not None:
            np.testing.assert_allclose(rotated,rotate(original),rtol=2.e-12,atol=2.e-12)
            assert np.max(np.abs(original-zero_mu)) > 1.e-8
