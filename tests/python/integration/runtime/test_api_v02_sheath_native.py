"""M14 spatial sheath/circuit receipt and rejection of its provisional wall current."""
from __future__ import annotations

import json
import numpy as np
import pops
import pytest

from examples.api_v02.models_M14 import build_m14, INITIAL_WALL_CHARGE
from tests.python.support.collective_checks import collective_call, collective_check, collective_attempt
from tests.python.support.integral_state_receipts import collective_directory
from tests.python.support.native_execution_context import artifact_execution_context
from tests.python.support.public_native_profile import compile_profile_plan_once, native_profile


pytestmark = [pytest.mark.compiler, pytest.mark.kokkos, pytest.mark.native_loader]


def _assert_same_accepted_image(before, after):
    """Compare every accepted payload; first-run provenance is a separate host effect."""
    assert set(before) == set(after)
    envelopes = {"pops_checkpoint_manifest", "pops_restart_identity"}
    for key in set(before)-envelopes:
        np.testing.assert_array_equal(after[key], before[key], err_msg=key)
    first = json.loads(before["pops_checkpoint_manifest"].item())
    last = json.loads(after["pops_checkpoint_manifest"].item())
    # Calling run establishes a host run identity even when its first scientific attempt
    # is rejected. The restart seal also authenticates that provenance. Neither is a trial
    # scientific resource, and no payload hash may change under this allowance.
    host_provenance = {"schema_version", "origin", "run_identity", "restart_identity"}
    assert {key: value for key, value in first.items() if key not in host_provenance} == {
        key: value for key, value in last.items() if key not in host_provenance}
    assert first["arrays"] == last["arrays"]
    assert first.get("origin", {}).get("kind") == "bound_initial"
    assert first.get("run_identity") is None and last["run_identity"]["domain"] == "run"


def test_sheath_rejected_wall_current_restores_accepted_checkpoint(
        isolated_native_cache, native_cxx, kokkos_root, tmp_path, record_property):
    del isolated_native_cache, native_cxx, kokkos_root
    _native, world = native_profile(dimension=2, scope="M14 sheath rejected wall current")
    spec, wall, reservoir = collective_call(world, lambda: build_m14(reject=True, receipt=True))
    plan = collective_call(world, lambda: pops.resolve(pops.validate(spec.case), layout=spec.layout))
    artifact = compile_profile_plan_once(world, plan, route="M14 rejection", compile_artifact=pops.compile)
    runtime = collective_call(world, lambda: pops.bind(artifact,
        initial_values=dict(spec.initial_values),
        resources={"execution_context": artifact_execution_context(artifact)}))
    directory = collective_directory(world, tmp_path)
    path = collective_call(world, lambda: runtime.checkpoint(directory / "before"))
    before = None
    with collective_check(world):
        if world is None or int(world.rank) == 0:
            with np.load(path, allow_pickle=False) as data:
                before = {key: data[key].copy() for key in data.files}
    _, errors = collective_attempt(world, lambda: pops.run(
        runtime, t_end=spec.dt, max_steps=1, console=False))
    with collective_check(world):
        assert all(error and "reject_after_provisional_wall_current" in error[1]
                   for error in errors), errors
        assert runtime.time() == 0. and runtime.macro_step() == 0
        assert runtime.integral_state(wall) == INITIAL_WALL_CHARGE
        assert runtime.integral_state(reservoir) == 0.
    path = collective_call(world, lambda: runtime.checkpoint(directory / "after"))
    with collective_check(world):
        if world is None or int(world.rank) == 0:
            with np.load(path, allow_pickle=False) as after:
                _assert_same_accepted_image(before, {key: after[key] for key in after.files})
    record_property("artifact_identity", artifact.artifact_identity.hexdigest)
    record_property("rejection_errors", str(errors))
    record_property("rollback_scope", "every public accepted checkpoint scientific payload and manifest content hash, including volume, field, history and integral/exchange state; first-run host provenance separately identified")
