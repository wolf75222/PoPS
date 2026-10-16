"""Real cut-cell measures enter authored relaxation and survive its transaction."""
import numpy as np
import pops
import pytest
from examples.api_v02.method_relaxed_cutcell import build_t10_cutcell
from tests.python.support.api_v02_native import _compile, _bind
from tests.python.support.collective_checks import collective_call, collective_check, collective_attempt
from tests.python.support.integral_state_receipts import collective_directory
from tests.python.support.public_native_profile import profile_gather_value
from tests.python.support import cutcell_composite_oracle as oracle

pytestmark = [pytest.mark.compiler, pytest.mark.kokkos, pytest.mark.native_loader]
POPS_PROCESS_TIMEOUT = 900


@pytest.mark.parametrize("ratio", (1, 2))
def test_native_cutcell_volume_controls_rrk_energy_and_checkpoint(
        ratio, tmp_path, isolated_native_cache, native_cxx, kokkos_root, record_property):
    del isolated_native_cache, native_cxx, kokkos_root
    baseline_spec, baseline_artifact, world = _compile("T10_cutcell_candidate", builder=build_t10_cutcell,
                                                       temporal_ratio=ratio, relaxed=False)
    spec, artifact, world = _compile("T10_cutcell", builder=build_t10_cutcell, temporal_ratio=ratio)
    baseline, runtime = _bind(baseline_spec, baseline_artifact, world), _bind(spec, artifact, world)
    before = oracle.snapshot(runtime, world)
    for instance in (baseline, runtime):
        report = collective_call(world, lambda: pops.step(instance, t_bound=1., console=False))
        with collective_check(world):
            assert report.accepted_steps == 1
    provisional, accepted = oracle.snapshot(baseline, world), oracle.snapshot(runtime, world)
    def inspect():
        if before["states"][0] is None:
            return None
        gamma = oracle.factor(before["states"], provisional["states"], before["weights"])
        wrong = oracle.factor(before["states"], provisional["states"], before["unweighted"])
        error = max(float(np.max(abs(a-(o+gamma*(p-o))))) for o, p, a in
                    zip(before["states"], provisional["states"], accepted["states"], strict=True))
        return {"gamma": gamma, "unweighted_gamma": wrong, "state_error": error,
                "energy_before": oracle.pair(before["states"], before["states"], before["weights"]),
                "energy_after": oracle.pair(accepted["states"], accepted["states"], before["weights"])}
    metrics = profile_gather_value(world, collective_call(world, inspect))
    diagnostics = collective_call(world, runtime.program_report).diagnostics
    with collective_check(world):
        assert len(before["weights"]) == 2
        assert any(np.any((a > 0) & (a < 1)) for a in before["fractions"])
        volume = sum(float(np.sum(w)) for w in before["weights"])
        assert .4 < volume < .8
        assert abs(diagnostics["metric.volume"]-volume) < 2.e-13
        assert abs(diagnostics["metric.energy"]-diagnostics["metric.pair"]) < 2.e-13
        for row in metrics:
            if row is None:
                continue
            assert abs(row["gamma"]-row["unweighted_gamma"]) > 1.e-7
            assert abs(row["gamma"]-runtime.time()) < 2.e-13
            assert row["state_error"] < 2.e-12
            assert abs(row["energy_before"]-row["energy_after"]) < 2.e-12
        for key in ("weights", "fractions"):
            for a, b in zip(before[key], accepted[key], strict=True):
                np.testing.assert_array_equal(a, b)
    directory = collective_directory(world, tmp_path)
    checkpoint = collective_call(world, lambda: runtime.checkpoint(str(directory/"cutcell_relaxation")))
    resumed = _bind(spec, artifact, world)
    collective_call(world, lambda: resumed.restart(checkpoint))
    images = []
    for active in (runtime, resumed):
        collective_call(world, lambda active=active: pops.step(active, t_bound=active.time()+1., console=False))
        images.append(oracle.snapshot(active, world))
    with collective_check(world):
        assert images[0]["time"] == images[1]["time"] and images[0]["macro"] == images[1]["macro"] == 2
        for a, b in zip(images[0]["states"], images[1]["states"], strict=True):
            if a is not None:
                np.testing.assert_array_equal(a, b)
    record_property("artifact_identity", artifact.artifact_identity.hexdigest)
    record_property("physical_measure_metrics", str(metrics))


def test_cutcell_late_rejection_restores_state_clock_and_measure(
        isolated_native_cache, native_cxx, kokkos_root):
    del isolated_native_cache, native_cxx, kokkos_root
    spec, artifact, world = _compile("T10_cutcell_rollback", builder=build_t10_cutcell, reject_parent=True)
    runtime = _bind(spec, artifact, world)
    before = oracle.snapshot(runtime, world)
    image = collective_call(world, lambda: bytes(runtime._executor.program_accepted_state()))
    _, errors = collective_attempt(world, lambda: pops.step(runtime, t_bound=1., console=False))
    after = oracle.snapshot(runtime, world)
    restored = collective_call(world, lambda: bytes(runtime._executor.program_accepted_state()))
    with collective_check(world):
        assert errors and all(row and "reject_cutcell_acceptance" in row[1] for row in errors), errors
        assert image == restored and before["time"] == after["time"] == 0.
        assert before["macro"] == after["macro"] == 0
        for key in ("states", "weights", "fractions"):
            for a, b in zip(before[key], after[key], strict=True):
                if a is not None:
                    np.testing.assert_array_equal(a, b)
