"""Source-only checks for the separate native resource-observation tool."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


SCRIPT = (Path(__file__).resolve().parents[4] / "docs" / "development" / "api_040"
          / "joint_reconstruction_resource_probe.py")


def _probe():
    spec = importlib.util.spec_from_file_location("joint_resource_probe", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_companion_uses_unchanged_v1_case() -> None:
    probe = _probe()
    v1 = probe._frozen_v1()
    assert (v1.N, v1.WIDTH, v1.STEPS, v1.DT) == (48, 5, 12, 1e-4)
    assert probe.V1_SHA256 == probe._sha(
        SCRIPT.with_name("joint_reconstruction_benchmark.py"))


def test_absent_counter_is_not_a_measured_zero() -> None:
    probe = _probe()
    absent = probe._counter({}, "scratch_allocs")
    present_zero = probe._counter({"scratch_allocs": 0}, "scratch_allocs")
    assert absent == {"available": False, "value": None,
                      "reason": "counter absent from native profile snapshot"}
    assert present_zero == {"available": True, "value": 0, "unit": "allocations"}
    for invalid in (True, 1.0, -1):
        with pytest.raises(ValueError):
            probe._counter({"scratch_allocs": invalid}, "scratch_allocs")


def test_kernel_operations_are_not_substituted_for_kernel_launches() -> None:
    selected = _probe()._selected_counters({"kernels": 17})
    assert selected["kernels"]["value"] == 17
    assert selected["kernel_launches"]["available"] is False
    assert selected["scratch_peak_bytes"]["available"] is False
    assert selected["mpi_messages"]["available"] is False


def test_partial_instrumentation_preserves_raw_values_without_pair_claim() -> None:
    probe = _probe()
    baseline = probe._selected_counters({"kernels": 2, "scratch_allocs": 0})
    candidate = probe._selected_counters({"kernels": 3})
    rows = {
        "baseline": {"samples": [{"selected_counters": baseline}]
                     * probe.PROFILED_RUNS},
        "candidate": {"samples": [{"selected_counters": candidate}]
                      * probe.PROFILED_RUNS},
    }
    summary = probe._aggregate(rows)
    assert summary["kernels"] == {
        "available": True, "unit": "program_kernel_operations_or_batches",
        "values": {"baseline": [2] * probe.PROFILED_RUNS,
                   "candidate": [3] * probe.PROFILED_RUNS},
    }
    assert summary["scratch_allocs"]["available"] is False
    assert summary["scratch_allocs"]["observed_values"] == {
        "baseline": [0] * probe.PROFILED_RUNS}
