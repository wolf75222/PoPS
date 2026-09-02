#!/usr/bin/env python3
"""Strict-v9 scheduler-cache checkpoint preflight (Spec 3 section 30, ADC-458).

A compiled `Program` with a held schedule (`every(N).hold` / `accumulate_dt`) caches the System aux /
a scratch in a bind-sealed dense slot so the field solve runs only when DUE. The cache lives in the System
(`System::program_cache()`, NOT the `.so` step closure), so checkpointing it makes a
`(run, checkpoint, restart, continue)` run bit-for-bit identical to a continuous run -- without it the
first post-restart step would cold-start the held slot off its cadence.

This test sends a complete strict-v9 NPZ payload through the production
``preflight_uniform_restart`` route, then proves that a listed cache slot whose value array is absent
is rejected by that same route. It does not reimplement the guard in test code.

The full compiled-`.so` held-schedule continuous == restart RUN is Kokkos-only AOT (ROMEO); the
CacheManager serialize/restore round-trip is unit-tested host-side by
`tests/cpp/integration/runtime/test_checkpoint_cache.cpp`.
"""
import os
import tempfile

import pytest

from tests.python.support.requirements import run_process_test_cases


def test_cache_v9_preflight_accepts_complete_payload_and_rejects_truncation():
    try:
        import numpy as np
    except Exception as exc:  # noqa: BLE001 -- numpy unavailable in this interpreter
        raise RuntimeError("test_scheduler_cache_checkpoint requires NumPy") from exc
    from pops._generated_release_contract import UNIFORM_CHECKPOINT_PAYLOAD_VERSION
    from pops.runtime._uniform_restart_preflight import preflight_uniform_restart

    slot = 0
    cold_slot = 1
    name = "fields_from_state"
    cold_name = "lazy_flux"
    ncomp, ny, nx = 1, 4, 4
    value = np.arange(ncomp * ny * nx, dtype=np.float64).reshape(ncomp, ny, nx)
    out = {
        "pops_checkpoint_version": np.array(
            UNIFORM_CHECKPOINT_PAYLOAD_VERSION, dtype=np.int64
        ),
        "t": np.array(0.3, dtype=np.float64),
        "macro_step": np.array(3, dtype=np.int64),
        "pops_spatial_contract": np.array("{}"),
        "pops_embedded_boundary_contract": np.array("{}"),
        "program_hash": np.array("deadbeef" * 8),
        "temporal_restart_state": np.array("{}"),
        "program_cadence_substeps": np.array(1, dtype=np.int64),
        "program_cadence_stride": np.array(1, dtype=np.int64),
        "program_cadence_window_steps": np.array(0, dtype=np.int64),
        "program_cadence_window_dt": np.array(0.0, dtype=np.float64),
        "program_cadence_window_start_time": np.array(0.0, dtype=np.float64),
        "program_last_dt": np.array(0.0, dtype=np.float64),
        "history_names": np.array([], dtype="U1"),
        "cache_slots": np.array([slot, cold_slot], dtype=np.int64),
        "cache_plan_schema": np.array("program-resource-plan:v1"),
        "cache_plan_digest": np.array("a" * 64),
        "cache_names": np.array([name, cold_name]),
        "cache_valid": np.array([True, False], dtype=np.bool_),
        "cache_cold": np.array([False, True], dtype=np.bool_),
        "cache_ncomp_%d" % slot: np.array(ncomp, dtype=np.int64),
        "cache_ngrow_%d" % slot: np.array(0, dtype=np.int64),
        "cache_last_update_%d" % slot: np.array(2, dtype=np.int64),
        "cache_accum_dt_%d" % slot: np.array(0.0035, dtype=np.float64),
        "cache_value_%d" % slot: value,
        "cache_ncomp_%d" % cold_slot: np.array(0, dtype=np.int64),
        "cache_ngrow_%d" % cold_slot: np.array(0, dtype=np.int64),
        "cache_last_update_%d" % cold_slot: np.array(-1, dtype=np.int64),
        "cache_accum_dt_%d" % cold_slot: np.array(0.25, dtype=np.float64),
    }

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "ckpt.npz")
        with open(path, "wb") as f:
            np.savez_compressed(f, **out)
        with np.load(path, allow_pickle=False) as payload:
            preflight_uniform_restart(payload)

    truncated = {k: v for k, v in out.items() if k != "cache_value_%d" % slot}
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "truncated.npz")
        with open(path, "wb") as f:
            np.savez_compressed(f, **truncated)
        with np.load(path, allow_pickle=False) as payload:
            with pytest.raises(
                ValueError,
                match=r"scheduled cache slot 0 has an incomplete strict manifest .*cache_value_0",
            ):
                preflight_uniform_restart(payload)


if __name__ == "__main__":
    run_process_test_cases(
        {
            "test_cache_v9_preflight_accepts_complete_payload_and_rejects_truncation":
                test_cache_v9_preflight_accepts_complete_payload_and_rejects_truncation,
        }
    )
    print("OK test_scheduler_cache_checkpoint")
