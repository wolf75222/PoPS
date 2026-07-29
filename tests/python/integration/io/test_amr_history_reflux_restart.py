"""Strict AMR restart preserves the load-bearing AB2 reflux continuation.

This is the aggregate ADC-700 acceptance that the separate history-checkpoint and
Program-reflux tests cannot provide on their own: one compiled AB2 Program carries a
non-zero transport flux through a real two-level regrid, checkpoints the accepted
history-flux ledger, restores a fresh runtime under the strict policy, and continues
bit-identically through the same reflux/average-down route.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pops
import pops.lib.time as libtime
import pytest
from pops.lib.initial import BindArray
from pops.time import FailRun, FixedDt
from tests.python.integration._final_field_program import (
    resolve_periodic_field_program,
    scalar_advection_model,
)
from tests.python.support.native_execution_context import artifact_execution_context


ROOT = Path(__file__).resolve().parents[4]
N = 16
DT = 1.0 / 128.0
STEPS_BEFORE_RESTART = 4
STEPS_AFTER_RESTART = 4

pytestmark = [
    pytest.mark.compiler,
    pytest.mark.kokkos,
    pytest.mark.native_loader,
    pytest.mark.regression,
]


def _blob() -> np.ndarray:
    x = (np.arange(N) + 0.5) / N
    xx, yy = np.meshgrid(x, x, indexing="ij")
    return 1.0 + 0.5 * np.exp(-((xx - 0.5) ** 2 + (yy - 0.5) ** 2) / (0.12**2))


def _compile(native_cxx: str) -> tuple[Any, Any, Any]:
    model = scalar_advection_model("adc700_strict_restart_reflux")

    def program(state: Any, rate: Any, fields: Any) -> Any:
        result = libtime.AdamsBashforth(
            state,
            rate=rate,
            fields=fields,
            order=2,
            solve_action=FailRun(),
        )
        result.step_strategy(FixedDt(DT))
        return result

    plan_name = "adc700_strict_restart_reflux_ab2"
    plan = resolve_periodic_field_program(
        model,
        program,
        name=plan_name,
        block_name="blk",
        target="amr_system",
        n=N,
        regrid_every=2,
        initial_profile=BindArray(),
        cxx=native_cxx,
        include=str(ROOT / "include"),
        strict_restart=True,
    )
    bindings = tuple(plan.initial_condition_plan.bindings)
    thresholds = tuple(
        slot.handle
        for slot in plan.bind_schema.runtime_slots
        if slot.handle.local_id == "%s_refine_threshold" % plan_name
    )
    assert len(bindings) == len(thresholds) == 1
    artifact = pops.compile(plan)
    assert artifact.plan.restart_authority.operation.bit_identical is True
    return artifact, bindings[0].subject, thresholds[0]


def _bind(compiled: tuple[Any, Any, Any], initial: np.ndarray) -> Any:
    artifact, initial_subject, threshold = compiled
    return pops.bind(
        artifact,
        params={threshold: 1.2},
        initial_values={
            initial_subject: np.ascontiguousarray(initial[None, ...], dtype=np.float64),
        },
        resources={"execution_context": artifact_execution_context(artifact)},
    )


def _advance(runtime: Any, steps: int, output_dir: Path) -> None:
    step_before = int(runtime.macro_step())
    result = pops.run(
        runtime,
        t_end=float(runtime.time()) + steps * DT,
        max_steps=steps,
        output_dir=output_dir,
    )
    assert result.accepted_steps > 0
    assert result.rejected_steps == 0
    assert int(runtime.macro_step()) == step_before + steps


def _capture(runtime: Any) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    report = runtime.program_report()
    regrid = runtime.amr.explain_regrid()
    metadata = {
        "time_bits": np.float64(runtime.time()).tobytes(),
        "macro_step": int(runtime.macro_step()),
        "patch_boxes": tuple(tuple(int(value) for value in row) for row in runtime.patch_boxes()),
        "regrid_count": int(regrid.regrid_count),
        "topology_epoch": int(regrid.topology_epoch),
        "consumer_cursors": runtime.consumer_cursors.to_data(),
        "program_report": report.to_dict(),
    }
    arrays = {
        "state:%d" % level: np.ascontiguousarray(
            runtime.block_level_state_global("blk", level), dtype=np.float64
        )
        for level in range(int(runtime.n_levels()))
    }
    for name in runtime.history_names():
        for lag in range(int(runtime.history_depth(name))):
            arrays["history:%s:%d" % (name, lag)] = np.ascontiguousarray(
                runtime.history_global(name, lag), dtype=np.float64
            )
    assert any(name.startswith("history:") for name in arrays)
    return metadata, arrays


def _assert_bit_identical(
    expected: tuple[dict[str, Any], dict[str, np.ndarray]],
    actual: tuple[dict[str, Any], dict[str, np.ndarray]],
) -> None:
    expected_metadata, expected_arrays = expected
    actual_metadata, actual_arrays = actual
    assert actual_metadata == expected_metadata
    assert tuple(actual_arrays) == tuple(expected_arrays)
    for name, expected_array in expected_arrays.items():
        actual_array = actual_arrays[name]
        assert actual_array.dtype == expected_array.dtype
        assert actual_array.shape == expected_array.shape
        assert actual_array.tobytes(order="C") == expected_array.tobytes(order="C")


def _program_accepted_state(path: str | Path) -> tuple[tuple[str, bytes], ...]:
    with np.load(path, allow_pickle=False) as checkpoint:
        keys = tuple(
            sorted(
                name for name in checkpoint.files if name.startswith("program_accepted_state_rank_")
            )
        )
        assert keys
        return tuple(
            (name, np.asarray(checkpoint[name], dtype=np.uint8).tobytes()) for name in keys
        )


def _require_reflux_report(runtime: Any) -> None:
    report = runtime.program_report()
    level_clocks = [row for row in report.clocks if row["kind"] == "level"]
    assert {int(row["level"]) for row in level_clocks} == {0, 1}
    assert all(row["phase"] == {"numerator": 0, "denominator": 1} for row in level_clocks)
    assert report.histories
    assert all(row["initialized"] is True for row in report.histories)
    assert {int(row["level"]) for row in report.flux_ledger} == {0, 1}
    assert all(float(row["substep_duration"]) > 0.0 for row in report.flux_ledger)
    groups: dict[tuple[int, ...], list[str]] = {}
    for row in report.synchronization:
        phase = row["clock_phase"]
        key = (
            int(row["parent_level"]),
            int(row["child_level"]),
            int(row["block"]),
            int(row["macro_step"]),
            int(phase["numerator"]),
            int(phase["denominator"]),
        )
        groups.setdefault(key, []).append(str(row["phase"]))
    assert groups
    assert all(tuple(phases) == ("reflux", "average_down") for phases in groups.values())


def test_strict_restart_preserves_ab2_history_flux_and_reflux_continuation(
    tmp_path: Path,
    native_cxx: str,
    isolated_native_cache: Any,
    kokkos_root: Any,
) -> None:
    del isolated_native_cache, kokkos_root
    initial = _blob()
    compiled = _compile(native_cxx)
    uninterrupted = _bind(compiled, initial)
    initial_regrids = int(uninterrupted.amr.explain_regrid().regrid_count)
    initial_mass = float(uninterrupted.integral("blk", levels=(0,)))

    _advance(
        uninterrupted,
        STEPS_BEFORE_RESTART,
        tmp_path / "uninterrupted",
    )
    accepted = _capture(uninterrupted)
    assert accepted[0]["regrid_count"] > initial_regrids
    assert any(int(row["committed_samples"]) > 0 for row in accepted[0]["consumer_cursors"]["rows"])
    _require_reflux_report(uninterrupted)

    checkpoint = uninterrupted.checkpoint(tmp_path / "accepted")
    accepted_program_state = _program_accepted_state(checkpoint)
    assert any(payload for _name, payload in accepted_program_state)

    restarted = _bind(compiled, initial)
    restarted.restart(checkpoint)
    restored = _capture(restarted)
    _assert_bit_identical(accepted, restored)
    _require_reflux_report(restarted)

    roundtrip = restarted.checkpoint(tmp_path / "restored")
    assert _program_accepted_state(roundtrip) == accepted_program_state

    _advance(
        uninterrupted,
        STEPS_AFTER_RESTART,
        # A checkpoint publication is immutable.  Give the second run interval its
        # own root so the on-start consumer proves its restored cursor without
        # colliding with the first interval's accepted sample.
        tmp_path / "uninterrupted_continuation",
    )
    _advance(
        restarted,
        STEPS_AFTER_RESTART,
        tmp_path / "restarted",
    )
    _assert_bit_identical(_capture(uninterrupted), _capture(restarted))
    _require_reflux_report(uninterrupted)
    _require_reflux_report(restarted)

    uninterrupted_mass = float(uninterrupted.integral("blk", levels=(0,)))
    restarted_mass = float(restarted.integral("blk", levels=(0,)))
    assert np.float64(uninterrupted_mass).tobytes() == np.float64(restarted_mass).tobytes()
    assert abs(uninterrupted_mass - initial_mass) < 1.0e-8
