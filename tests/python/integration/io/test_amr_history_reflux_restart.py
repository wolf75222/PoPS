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
# Regrid occurs at macro step two; checkpoint after the first accepted post-regrid step rather
# than at the topology-transition boundary.
STEPS_BEFORE_RESTART = 3
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
        "patch_boxes": tuple(
            (int(level), tuple(int(value) for value in lower), tuple(int(value) for value in upper))
            for level, lower, upper in runtime.patch_boxes()
        ),
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
        for level in runtime.history_levels(name):
            for lag in range(int(runtime.history_depth(name))):
                arrays["history:%s:%d:%d" % (name, level, lag)] = np.ascontiguousarray(
                    runtime.history_global(name, level, lag), dtype=np.float64
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
        key = "program_accepted_state"
        assert key in checkpoint.files
        return ((key, np.asarray(checkpoint[key], dtype=np.uint8).tobytes()),)


def _accepted_state_history_flux(payload: bytes) -> tuple[list[dict[str, Any]], bytes]:
    """Decode only the POPSAND5 history-flux envelope, retaining exact byte boundaries.

    This is deliberately a test-side reader: it proves the wire image contains actual retained
    expressions and also catches accidental offset drift before a checkpoint reaches native restore.
    """
    cursor = 0

    def require(count: int) -> None:
        assert 0 <= count <= len(payload) - cursor, "POPSAND5 payload is truncated"

    def read_u64() -> int:
        nonlocal cursor
        require(8)
        value = int.from_bytes(payload[cursor : cursor + 8], "little")
        cursor += 8
        return value

    def skip(count: int) -> None:
        nonlocal cursor
        require(count)
        cursor += count

    def read_string() -> str:
        size = read_u64()
        require(size)
        nonlocal_cursor = cursor
        skip(size)
        return payload[nonlocal_cursor : nonlocal_cursor + size].decode("utf-8")

    assert payload[:8] == b"POPSAND5"
    cursor = 8 + 8
    skip(read_u64())  # spatial contract
    skip(16)  # topology epoch, materialization generation
    for _ in range(read_u64()):
        skip(40)
    for _ in range(read_u64()):
        skip(read_u64() + 8)
    for _ in range(read_u64()):
        read_string()
        skip(8)
        for _identity in range(4):
            read_string()
        skip(16)
    for _ in range(read_u64()):
        read_string()
        skip(40)
    for _ in range(read_u64()):
        read_string()
        skip(12 * 8)
    history_flux_size = read_u64()
    history_flux_offset = cursor
    skip(history_flux_size)
    history_flux = payload[history_flux_offset:cursor]

    flux_cursor = 0

    def flux_u64() -> int:
        nonlocal flux_cursor
        assert flux_cursor + 8 <= len(history_flux), "history-flux size is truncated"
        value = int.from_bytes(history_flux[flux_cursor : flux_cursor + 8], "little")
        flux_cursor += 8
        return value

    def flux_i64() -> int:
        nonlocal flux_cursor
        assert flux_cursor + 8 <= len(history_flux), "history-flux scalar is truncated"
        value = int.from_bytes(history_flux[flux_cursor : flux_cursor + 8], "little", signed=True)
        flux_cursor += 8
        return value

    def flux_skip(count: int) -> None:
        nonlocal flux_cursor
        assert 0 <= count <= len(history_flux) - flux_cursor, "history-flux record is truncated"
        flux_cursor += count

    def flux_string() -> str:
        size = flux_u64()
        start = flux_cursor
        flux_skip(size)
        return history_flux[start : start + size].decode("utf-8")

    rings: list[dict[str, Any]] = []
    if history_flux:
        for _ in range(flux_u64()):
            name = flux_string()
            slots = []
            for _ in range(flux_u64()):
                bases = []
                for _ in range(flux_u64()):
                    identity = flux_u64()
                    coefficients = flux_u64()
                    flux_skip(coefficients * 24)
                    runtime_block = flux_u64()
                    level = flux_i64()
                    flux_i64()  # RHS id
                    flux_skip(8)  # provider
                    # point: clock, tick, level/substep/stage, rational, dt/time, three identities
                    flux_string()
                    flux_skip(8 + 3 * 8 + 2 * 8 + 2 * 8)
                    flux_string()
                    flux_string()
                    flux_string()
                    flux_skip(80)  # two exact clock stamps
                    faces = []
                    for _ in range(flux_u64()):
                        flux_skip(8 + 8 + 4 * 8 + 8)
                        density_count = flux_u64()
                        density_offset = flux_cursor
                        flux_skip(density_count * 8)
                        faces.append((density_count, history_flux[density_offset:flux_cursor]))
                    bases.append({"identity": identity, "runtime_block": runtime_block,
                                  "level": level, "faces": faces})
                slots.append(bases)
            rings.append({"name": name, "slots": slots})
        assert flux_cursor == len(history_flux), "history-flux parser left trailing bytes"
    return rings, history_flux


def _require_reflux_report(runtime: Any, *, require_current_ledger: bool = True) -> None:
    report = runtime.program_report()
    level_clocks = [row for row in report.clocks if row["kind"] == "level"]
    assert {int(row["level"]) for row in level_clocks} == {0, 1}
    assert all(row["phase"] == {"numerator": 0, "denominator": 1} for row in level_clocks)
    assert report.histories
    assert all(
        level["initialized"] is True
        for row in report.histories
        for level in row["levels"]
    )
    if require_current_ledger:
        assert report.flux_ledger, "a live AB2/reflux run must materialize a per-level flux ledger"
    if report.flux_ledger:
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
    flux_rings, flux_payload = _accepted_state_history_flux(accepted_program_state[0][1])
    assert flux_payload and flux_rings
    assert all(ring["name"] and len(ring["slots"]) == 2 for ring in flux_rings)
    assert all(slot for ring in flux_rings for slot in ring["slots"])
    assert any(base["faces"] for ring in flux_rings for slot in ring["slots"] for base in slot)

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
    _require_reflux_report(uninterrupted, require_current_ledger=False)
    _require_reflux_report(restarted, require_current_ledger=False)

    uninterrupted_mass = float(uninterrupted.integral("blk", levels=(0,)))
    restarted_mass = float(restarted.integral("blk", levels=(0,)))
    assert np.float64(uninterrupted_mass).tobytes() == np.float64(restarted_mass).tobytes()
    assert abs(uninterrupted_mass - initial_mass) < 1.0e-8
