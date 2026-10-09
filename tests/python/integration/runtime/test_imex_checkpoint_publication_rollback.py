"""Real final IMEX physics, failed final-checkpoint publication and compensation.

Prospective Native test: Root must first authorize the corrected Native/Python pair.
Only the publication seam is patched; every solver, capture and writer remains real.
"""
from dataclasses import replace
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pops
import pytest
from tests.python.support.collective_checks import collective_attempt, collective_call, collective_check
from tests.python.support.evolved_stage_v_capture import pin, retain_v_provenance, save_json
from tests.python.support.integral_state_receipts import collective_directory
from tests.python.integration.runtime.test_final_imex_original_reception import _save_snapshot

pytestmark = [pytest.mark.compiler, pytest.mark.native_loader]
EXAMPLE = Path(__file__).resolve().parents[4] / "examples/final/EXEMPLE_SPEC_FINALE_ADVECTION_IMEX_AMR.py"
FAULT = "IMEX deterministic publication failure after committed final checkpoint"
# Fixed before execution. Checkpoint reseals may differ. Failed-attempt statistics
# alone may survive rollback by RuntimeInstance._restore_unsuccessful_attempt_stats.
RESEAL_KEYS = frozenset(("pops_checkpoint_manifest", "pops_restart_identity"))


def _files(root):
    return {str(p.relative_to(root)): dict(bytes=p.stat().st_size,
        sha256=hashlib.sha256(p.read_bytes()).hexdigest())
        for p in sorted(root.rglob("*")) if p.is_file()}


def _metadata_rollback(before, after, path=()):
    """Allow only Source-declared failed counters, never executable state/cursors."""
    if isinstance(before, dict):
        assert isinstance(after, dict) and set(before) == set(after), path
        if path and path[-1] == "transaction_stats":
            assert set(before) == {"accepted", "failed", "rejected"}
            assert after["accepted"] == before["accepted"]
            assert after["rejected"] == before["rejected"]
            assert type(after["failed"]) is int and before["failed"] <= after["failed"] <= before["failed"] + 1
            return
        for key in before:
            _metadata_rollback(before[key], after[key], (*path, key))
    elif isinstance(before, list):
        assert isinstance(after, list) and len(before) == len(after), path
        for i, (a, b) in enumerate(zip(before, after, strict=True)):
            _metadata_rollback(a, b, (*path, str(i)))
    else:
        assert before == after, path


def _checkpoint_arrays(path):
    with np.load(path, allow_pickle=False) as data:
        return {name: data[name].copy() for name in data.files}


def _save_science(directory, phase, snapshot):
    _save_snapshot(directory, phase, snapshot)  # Original snapshot serialization is unchanged.
    for family in ("states", "fields"):
        for route, levels in getattr(snapshot, family).items():
            token = hashlib.sha256(route.encode()).hexdigest()[:16]
            for level, array in enumerate(levels):
                np.save(directory / ("%s-%s-%s-level%d.npy" % (phase, family, token, level)),
                        array, allow_pickle=False)


def _same_checkpoints(before, after):
    assert set(before) == set(after)
    for name in before:
        a, b = before[name], after[name]
        assert a.dtype == b.dtype and a.shape == b.shape, name
        if name in RESEAL_KEYS:
            continue
        if name == "temporal_restart_state":
            _metadata_rollback(json.loads(str(a)), json.loads(str(b)))
        else:
            assert a.tobytes(order="C") == b.tobytes(order="C"), name


def test_original_imex_final_checkpoint_failure_restores_science_and_outputs(
        tmp_path, monkeypatch, record_property, isolated_native_cache, native_cxx, kokkos_root):
    del isolated_native_cache, native_cxx, kokkos_root
    from pops._native_selector import select_native_dimension
    from pops.output._restart_provider import _RestartSnapshot
    native = select_native_dimension(2)
    world = native.mpi_world()
    rank = int(world.rank)
    directory = collective_directory(world, tmp_path / "imex-checkpoint-publication")
    spec = importlib.util.spec_from_file_location("real_final_imex_publication_case", EXAMPLE)
    example = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = example
    spec.loader.exec_module(example)
    # Use the original physical authoring, callbacks, numerics, parameters and consumers.
    target, resolved, artifact = collective_call(world, lambda: example.compile_final_case(use_preset=False))
    collective_call(world, lambda: retain_v_provenance(artifact, native, directory, rank))
    simulation = collective_call(world, lambda: example._bind_artifact(
        artifact, params=example.build_bind_params(target.authoring)))
    output_root = directory / "original-output"
    baseline_files = collective_call(world, lambda: _files(output_root) if rank == 0 else {})
    events = []
    depth_zero = []
    original_publish = _RestartSnapshot.publish

    def fault_after_real_checkpoint(snapshot, destination):
        nonlocal baseline_files
        capture = snapshot._prepared_capture
        if snapshot._runtime is not simulation or capture is None:
            produced = original_publish(snapshot, destination)
            depth_zero.append(str(produced))
            # Initial consumers are legitimate baseline artifacts, not part of the failed step.
            if not events and Path(destination).is_relative_to(output_root):
                baseline_files = collective_call(world, lambda: _files(output_root) if rank == 0 else {})
            return produced
        with collective_check(world):
            assert simulation._executor._s._step_transaction_depth() == 1
            assert capture.contract == "pops.amr.prepared-checkpoint-capture@1"
        collective_call(world, lambda: simulation._validate_committed_checkpoint_candidate(capture))
        # The real final checkpoint is published after the original HDF5/Integral/NPZ/ParaView.
        produced = original_publish(snapshot, destination)
        committed_diagnostics = collective_call(world, lambda: simulation.program_report().to_dict()["diagnostics"])
        def retain_fault_boundary():
            if rank != 0:
                return
            published = _files(output_root)
            changed = {name: row for name, row in published.items() if baseline_files.get(name) != row}
            assert any(name.startswith("hdf5/") and name.endswith(".h5") for name in changed)
            assert any(name.startswith("npz/") and name.endswith(".npz") for name in changed)
            assert any(name.startswith("paraview/") for name in changed)
            assert committed_diagnostics, "the original Integral diagnostic must exist before checkpoint fault"
            retained = directory / "real-committed-checkpoint-before-fault.npz"
            retained.write_bytes(Path(produced).read_bytes())
            save_json(directory / "fault-boundary.json", dict(
                actual_public_checkpoint=str(produced), retained_checkpoint=pin(retained),
                prepared_capture_committed_validated=True, transaction_depth=1,
                earlier_original_outputs=published, newly_published_outputs=changed,
                diagnostics=committed_diagnostics,
                numerical_acceptance=False, root_received=False))
        collective_call(world, retain_fault_boundary)
        events.append("committed-capture-real-publication-then-fault")
        raise RuntimeError(FAULT)  # Identical on every MPI participant; no physics callback changed.

    with monkeypatch.context() as patch:
        patch.setattr(_RestartSnapshot, "publish", fault_after_real_checkpoint)
        # Public zero-step initialization establishes legitimate initial outputs/cursors.
        # Source explicitly forbids a fabricated AtEnd occurrence for this zero-step route.
        initial = collective_call(world, lambda: pops.run(simulation,
            t_end=simulation.time(), max_steps=0, console=False, output_dir=output_root))
        with collective_check(world):
            assert initial.accepted_steps == 0 and not events
        baseline_files = collective_call(world, lambda: _files(output_root) if rank == 0 else {})
        before = collective_call(world, lambda: example._snapshot(simulation))
        before_cp = collective_call(world, lambda: simulation.checkpoint(directory / "before-public"))
        before_arrays = collective_call(world, lambda: _checkpoint_arrays(before_cp) if rank == 0 else {})
        before_fence = simulation._failed_run_effect_fence()
        collective_call(world, lambda: _save_science(directory, "before-rank%d" % rank, before))
        controls = dict(target.authoring.run_controls)
        controls["output_dir"] = output_root
        _, failures = collective_attempt(world, lambda: pops.run(simulation, **controls))
        after = collective_call(world, lambda: example._snapshot(simulation))
        after_cp = collective_call(world, lambda: simulation.checkpoint(directory / "after-public"))
        after_arrays = collective_call(world, lambda: _checkpoint_arrays(after_cp) if rank == 0 else {})
        collective_call(world, lambda: _save_science(directory, "after-rank%d" % rank, after))
        collective_call(world, lambda: save_json(directory / ("failure-rank%d.json" % rank), dict(
            failures=failures, events=events, depth_zero_passed=depth_zero)))
        with collective_check(world):
            assert all(row is not None and FAULT in row[1] for row in failures), failures
            assert events == ["committed-capture-real-publication-then-fault"]
            assert len(depth_zero) >= 2, "public depth-zero checkpoints must pass through without fault"
            _metadata_rollback(json.loads(before.program_transaction_state), json.loads(after.program_transaction_state))
            # Original strict physical oracle unchanged; only the declared failed stats projection differs.
            example._require_same_snapshot(before, replace(after,
                program_transaction_state=before.program_transaction_state), where="final checkpoint publication rollback")
            assert simulation._failed_run_effect_fence() == before_fence
            assert simulation.consumer_recoveries == ()
            if rank == 0:
                _same_checkpoints(before_arrays, after_arrays)
                assert _files(output_root) == baseline_files
                assert not list(output_root.rglob("*.pops-*")), "no checkpoint staging artifacts may survive"
            report = simulation._executor._last_step_transaction_report
            assert report.status == "failed" and report.phase == "effect" and report.action == "fail_run"
            assert report.attempts == 1 and report.staged_effects == report.rolled_back_effects
            assert any(FAULT in message for message in report.diagnostics)
    record_property("imex_publication_rollback_receipt", str(directory / ("failure-rank%d.json" % rank)))
