"""Future installed Native Serial/MPI reception of durable accepted diagnostic maps.

Only the admission tests run source-only. All marked cases compile/bind/run real
Uniform or AMR providers; no monkeypatch substitutes a DSO, rank or collective.
"""
from pathlib import Path
import ast
import inspect
import json
import sys

import numpy as np
import pops
import pytest

from tests.python.support.collective_checks import collective_attempt, collective_call, collective_check
from tests.python.support.program_diagnostic_native_receipts import (
    DT, RAW_BITS, bind, build, decoded, prepare, rank, record_raw, resealed_fault,
    same_images, save, size, snapshot, snapshot_geometry,
)


def world_context():
    from pops._native_selector import select_native_dimension
    from pops.codegen._native_mpi import native_mpi_communicator
    native = select_native_dimension(2)
    return native.mpi_world() if native_mpi_communicator(native) == "MPI_COMM_WORLD" else None


def require_failures(world, operation, message):
    _, failures = collective_attempt(world, operation)
    with collective_check(world):
        assert len(failures) == size(world) and all(failures), failures
        assert all(message in failure[1] for failure in failures), failures
    return failures


def admitted(world, tmp_path, amr):
    artifact, context, directory = prepare(world, tmp_path, amr)
    runtime = bind(world, artifact, context)
    with collective_check(world):
        assert Path(pops.__file__).resolve().is_relative_to(Path(sys.prefix).resolve())
        assert artifact.resolved_dimension == 2
    initial = snapshot(world, runtime, amr)
    with collective_check(world):
        assert initial["lifecycle"][:2] == (0.0.hex(), 0)
        if amr:
            assert len([key for key in initial["arrays"] if key.endswith("_state")]) == 2
            active = initial["arrays"]["level0_active"]
            assert np.any(active) and not np.all(active), "require actual partial AMR coverage"
    initial_capacity = len(initial["diagnostics"])
    collective_call(world, lambda: runtime.configure_checkpoint_diagnostics(capacity_per_rank=initial_capacity))
    initial_checkpoint = collective_call(world, lambda: runtime.checkpoint(directory / "initial"))
    save(world, runtime, artifact, directory, "initial", initial, checkpoint=initial_checkpoint)
    initial_reload = bind(world, artifact, context)
    collective_call(world, lambda: initial_reload.configure_checkpoint_diagnostics(capacity_per_rank=initial_capacity))
    collective_call(world, lambda: initial_reload._executor._s.record_program_diagnostic("stale-initial", 31.))
    collective_call(world, lambda: initial_reload.restart(initial_checkpoint))
    reloaded_initial = snapshot(world, initial_reload, amr)
    same_images(world, initial, reloaded_initial)
    save(world, initial_reload, artifact, directory, "initial-reloaded", reloaded_initial)
    collective_call(world, lambda: pops.run(runtime, t_end=DT, max_steps=1, console=False))
    record_raw(world, runtime)
    accepted = snapshot(world, runtime, amr)
    with collective_check(world):
        assert accepted["lifecycle"][:2] == (DT.hex(), 1)
        for key, before in initial["arrays"].items():
            if not key.endswith("_state"):
                continue
            after = accepted["arrays"][key]
            if before.size:
                shape = before.shape
                expected = before.reshape(2, -1) * np.array((1-DT, 1+.5*DT))[:, None]
                np.testing.assert_allclose(after, expected.reshape(shape), rtol=0, atol=3e-14)
    # Same reduced sink on every real rank, deliberately distinct raw local sink.
    if world is not None:
        from pops._native_collectives import allgather_value
        values = collective_call(world, lambda: allgather_value(world, decoded(accepted["diagnostics"])[2]["global-energy"]))
        with collective_check(world):
            assert len(set(values)) == 1
    capacity = len(accepted["diagnostics"])
    collective_call(world, lambda: runtime.configure_checkpoint_diagnostics(capacity_per_rank=capacity))
    checkpoint = collective_call(world, lambda: runtime.checkpoint(directory / "accepted"))
    save(world, runtime, artifact, directory, "accepted", accepted, checkpoint=checkpoint)
    return runtime, artifact, context, directory, accepted, checkpoint, capacity


@pytest.mark.parametrize("amr", (False, True))
def test_diagnostic_fixture_source_admission(amr):
    from pops.codegen.program_codegen import emit_cpp_program
    from pops.codegen.program_models import ProgramModelGraph
    case, layout = build(amr)
    plan = pops.resolve(pops.validate(case), layout=layout)
    cpp = emit_cpp_program(plan.time, model=ProgramModelGraph.from_resolved_blocks(plan.blocks),
                           target="amr_system" if amr else "system")
    assert 'ctx.record_scalar("global-energy"' in cpp
    assert "dot_all" in cpp and "accepted-fluid" in cpp
    assert "updated" in cpp


def test_diagnostic_geometry_routes_match_real_native_provider_bindings():
    """Source admission of the real APIs, without substituting a runtime object."""
    from pops.runtime._runtime_instance import RuntimeInstance

    function = ast.parse(inspect.getsource(snapshot_geometry)).body[0]
    branch = function.body[0]
    assert isinstance(branch, ast.If) and isinstance(branch.test, ast.Name)
    assert branch.test.id == "amr" and not branch.orelse

    def runtime_methods(nodes):
        return {(node.attr, isinstance(node.ctx, ast.Load))
                for root in nodes for node in ast.walk(root)
                if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
                and node.value.id == "runtime"}

    assert runtime_methods(branch.body) == {("patch_boxes", True)}
    assert runtime_methods(function.body[1:]) == {("spatial_shape", True), ("local_boxes", True)}
    uniform = "\n".join(inspect.getsource(getattr(RuntimeInstance, name))
                        for name in ("spatial_shape", "local_boxes"))
    assert "patch_boxes" not in uniform
    root = Path(__file__).resolve().parents[4]
    native = (root / "python/bindings/core/init/init_system.cpp").read_text()
    assert '.def("spatial_shape"' in native
    assert '"local_boxes"' in native and "system.local_boxes(name)" in native
    assert '.def("patch_boxes"' not in native


@pytest.mark.compiler
@pytest.mark.kokkos
@pytest.mark.native_loader
@pytest.mark.parametrize("amr", (False, True))
def test_native_diagnostic_bits_checkpoint_restart_replay(
        amr, isolated_native_cache, native_cxx, kokkos_root, tmp_path, record_property):
    del isolated_native_cache, native_cxx, kokkos_root
    world = world_context()
    runtime, artifact, context, directory, accepted, checkpoint, capacity = admitted(world, tmp_path, amr)
    collective_call(world, lambda: pops.run(runtime, t_end=2*DT, max_steps=1, console=False))
    continuous = snapshot(world, runtime, amr)
    save(world, runtime, artifact, directory, "continuous", continuous)
    restored = bind(world, artifact, context)
    collective_call(world, lambda: restored.configure_checkpoint_diagnostics(capacity_per_rank=capacity))
    # A stale map must be replaced entirely, not merged on restore.
    collective_call(world, lambda: restored._executor._s.record_program_diagnostic("stale-only", 91.))
    collective_call(world, lambda: restored.restart(checkpoint))
    reloaded = snapshot(world, restored, amr)
    same_images(world, accepted, reloaded)
    save(world, restored, artifact, directory, "reloaded", reloaded)
    collective_call(world, lambda: pops.run(restored, t_end=2*DT, max_steps=1, console=False))
    replay = snapshot(world, restored, amr)
    same_images(world, continuous, replay)
    save(world, restored, artifact, directory, "replay", replay)
    for key, value in (("evidence_path", str(directory)), ("mpi_rank", rank(world)),
                       ("mpi_size", size(world)), ("native_dimension", 2),
                       ("artifact_identity", artifact.artifact_identity.token)):
        record_property(key, value)


@pytest.mark.compiler
@pytest.mark.kokkos
@pytest.mark.native_loader
@pytest.mark.parametrize("amr", (False, True))
def test_native_diagnostic_capacity_faults_and_joined_restore(
        amr, isolated_native_cache, native_cxx, kokkos_root, tmp_path, monkeypatch, record_property):
    del isolated_native_cache, native_cxx, kokkos_root
    world = world_context()
    runtime, artifact, context, directory, accepted, checkpoint, capacity = admitted(world, tmp_path, amr)
    native = runtime._executor._s
    refused = {}
    refused["reserved-name"] = require_failures(world, lambda: native.record_program_diagnostic(
        "pops.balance-term.invalid", 1.), "reserved namespace")
    # Invalid on rank zero, valid elsewhere: actual owner lane must vote first.
    refused["invalid-value-on-rank-zero"] = require_failures(world, lambda: runtime.configure_checkpoint_diagnostics(
        capacity_per_rank=True if rank(world) == 0 else capacity), "exact positive int")
    same_images(world, accepted, snapshot(world, runtime, amr))
    refused["capacity-overflow"] = require_failures(world, lambda: runtime.configure_checkpoint_diagnostics(
        capacity_per_rank=sys.maxsize), "capacity")
    if size(world) > 1:
        refused["proposal-mismatch"] = require_failures(world, lambda: runtime.configure_checkpoint_diagnostics(
            capacity_per_rank=capacity + rank(world)), "proposal differs across ranks")
    same_images(world, accepted, snapshot(world, runtime, amr))
    collective_call(world, native._begin_step_transaction)
    try:
        refused["active-attempt"] = require_failures(world, lambda: runtime.configure_checkpoint_diagnostics(
            capacity_per_rank=capacity + 100), "fully accepted native state")
    finally:
        collective_call(world, native._rollback_step_transaction)
    same_images(world, accepted, snapshot(world, runtime, amr))
    # Exact boundary succeeds above; one byte less must refuse before publishing a CP.
    collective_call(world, lambda: runtime.configure_checkpoint_diagnostics(capacity_per_rank=capacity-1))
    bad_target = directory / "too-small.npz"
    refused["capture-one-byte-over-reserve"] = require_failures(world, lambda: runtime.checkpoint(bad_target), "configure_checkpoint_diagnostics")
    with collective_check(world):
        assert not bad_target.exists()
    same_images(world, accepted, snapshot(world, runtime, amr))
    collective_call(world, lambda: runtime.configure_checkpoint_diagnostics(capacity_per_rank=capacity))
    collective_call(world, lambda: pops.run(runtime, t_end=2*DT, max_steps=1, console=False))
    before = snapshot(world, runtime, amr)
    save(world, runtime, artifact, directory, "before-faults", before)
    attacks = [("offset", "diagnostic checkpoint"),
               ("rank-authority", "another rank authority"),
               ("native-body", "entry count exceeds actual bytes")]
    if size(world) > 1:
        attacks.append(("rank-order", "another rank authority"))
    for attack, message in attacks:
        path = resealed_fault(world, runtime, checkpoint, directory, attack)
        refused[attack] = require_failures(world, lambda path=path: runtime.restart(path), message)
        same_images(world, before, snapshot(world, runtime, amr))
    # Inject after actual state/history/map application, before the existing apply vote.
    original_apply = runtime._executor._apply_checkpoint_restart

    def applied_then_refused(prepared):
        result = original_apply(prepared)
        if rank(world) == 0:
            raise RuntimeError("diagnostic fixture failure after real native restore")
        return result

    with monkeypatch.context() as patch:
        patch.setattr(runtime._executor, "_apply_checkpoint_restart", applied_then_refused)
        refused["post-apply"] = require_failures(world, lambda: runtime.restart(checkpoint),
            "diagnostic fixture failure after real native restore")
    same_images(world, before, snapshot(world, runtime, amr))
    legacy = resealed_fault(world, runtime, checkpoint, directory, "legacy-absence")
    restored = bind(world, artifact, context)
    collective_call(world, lambda: restored.configure_checkpoint_diagnostics(capacity_per_rank=capacity))
    record_raw(world, restored)
    collective_call(world, lambda: restored.restart(legacy))
    legacy_image = snapshot(world, restored, amr)
    same_images(world, accepted, legacy_image, diagnostics=False)
    with collective_check(world):
        assert decoded(legacy_image["diagnostics"])[2] == {}
    save(world, restored, artifact, directory, "legacy-cleared", legacy_image, checkpoint=str(legacy))
    collective_call(world, lambda: pops.run(restored, t_end=2*DT, max_steps=1, console=False))
    legacy_replay = snapshot(world, restored, amr)
    same_images(world, before, legacy_replay, diagnostics=False)
    with collective_check(world):
        names = decoded(legacy_replay["diagnostics"])[2]
        assert not (set(RAW_BITS) | {"rank-local"}) & set(names)
        assert names["global-energy"] == decoded(before["diagnostics"])[2]["global-energy"]
        (directory / ("rank%d" % rank(world)) / "refusals.json").write_text(
            json.dumps(refused, indent=2) + "\n")
    save(world, restored, artifact, directory, "legacy-replay", legacy_replay)
    for key, value in (("evidence_path", str(directory)), ("mpi_rank", rank(world)), ("mpi_size", size(world))):
        record_property(key, value)
