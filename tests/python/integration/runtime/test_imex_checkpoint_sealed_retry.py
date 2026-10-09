"""Prospective Native original IMEX: one sealed capture, failed publication, byte-exact retry.

No Native execution is implied by this source file. Root must commit and rebuild first.
"""
from dataclasses import replace
import hashlib
import importlib.util
from pathlib import Path
import sys

import pops
import pytest

from pops.output._consumer_contracts import ConsumerGraph, ConsumerKind, Retry
from tests.python.support.collective_checks import collective_call, collective_check
from tests.python.support.evolved_stage_v_capture import pin, retain_v_provenance, save_json
from tests.python.support.integral_state_receipts import collective_directory
from tests.python.integration.runtime.test_imex_checkpoint_publication_rollback import _save_science

pytestmark = [pytest.mark.compiler, pytest.mark.native_loader]
EXAMPLE = Path(__file__).resolve().parents[4]/'examples/final/EXEMPLE_SPEC_FINALE_ADVECTION_IMEX_AMR.py'
FAULT = 'original IMEX first checkpoint publication failed after its real link'


def _retry_target(example, *, output_mode):
    """Original public authoring helpers; only the resolved checkpoint policy is different."""
    from pops.mesh import normalize_layout_plan
    core = example.build_authoring(use_preset=False)
    core.numerics.boundaries.add(example.build_boundaries(core))
    core.case.numerics(core.numerics, block=core.tracer)
    core.case.initials.add(example.build_initial(core))
    core.case.program(core.program)
    layout = example.build_layout(core)
    subjects = core.case.layout_subjects()
    layout_plan = normalize_layout_plan(layout, owner=core.case.owner_path.canonical(),
        states=subjects.states, fields=subjects.fields, blocks=subjects.blocks,
        handle_resolver=core.case.resolve)
    graph = example.build_consumers(core, output_mode=output_mode).resolve(
        core.case.resolve, layout_plan, owner=core.case.owner_path.canonical())
    checkpoints = [node for node in graph.nodes if node.kind is ConsumerKind.CHECKPOINT]
    assert len(checkpoints) == 1
    altered = tuple(replace(node, failure_action=Retry(2))
                    if node.kind is ConsumerKind.CHECKPOINT else node for node in graph.nodes)
    for old, new in zip(graph.nodes, altered, strict=True):
        assert replace(new, failure_action=old.failure_action) == old
    core.case.consumers(ConsumerGraph(altered))
    return example.FinalIMEXAMRCase(core, layout)


def test_original_imex_checkpoint_retry_replays_one_completed_seal(
        tmp_path, monkeypatch, record_property, isolated_native_cache, native_cxx, kokkos_root):
    del isolated_native_cache, native_cxx, kokkos_root
    from pops._native_selector import select_native_dimension
    from pops.output._restart_provider import _RestartSnapshot
    from pops.runtime._runtime_instance import RuntimeInstance
    native = select_native_dimension(2)
    world = native.mpi_world()
    rank = int(world.rank)
    native.native_execution_resource()
    directory = collective_directory(world, tmp_path/'imex-sealed-retry')
    collective_call(world, lambda: save_json(directory/('context-rank%d.json'%rank),
                                           native.runtime_environment_report()))
    spec = importlib.util.spec_from_file_location('real_original_imex_sealed_retry', EXAMPLE)
    example = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = example
    spec.loader.exec_module(example)
    target = collective_call(world, lambda: _retry_target(example, output_mode=example._native_output_mode()))
    resolved = collective_call(world, lambda: pops.resolve(pops.validate(target.authoring.case), layout=target.layout))
    artifact = collective_call(world, lambda: pops.compile(resolved))
    collective_call(world, lambda: retain_v_provenance(artifact, native, directory, rank))
    params = example.build_bind_params(target.authoring)
    bind = lambda: example._bind_artifact(artifact, params=params)
    reference = collective_call(world, bind)
    simulation = collective_call(world, bind)
    controls = dict(target.authoring.run_controls)
    reference_controls = dict(controls, output_dir=directory/'reference-output')
    reference_result = collective_call(world, lambda: pops.run(reference, **reference_controls))
    with collective_check(world):
        assert reference_result.accepted_steps > 0
    expected = collective_call(world, lambda: example._snapshot(reference))
    collective_call(world, lambda: _save_science(directory, 'reference-rank%d'%rank, expected))

    captures, factories, publications = [], [], []
    original_capture = RuntimeInstance._checkpoint_payload
    original_factory = RuntimeInstance._prepare_checkpoint_candidate
    original_publish = _RestartSnapshot.publish

    def capture_once(owner, *args, **kwargs):
        if owner is simulation and kwargs.get('prepared_capture') is not None:
            captures.append('completed-candidate-sealed')
        return original_capture(owner, *args, **kwargs)

    def factory_once(owner):
        candidate = original_factory(owner)
        if owner is simulation and candidate is not None:
            factories.append('candidate-native-factory')
        return candidate

    def fail_first_real_publication(snapshot, destination):
        if snapshot._runtime is not simulation or snapshot._prepared_capture is None:
            return original_publish(snapshot, destination)
        with collective_check(world):
            assert snapshot._sealed_replay.contract == 'pops.checkpoint.sealed-replay@1'
            assert snapshot._prepared_capture.contract == 'pops.amr.prepared-checkpoint-capture@1'
        produced = original_publish(snapshot, destination)  # genuine committed-token validation
        def retain_attempt():
            if rank != 0:
                return None
            raw = Path(produced).read_bytes()
            retained = directory/('published-attempt%d.npz'%(len(publications)+1))
            retained.write_bytes(raw)
            assert len(raw) == snapshot._sealed_replay.size
            assert hashlib.sha256(raw).hexdigest() == snapshot._sealed_replay.sha256
            return dict(target=str(produced), retained=pin(retained),
                        source_inode=list(snapshot._proof.owner), source_bytes=len(raw),
                        sealed_sha256=snapshot._sealed_replay.sha256)
        evidence = collective_call(world, retain_attempt)
        publications.append(evidence)
        if len(publications) == 1:
            raise OSError(FAULT)  # identical participant failure; physical callback unchanged
        return produced

    output_root = directory/'retry-output'
    retry_controls = dict(controls, output_dir=output_root)
    with monkeypatch.context() as patch:
        patch.setattr(RuntimeInstance, '_checkpoint_payload', capture_once)
        patch.setattr(RuntimeInstance, '_prepare_checkpoint_candidate', factory_once)
        patch.setattr(_RestartSnapshot, 'publish', fail_first_real_publication)
        result = collective_call(world, lambda: pops.run(simulation, **retry_controls))
    accepted = collective_call(world, lambda: example._snapshot(simulation))
    collective_call(world, lambda: _save_science(directory, 'accepted-rank%d'%rank, accepted))
    with collective_check(world):
        assert result.accepted_steps > 0
        assert captures == ['completed-candidate-sealed']
        assert factories == ['candidate-native-factory']
        assert len(publications) == 2
        example._require_same_snapshot(expected, accepted, where='original IMEX publication retry')
        assert simulation.consumer_recoveries == ()
        if rank == 0:
            assert publications[0]['sealed_sha256'] == publications[1]['sealed_sha256']
            assert publications[0]['source_inode'] == publications[1]['source_inode']
            assert Path(publications[0]['retained']['path']).read_bytes() == Path(publications[1]['retained']['path']).read_bytes()
            assert not list(output_root.rglob('.pops-restart-*'))

    # Share the actual published path through the existing collective checkpoint operation.
    from pops.output._checkpoint_collective import checkpoint_topology, root_value
    published_path = root_value(checkpoint_topology(simulation), 'test published restart target',
                                lambda: publications[-1]['target'])
    resumed = collective_call(world, bind)
    collective_call(world, lambda: resumed.restart(published_path))
    restored = collective_call(world, lambda: example._snapshot(resumed))
    collective_call(world, lambda: _save_science(directory, 'restored-rank%d'%rank, restored))
    with collective_check(world):
        example._require_same_snapshot(accepted, restored, where='strict replayed checkpoint restart')
    continuation = dict(t_end=2.0*float(controls['t_end']), max_steps=int(controls['max_steps']))
    collective_call(world, lambda: pops.run(simulation, **continuation, output_dir=directory/'continuous'))
    collective_call(world, lambda: pops.run(resumed, **continuation, output_dir=directory/'restarted'))
    continuous = collective_call(world, lambda: example._snapshot(simulation))
    restarted = collective_call(world, lambda: example._snapshot(resumed))
    collective_call(world, lambda: _save_science(directory, 'continuous-rank%d'%rank, continuous))
    collective_call(world, lambda: _save_science(directory, 'restarted-rank%d'%rank, restarted))
    with collective_check(world):
        example._require_same_snapshot(continuous, restarted, where='original continuation after sealed retry')
        example._require_regrid_window(accepted, continuous, where='sealed retry continuation')
    collective_call(world, lambda: save_json(directory/('retry-rank%d.json'%rank), dict(
        captures=captures, factories=factories, publications=publications,
        original_oracles_unchanged=True, root_received=False)))
    record_property('imex_sealed_retry_evidence', str(directory))
