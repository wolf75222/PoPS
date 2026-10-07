"""Preserve the real normative IMEX example's public lifecycle and returned evidence."""
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest

from tests.python.support.evolved_stage_v_capture import retain_v_provenance

pytestmark = [pytest.mark.compiler, pytest.mark.native_loader]
EXAMPLE = Path(__file__).resolve().parents[4] / 'examples/final/EXEMPLE_SPEC_FINALE_ADVECTION_IMEX_AMR.py'


def _save_snapshot(directory, phase, snapshot):
    arrays = {}
    metadata = {}
    for family in ('states', 'fields'):
        metadata[family] = {}
        for name, levels in getattr(snapshot, family).items():
            keys = []
            for level, value in enumerate(levels):
                key = '%s/%s/%d' % (family, name, level)
                arrays[key] = value
                keys.append(key)
            metadata[family][name] = keys
    np.savez(directory / (phase + '.npz'), **arrays)
    (directory / (phase + '-program.bin')).write_bytes(snapshot.program_accepted_state)
    for name in ('time', 'macro_step', 'patch_boxes', 'regrid_count', 'topology_epoch',
                 'program_hash', 'program_transaction_state', 'consumer_graph_identity',
                 'consumer_cursors'):
        metadata[name] = getattr(snapshot, name)
    (directory / (phase + '.json')).write_text(json.dumps(metadata, sort_keys=True, indent=2) + '\n')


@pytest.fixture(scope='module')
def original_manual(tmp_path_factory):
    from pops._native_selector import select_native_dimension
    native = select_native_dimension(2)
    before = native.runtime_environment_report()
    world = native.mpi_world()
    native.native_execution_resource()
    after = native.runtime_environment_report()
    assert world.active and world.rank == 0 and world.size == 1
    directory = tmp_path_factory.mktemp('imex-original')
    spec = importlib.util.spec_from_file_location('pops_final_imex_original_reception', EXAMPLE)
    example = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = example
    spec.loader.exec_module(example)
    (directory / 'context.json').write_text(json.dumps(dict(before=before, after=after), sort_keys=True, indent=2) + '\n')
    # This exact public compilation is the same input used by run_manual_and_restart.
    # Its second compile call uses the authenticated cache; no substitute runtime is bound.
    _target, _resolved, artifact = example.compile_final_case(use_preset=False)
    retain_v_provenance(artifact, native, directory, world.rank)
    evidence = example.run_manual_and_restart(directory / 'manual')
    assert evidence.artifact_identity == artifact.artifact_identity.token
    for phase in ('accepted', 'restored', 'continuous', 'restarted'):
        _save_snapshot(directory, phase, getattr(evidence, phase))
    (directory / 'manual-result.json').write_text(json.dumps(dict(
        artifact_identity=evidence.artifact_identity, level_count=evidence.level_count,
        hdf5_path=str(evidence.hdf5_path), paraview_path=str(evidence.paraview_path),
        checkpoint_path=str(evidence.checkpoint_path), hdf5_identity=evidence.hdf5_identity,
        paraview_identity=evidence.paraview_identity,
        flux_ledger_levels=evidence.program_evidence.flux_ledger_levels,
        synchronization_phases=evidence.program_evidence.synchronization_phases,
    ), sort_keys=True, indent=2) + '\n')
    return example, native, directory, evidence


def test_original_manual_state_field_restart(original_manual):
    _example, _native, _directory, evidence = original_manual
    # All scientific and bit-identical comparisons are executed in the original function.
    assert evidence.accepted.macro_step == 1
    assert evidence.restarted.macro_step == 2


def test_original_preset_parity_and_rejected_rollback(original_manual):
    example, native, directory, evidence = original_manual
    preset_dir = directory / 'preset-provenance'
    preset_dir.mkdir()
    _target, _resolved, artifact = example.compile_final_case(use_preset=True)
    retain_v_provenance(artifact, native, preset_dir, 0)
    preset = example.run_preset_parity(directory / 'preset', evidence.accepted)
    _save_snapshot(directory, 'preset', preset)
    rejected = example.run_rejected_attempt_rollback(directory / 'rejected')
    _save_snapshot(directory, 'rejected-before', rejected.before)
    _save_snapshot(directory, 'rejected-after', rejected.after)
    (directory / 'rejected-result.json').write_text(json.dumps(dict(error=rejected.error), sort_keys=True, indent=2) + '\n')
    assert rejected.error.startswith('step attempt rejected during ')
