"""Exercise orchestration with real subprocesses/files; no PoPS/JIT or perf observations."""
from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

DOC = Path(__file__).resolve().parents[4] / "docs/development/api_040"


def load(version):
    path = DOC / f"joint_reconstruction_resource_probe_{version}.py"
    spec = importlib.util.spec_from_file_location("resource_" + version, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# This child is a protocol test double, NOT a solver or benchmark. The driver and
# both subprocess/cache/receipt implementations remain the actual versioned tool.
CHILD = r'''
import argparse,json,os
from pathlib import Path
p=argparse.ArgumentParser()
p.add_argument('--lane');p.add_argument('--phase',default='profiling')
p.add_argument('--output');a,_=p.parse_known_args()
cache=Path(os.environ['POPS_CACHE_DIR']);codegen=Path(os.environ['POPS_CODEGEN_DIR'])
assert cache.is_dir() and codegen.is_dir()
assert not list(cache.iterdir()) and not list(codegen.iterdir())
event={'phase':a.phase,'lane':a.lane,'cache':str(cache),'codegen':str(codegen)}
with open(os.environ['DRIVER_EVENTS'],'a') as stream:stream.write(json.dumps(event)+'\n')
mode=os.environ.get('DRIVER_MODE','')
if a.phase=='metadata':
 row={'abi_environment':{'mpi':'same'},'sdk_version':{'version':'same'},'artifact_properties':['abi_key']}
 if mode=='metadata_mismatch' and a.lane=='candidate':row['abi_environment']['mpi']='other'
 if mode=='dirty_profile' and a.lane=='candidate':
  dirty=Path(os.environ['PROFILE_ROOT'])/'cache'/'baseline'
  dirty.mkdir(parents=True);(dirty/'retained.so').write_bytes(b'foreign-cache')
else:
 import numpy as np
 state=Path(a.output).with_suffix('.npy')
 np.save(state,np.array([1., 3. if mode=='state_mismatch' and a.lane=='candidate' else 2.]))
 units=json.loads(os.environ['COUNTER_NAMES'])
 selected={name:{'available':False,'reason':'absent'} for name in units}
 selected['kernels']={'available':True,'value':6}
 row={'platform_facts':{'precision':'float64'},'abi_environment':{'mpi':'same'},
      'compiler':'same','cxx_standard':'c++20','sdk_version':{'version':'same'},
      'loaded_external_images':{'available':True,'images':{'libmpi':'same'}},
      'state_file':str(state),'samples':[{'selected_counters':selected} for _ in range(3)]}
 (cache/'observed.dat').write_bytes(b'worker cache')
Path(a.output).write_text(json.dumps(row))
'''


def scenario(tmp_path, monkeypatch, version="v1_3", mode=""):
    tool = load(version)
    monkeypatch.setattr(tool, "sys", SimpleNamespace(platform="darwin"))
    # Snapshot cryptography is tested separately against the actual snapshots.
    # Here exact dummy identities isolate only the orchestration under test.
    monkeypatch.setattr(tool, "_identity", lambda *args: {
        "source_commit": "fixture-commit", "native_sha256": "fixture-native",
        "source_files_sha256": "fixture-manifest"})
    output = tmp_path / "run"
    events = tmp_path / "events.jsonl"
    child = tmp_path / "worker.py"
    child.write_text(CHILD)
    monkeypatch.setenv("DRIVER_EVENTS", str(events))
    monkeypatch.setenv("DRIVER_MODE", mode)
    monkeypatch.setenv("PROFILE_ROOT", str(output / "profiling"))
    monkeypatch.setenv("COUNTER_NAMES", json.dumps(list(tool._COUNTER_UNITS)))
    real_popen = subprocess.Popen

    def launch(command, **kwargs):
        assert command[1] == "-I" and command[3] == "--worker"
        assert "PYTHONPATH" not in kwargs["env"]
        assert "POPS_INCLUDE" not in kwargs["env"]
        return real_popen([*command[:2], str(child), *command[3:]], **kwargs)

    monkeypatch.setattr(subprocess, "Popen", launch)
    args = SimpleNamespace(baseline_root=tmp_path / "baseline", candidate_root=tmp_path / "candidate",
                           python=sys.executable, output=output, execute=True)
    return tool, args, events


def read_events(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_v12_driver_reproduces_collision_after_two_metadata_workers(tmp_path, monkeypatch):
    tool, args, events = scenario(tmp_path, monkeypatch, "v1_2")
    with pytest.raises(FileExistsError):
        tool._driver(args)
    assert [(row['phase'], row['lane']) for row in read_events(events)] == [
        ("metadata", "baseline"), ("metadata", "candidate")]
    assert not (args.output / "result.json").exists()


def test_v13_driver_separates_caches_and_completes_comparisons(tmp_path, monkeypatch):
    tool, args, events = scenario(tmp_path, monkeypatch)
    tool._driver(args)
    rows = read_events(events)
    assert [(row['phase'], row['lane']) for row in rows] == [
        ("metadata", "baseline"), ("metadata", "candidate"),
        ("profiling", "baseline"), ("profiling", "candidate")]
    assert len({row['cache'] for row in rows}) == 4
    assert len({row['codegen'] for row in rows}) == 4
    for row in rows[:2]:
        assert Path(row['cache']).is_relative_to(args.output / "preflight")
        assert not list(Path(row['cache']).iterdir())
    for row in rows[2:]:
        assert Path(row['cache']).is_relative_to(args.output / "profiling")
        assert (Path(row['cache']) / 'observed.dat').is_file()
    result = json.loads((args.output / 'result.json').read_text())
    assert result['schema'].endswith('.v1.3')
    assert result['numerical_equivalence']['passed']
    assert result['counter_comparison']['kernels']['values'] == {
        'baseline': [6, 6, 6], 'candidate': [6, 6, 6]}
    assert result['counter_comparison']['scratch_peak_bytes']['available'] is False
    with pytest.raises(ValueError, match='empty output directory'):
        tool._driver(args)
    assert read_events(events) == rows  # Refused restart does not launch workers.


@pytest.mark.parametrize('mode,error,phase_count', [
    ('dirty_profile', FileExistsError, 2),
    ('metadata_mismatch', ValueError, 2),
    ('state_mismatch', AssertionError, 4),
])
def test_driver_failures_never_publish_observed_result(tmp_path, monkeypatch, mode, error, phase_count):
    tool, args, events = scenario(tmp_path, monkeypatch, mode=mode)
    with pytest.raises(error):
        tool._driver(args)
    assert len(read_events(events)) == phase_count
    assert not (args.output / 'result.json').exists()
    if mode == 'dirty_profile':
        assert (args.output / 'profiling/cache/baseline/retained.so').read_bytes() == b'foreign-cache'


def test_measurement_and_identity_functions_are_unchanged():
    before = ast.parse((DOC / 'joint_reconstruction_resource_probe_v1_2.py').read_text())
    after = ast.parse((DOC / 'joint_reconstruction_resource_probe_v1_3.py').read_text())
    for name in ('_worker', '_profile_once', '_invoke', '_identity', '_frozen_v1_2', '_aggregate'):
        old = next(node for node in before.body if isinstance(node, ast.FunctionDef) and node.name == name)
        new = next(node for node in after.body if isinstance(node, ast.FunctionDef) and node.name == name)
        assert ast.dump(old) == ast.dump(new), name
    a, b = load('v1_2'), load('v1_3')
    for name in ('V1_2_SHA256', 'WARMUPS', 'PROFILED_RUNS', 'WORKER_TIMEOUT_S', '_COUNTER_UNITS',
                 'CANDIDATE_COMMIT', 'CANDIDATE_NATIVE', 'CANDIDATE_SOURCES'):
        assert getattr(a, name) == getattr(b, name)
