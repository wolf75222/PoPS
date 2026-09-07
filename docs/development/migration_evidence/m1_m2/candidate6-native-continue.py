"""Frozen candidate-six native qualification; writes only build/evidence outputs."""
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time
import xml.etree.ElementTree as ET

ROOT = Path('/Users/romaindespoulain/dev/tmp/PoPS-migration-20260907-m2')
OUT = Path('/Users/romaindespoulain/dev/tmp/PoPS-migration-20260907-evidence')
ENV = Path('/Users/romaindespoulain/miniforge3/envs/pops-migration-20260907')
BUILD = ROOT / 'build/migration-native'
PIN = 'a760245b004008cd464ad44ef7b5c629c3e728db'
TARGETS = tuple(json.loads((OUT / 'candidate5-native-summary.json').read_text())['targets'])
env = dict(os.environ, OMP_NUM_THREADS='2', OMP_PROC_BIND='false',
           Kokkos_ROOT=str(ENV), POPS_KOKKOS_ROOT=str(ENV), POPS_NATIVE_DIM='2')
env.pop('PYTHONPATH', None)
env['PATH'] = str(ENV / 'bin') + os.pathsep + env['PATH']
CURRENT_PIN = 'bd583faf196f3c1faeedec489e04d24e959ed00b'
ALLOWED = 'python/pops/codegen/program_emit_ops.py'
summary = json.loads((OUT / 'candidate6-native-initial-source-stop.json').read_text())
summary['status'] = 'running'
summary.pop('error', None)
summary['initial_source_change_receipt'] = 'candidate6-native-initial-source-stop.json'
summary['candidate7_native_source_equivalence'] = dict(
    native_source_pin=PIN, candidate7_source_pin=CURRENT_PIN,
    exact_changed_path=ALLOWED,
    scope='One Python model-free projection guard; every other tracked byte must match the initial source fingerprint.')

def save():
    (OUT / 'candidate6-native-summary.json').write_text(json.dumps(summary, indent=2) + '\n')

def raw(*args):
    return subprocess.check_output(['rtk', 'proxy', *args], cwd=ROOT, env=env).decode()

def freeze():
    head = raw('git', 'rev-parse', 'HEAD').strip()
    dirty = raw('git', 'status', '--porcelain', '--untracked-files=no').strip()
    changed = raw('git', 'diff', '--name-only', PIN, CURRENT_PIN).splitlines()
    if head != CURRENT_PIN or dirty or changed != [ALLOWED]:
        raise RuntimeError(f'equivalent source changed: head={head}, dirty={dirty!r}, delta={changed!r}')
    baseline_digest = hashlib.sha256()
    current_digest = hashlib.sha256()
    original = raw('git', 'show', f'{PIN}:{ALLOWED}').encode()
    for name in raw('git', 'ls-files', '-z').split('\0'):
        if name:
            content = (ROOT / name).read_bytes()
            current_digest.update(name.encode() + b'\0' + hashlib.sha256(content).digest())
            baseline_digest.update(name.encode() + b'\0' + hashlib.sha256(original if name == ALLOWED else content).digest())
    summary['candidate7_native_source_equivalence'].update(
        candidate7_tracked_bytes_sha256=current_digest.hexdigest(),
        restored_baseline_tracked_bytes_sha256=baseline_digest.hexdigest(),
        baseline_python_file_sha256=hashlib.sha256(original).hexdigest(),
        candidate7_python_file_sha256=hashlib.sha256((ROOT / ALLOWED).read_bytes()).hexdigest())
    return baseline_digest.hexdigest()

def run(label, args, timeout=7200):
    step = dict(label=label, command=['rtk', 'proxy', *map(str, args)],
                started=time.time(), log=f'candidate6-native-{label}.log')
    summary['steps'].append(step)
    save()
    print(f'{label}: started', flush=True)
    with (OUT / step['log']).open('w') as log:
        result = subprocess.run(step['command'], cwd=ROOT, env=env,
                                stdout=log, stderr=subprocess.STDOUT, timeout=timeout)
    step.update(exit_code=result.returncode, elapsed_seconds=time.time() - step['started'])
    save()
    print(f'{label}: exit={result.returncode} elapsed={step["elapsed_seconds"]:.1f}s', flush=True)
    if result.returncode:
        raise RuntimeError(f'{label} failed; see {step["log"]}')
    return OUT / step['log']

try:
    assert freeze() == summary['start_tracked_bytes_sha256']
    summary['compiler_version'] = raw('/usr/bin/clang++', '--version').strip()
    save()
    assert summary['steps'][-1]['label'] == 'build' and summary['steps'][-1]['exit_code'] == 0
    assert freeze() == summary['start_tracked_bytes_sha256']
    summary['targets'] = {target: hashlib.sha256((BUILD / 'bin' / target).read_bytes()).hexdigest()
                          for target in TARGETS}
    inventory = json.loads(raw(str(ENV / 'bin/ctest'), '--test-dir', str(BUILD), '--show-only=json-v1'))
    (OUT / 'candidate6-native-ctest-inventory.json').write_text(json.dumps(inventory, indent=2) + '\n')
    serial = [test for test in inventory['tests'] if test.get('command')
              and Path(test['command'][0]).name in TARGETS]
    mpi = [test for test in inventory['tests'] if test.get('command')
           and Path(test['command'][0]).name in ('mpiexec', 'mpirun')
           and any(Path(arg).name in TARGETS for arg in test['command'][1:])]
    serial_targets = {Path(test['command'][0]).name for test in serial}
    assert serial_targets == set(TARGETS) - {'test_mpi_prepared_amr_ghost_fill'}, serial_targets
    assert {test['name'] for test in mpi} == {
        'test_mpi_prepared_amr_ghost_fill_np3', 'test_program_context_contract_np2',
        'test_generated_amr_system_block_np2', 'test_coupled_fieldsolve_np2'}
    selector = OUT / 'candidate6-native-serial-tests.txt'
    selector.write_text(''.join(test['name'] + '\n' for test in serial))
    serial_xml = OUT / 'candidate6-native-tests.xml'
    run('tests', [ENV / 'bin/ctest', '--test-dir', BUILD, '--tests-from-file', selector,
        '--output-on-failure', '--no-tests=error', '--parallel', '1', '--output-junit', serial_xml])
    junit = ET.parse(serial_xml).getroot()
    assert int(junit.attrib['tests']) == len(serial) > 0
    summary['serial'] = dict(junit.attrib)
    summary['serial_skips'] = [test.attrib['name'] for test in junit.findall('testcase')
                               if test.find('skipped') is not None]
    assert set(summary['serial_skips']) == {
        'ProgramContextContract.AuxiliaryReadRefusesRankDivergentPublicationBeforeNoWorkBranch',
        'ProgramContextContract.PreparedLinearSolveRefusesRankDivergentSameSolveIdLevelOwnerSelection',
        'ProgramContextContract.ApplyProjectionRefusesRankDivergentPreparedBlockRouteBeforeProviderInvocation'}
    save()
    mpi_xml = OUT / 'candidate6-native-mpi.xml'
    pattern = '^(' + '|'.join(re.escape(test['name']) for test in mpi) + ')$'
    mpi_log = run('mpi', [ENV / 'bin/ctest', '--test-dir', BUILD, '-R', pattern,
        '--verbose', '--no-tests=error', '--parallel', '1', '--output-junit', mpi_xml])
    summary['mpi_ctest'] = dict(ET.parse(mpi_xml).getroot().attrib)
    summary['mpi_effective_gtests'] = {}
    for test in mpi:
        output_arg = next(arg for arg in test['command'] if arg.startswith('--gtest_output=xml:'))
        source = Path(output_arg.removeprefix('--gtest_output=xml:'))
        destination = OUT / ('candidate6-native-' + test['name'] + '.xml')
        destination.write_bytes(source.read_bytes())
        record = ET.parse(destination).getroot()
        count = int(record.attrib['tests'])
        assert count > 0 and int(record.attrib.get('failures', 0)) == 0, record.attrib
        summary['mpi_effective_gtests'][test['name']] = count
    assert not re.search(r'Running 0 tests? from', mpi_log.read_text())
    summary['end_restored_baseline_tracked_bytes_sha256'] = freeze()
    assert summary['end_restored_baseline_tracked_bytes_sha256'] == summary['start_tracked_bytes_sha256']
    summary['verified_end_head'] = CURRENT_PIN
    summary['final_qualification'] = dict(
        serial_passes=len(serial) - len(summary['serial_skips']),
        serial_mpi_only_skips=len(summary['serial_skips']),
        effective_mpi_gtests=sum(summary['mpi_effective_gtests'].values()),
        mpi_ranks=[2, 3], remaining_failures=[])
    summary['status'] = 'passed'
except BaseException as error:
    summary['status'] = 'failed'
    summary['error'] = repr(error)
    raise
finally:
    save()
