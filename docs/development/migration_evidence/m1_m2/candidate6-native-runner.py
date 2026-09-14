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
summary = dict(source_sha=PIN, dimension=2, mpi_enabled=True,
               kokkos_backends=['OPENMP', 'SERIAL'], compiler='/usr/bin/clang++',
               build_type='Release', build_parallelism=2,
               environment={key: env[key] for key in ('OMP_NUM_THREADS', 'OMP_PROC_BIND',
                   'Kokkos_ROOT', 'POPS_KOKKOS_ROOT', 'POPS_NATIVE_DIM')}, steps=[],
               status='running', limitations=['Requested 14-target native matrix only; no Python wheel qualification.'])

def save():
    (OUT / 'candidate6-native-summary.json').write_text(json.dumps(summary, indent=2) + '\n')

def raw(*args):
    return subprocess.check_output(['rtk', 'proxy', *args], cwd=ROOT, env=env).decode()

def freeze():
    head = raw('git', 'rev-parse', 'HEAD').strip()
    dirty = raw('git', 'status', '--porcelain', '--untracked-files=no').strip()
    if head != PIN or dirty:
        raise RuntimeError(f'frozen source changed: head={head}, dirty={dirty!r}')
    digest = hashlib.sha256()
    for name in raw('git', 'ls-files', '-z').split('\0'):
        if name:
            digest.update(name.encode() + b'\0' + hashlib.sha256((ROOT / name).read_bytes()).digest())
    return digest.hexdigest()

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
    summary['start_tracked_bytes_sha256'] = freeze()
    summary['compiler_version'] = raw('/usr/bin/clang++', '--version').strip()
    save()
    run('configure', [ENV / 'bin/cmake', '-S', ROOT, '-B', BUILD,
        '-DCMAKE_BUILD_TYPE=Release', '-DCMAKE_CXX_COMPILER=/usr/bin/clang++',
        '-DCMAKE_C_COMPILER=/usr/bin/clang', f'-DCMAKE_PREFIX_PATH={ENV}',
        f'-DKokkos_ROOT={ENV}', '-DPOPS_NATIVE_DIM=2', '-DPOPS_REAL_TYPE=double',
        '-DPOPS_BUILD_TESTS=ON', '-DPOPS_BUILD_PYTHON=OFF',
        '-DPOPS_USE_KOKKOS=ON', '-DPOPS_USE_MPI=ON', '-DPOPS_USE_HDF5=ON'])
    run('build', [ENV / 'bin/cmake', '--build', BUILD, '--clean-first',
                  '--parallel', '2', '--target', *TARGETS])
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
    summary['end_tracked_bytes_sha256'] = freeze()
    assert summary['end_tracked_bytes_sha256'] == summary['start_tracked_bytes_sha256']
    summary['verified_end_head'] = PIN
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
