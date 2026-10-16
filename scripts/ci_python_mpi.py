#!/usr/bin/env python3
"""Closed MPI Python allocation and execution using one authenticated installed wheel."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ci_select_tests as selection

ROOT = Path(__file__).resolve().parents[1]
TEST_STEP_SECONDS = 35 * 60
LEGACY_PARENT_CASES = {
    'tests/python/integration/mpi/test_api_v02_continuation_migration.py': 1,
    'tests/python/integration/mpi/test_amr_rank_change_restart.py': 1,
    'tests/python/integration/mpi/test_initial_field_ghost_mpi.py': 2,
}


def make_plan(manifest=None):
    manifest = selection.load_manifest() if manifest is None else manifest
    entries = [row for row in selection.manifest_python_mpi_entrypoints(manifest)
               if row.get('native_dimension', 2) == 2]
    drivers = selection.manifest_python_mpi_orchestrators(manifest)
    pairs, groups = {}, []
    for row in drivers:
        path = row['path']
        match = re.fullmatch(r'(test_public_.+)_mpi([12])\.py', Path(path).name)
        if match:
            worlds = pairs.setdefault(match[1], {})
            world = int(match[2])
            if world in worlds:
                raise ValueError('MPI profile duplicates a world driver: ' + path)
            worlds[world] = row
        elif path in LEGACY_PARENT_CASES:
            if 'partitions' in row:
                raise ValueError('Legacy MPI driver has no partition authority: ' + path)
            groups.append({'group': 'driver-' + Path(path).stem, 'launches': [{
                'kind': 'orchestrator', 'path': path, 'timeout_seconds': 1500,
                'parent_cases': LEGACY_PARENT_CASES[path]}]})
        else:
            raise ValueError('MPI orchestrator lacks an explicit bounded allocation: ' + path)
    for stem, worlds in sorted(pairs.items()):
        if set(worlds) != {1, 2}:
            raise ValueError('MPI profile requires both mandatory world drivers: ' + stem)
        partitions = [worlds[world].get('partitions') for world in (1, 2)]
        if partitions == [None, None]:
            groups.append({'group': 'worlds-' + stem, 'launches': [
                {'kind': 'orchestrator', 'path': worlds[world]['path'], 'timeout_seconds': 900,
                 'parent_cases': 1} for world in (1, 2)]})
            continue
        if (any(partition is None for partition in partitions)
                or [row['id'] for row in partitions[0]] != [row['id'] for row in partitions[1]]):
            raise ValueError('MPI profile partitions differ between mandatory worlds: ' + stem)
        for index, partition in enumerate(partitions[0]):
            groups.append({'group': 'worlds-' + stem + '-' + partition['id'], 'launches': [
                {'kind': 'orchestrator', 'path': worlds[world]['path'], 'timeout_seconds': 900,
                 'parent_cases': 1, 'partition_id': partition['id'],
                 'parent_node': partitions[world - 1][index]['parent_node']}
                for world in (1, 2)]})
    for row in entries:
        groups.append({'group': 'ranked-' + Path(row['path']).stem, 'launches': [{
            'kind': 'ranked', 'path': row['path'], 'nproc': row['nproc'],
            'timeout_seconds': 1200}]})
    groups.sort(key=lambda row: row['group'])
    assigned = [(launch['path'], launch.get('parent_node', ''))
                for group in groups for launch in group['launches']]
    declared = [(row['path'], partition['parent_node'])
                for row in drivers for partition in row.get('partitions', [])]
    declared.extend((row['path'], '') for row in entries + drivers if 'partitions' not in row)
    owners = {row['path'] for row in entries + drivers}
    if (sorted(assigned) != sorted(declared) or len(assigned) != len(set(assigned))
            or {path for path, _node in assigned} != owners):
        raise ValueError('MPI allocations are not an exact disjoint manifest cover')
    if not groups or len({row['group'] for row in groups}) != len(groups):
        raise ValueError('MPI allocations have empty or duplicate group names')
    for group in groups:
        ceiling = sum(launch['timeout_seconds'] + 30 for launch in group['launches'])
        if ceiling > TEST_STEP_SECONDS:
            raise ValueError('MPI allocation exceeds the unchanged 35-minute test step')
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    if not re.fullmatch('[0-9a-f]{40}', revision):
        raise ValueError('MPI allocation requires an exact checkout revision')
    return {'schema': 'pops.mpi-python-allocation@1', 'source_revision': revision,
            'manifest_sha256': hashlib.sha256(selection.MANIFEST.read_bytes()).hexdigest(),
            'native_dimension': 2, 'test_step_seconds': TEST_STEP_SECONDS,
            'entrypoint_count': len(entries), 'orchestrator_count': len(drivers),
            'contract_count': len(owners), 'parent_launch_count': len(assigned), 'groups': groups}


def validate_plan(path):
    plan = json.loads(Path(path).read_text())
    if plan != make_plan():
        raise ValueError('MPI allocation differs from the exact current manifest/revision')
    return plan


def installed_environment():
    environment = os.environ.copy()
    for key in ('PYTHONPATH', 'PYTHONOPTIMIZE', 'POPS_NATIVE_VARIANTS_ROOT', 'POPS_INCLUDE',
                'POPS_CI_NATIVE_PACKAGE', 'PYTEST_ADDOPTS', 'PIP_PREFIX', 'PIP_TARGET', 'PIP_USER'):
        environment.pop(key, None)
    environment.update(POPS_NATIVE_DIM='2', POPS_REQUIRE_NATIVE_TESTS='1',
                       POPS_REQUIRE_MPI_TESTS='1', PYTHONNOUSERSITE='1',
                       POPS_CI_TEST_EXECUTION='installed')
    return environment


def authenticate_install(wheel_directory, install_root, receipt_directory, *, existing=False):
    wheel_directory, prefix = Path(wheel_directory).resolve(), Path(install_root).resolve()
    receipts = Path(receipt_directory)
    if wheel_directory.is_relative_to(ROOT) or prefix.is_relative_to(ROOT):
        raise ValueError('MPI wheel and installation must both remain outside checkout')
    wheels = list(wheel_directory.glob('pops-*.whl'))
    if len(wheels) != 1 or wheels[0].is_symlink():
        raise ValueError('MPI artifact must contain one exact retained wheel')
    wheel = wheels[0]
    assembly = json.loads((wheel_directory / 'assembly-receipt.json').read_text())
    if (type(assembly.get('dimension')) is not int or assembly['dimension'] != 2
            or assembly.get('backend') != 'mpi'
            or hashlib.sha256(wheel.read_bytes()).hexdigest() != assembly.get('wheel_sha256')):
        raise ValueError('MPI wheel bytes/backend/dimension differ from the assembly receipt')
    if prefix.exists() is not existing:
        raise ValueError('MPI authentication requires the declared fresh/existing installation state')
    receipts.mkdir(parents=True, exist_ok=True)
    python = prefix / 'bin/python'
    environment = installed_environment()
    sdk_bootstrap = (
        'from pops._native_selector import select_native_dimension; select_native_dimension(2); '
        'from pops.codegen import abi,toolchain; import json,sys; '
        'signature=toolchain.pops_header_signature(toolchain.pops_include()); '
        'assert signature==abi.module_header_signature()==sys.argv[1], "MPI SDK/native identity differs"; '
        'selection=toolchain._native_kokkos_selection(); '
        'assert selection is not None, "MPI Kokkos SDK missing"; '
        'flags,links=toolchain._native_kokkos_flags(); '
        'print(json.dumps({"header_signature":signature,"kokkos_prefix":selection[0], '
        '"compile_flags":flags,"link_flags":links}))'
    )
    stages = [
        ('venv', [sys.executable, '-m', 'venv', '--system-site-packages', str(prefix)]),
        ('pip', [str(python), '-m', 'pip', '--isolated', 'install', '--ignore-installed',
                 '--no-deps', '--prefix', str(prefix), str(wheel)]),
        ('installed-wheel-proof', [str(python), '-I', str(ROOT / 'scripts/prove_installed_wheel.py'),
                                   '--wheel', str(wheel), '--expect-dim', '2']),
        ('installed-native-proof', [str(python), '-I', str(ROOT / 'scripts/verify_installed_native.py'),
                                    '--expect-dim', '2', '--expect-mpi', '--expect-parallel-hdf5', '--json']),
        ('installed-sdk-proof', [str(python), '-I', '-c', sdk_bootstrap, assembly['SDK']]),
    ]
    if existing:
        stages = stages[2:]
    outcomes = []
    for name, command in stages:
        with (receipts / (name + '.log')).open('w') as output:
            result = subprocess.run(command, cwd=ROOT, env=environment, stdin=subprocess.DEVNULL,
                                    stdout=output, stderr=None, check=False)
        outcomes.append({'stage': name, 'argv': command, 'returncode': result.returncode})
        (receipts / 'installation-stages.json').write_text(json.dumps(outcomes, indent=2) + '\n')
        if result.returncode:
            raise RuntimeError('MPI installation authentication failed: ' + name)
    native = json.loads((receipts / 'installed-native-proof.log').read_text())
    if native['sha256'] != assembly['native_sha256'] or native['has_mpi'] is not True \
            or native['has_parallel_hdf5'] is not True:
        raise RuntimeError('MPI installed native facts differ from the exact wheel receipt')
    return python, environment


def require_parent_receipt(path, expected, *, launch=None):
    cases = list(ET.parse(path).getroot().iter('testcase'))
    identities = {(case.attrib.get('classname'), case.attrib['name']) for case in cases}
    if (len(cases) != expected or len(identities) != len(cases)
            or any(case.find(tag) is not None for case in cases
                   for tag in ('failure', 'error', 'skipped'))):
        raise RuntimeError('MPI driver did not receive every mandatory parent case')
    if launch is not None and 'parent_node' in launch:
        identity = (str(Path(launch['path']).with_suffix('')).replace('/', '.'), launch['parent_node'])
        if expected != 1 or identities != {identity}:
            raise RuntimeError('MPI driver receipt differs from its exact manifest parent node')


def run_launches(group, python, environment, receipts):
    receipts = Path(receipts)
    receipts.mkdir(parents=True, exist_ok=True)
    launcher = shutil.which('mpiexec', path=environment.get('PATH'))
    if launcher is None:
        raise RuntimeError('required MPI launcher is unavailable')
    version = subprocess.check_output([launcher, '--version'], env=environment, text=True)
    mpi_flags = ['--mca', 'orte_abort_on_non_zero_status', '1'] if re.search('Open MPI|OpenRTE', version, re.I) else []
    bootstrap = (
        'from pops._native_selector import select_native_dimension; select_native_dimension(2); '
        'import os,runpy,sys; sys.argv=sys.argv[1:]; sys.path.insert(0,os.getcwd()); '
        'sys.path.insert(0,os.path.dirname(os.path.abspath(sys.argv[0]))); '
        "runpy.run_path(sys.argv[0],run_name='__main__')"
    )
    pytest_bootstrap = (
        'from pops._native_selector import select_native_dimension; select_native_dimension(2); '
        'import os,sys,pytest; sys.path.insert(0,os.getcwd()); '
        'raise SystemExit(pytest.main(sys.argv[1:]))'
    )
    status, outcomes = 0, []
    for index, launch in enumerate(group['launches']):
        xml = receipts.resolve() / ('driver-%d.xml' % index)
        if xml.exists():
            raise RuntimeError('MPI group receipt directory contains stale parent outcomes')
        if launch['kind'] == 'ranked':
            child = [launcher, *mpi_flags, '-n', str(launch['nproc']), str(python),
                     '-u', '-c', bootstrap, launch['path']]
        else:
            selector = launch['path'] + ('::' + launch['parent_node'] if 'parent_node' in launch else '')
            child = [str(python), '-I', '-u', '-c', pytest_bootstrap, '-q', '-ra',
                     '--maxfail=1', '-o', 'pythonpath=', '--junitxml=' + str(xml), selector]
        command = ['timeout', '--signal=TERM', '--kill-after=30s',
                   str(launch['timeout_seconds']) + 's', *child]
        print('MPI group %s: %s' % (group['group'], launch['path']), flush=True)
        result = subprocess.run(command, cwd=ROOT, env=environment, stdin=subprocess.DEVNULL, check=False)
        code = result.returncode
        if code == 0 and launch['kind'] == 'orchestrator':
            try:
                require_parent_receipt(xml, launch['parent_cases'], launch=launch)
            except (OSError, ValueError, ET.ParseError, RuntimeError) as error:
                print(str(error), file=sys.stderr)
                code = 1
        outcomes.append({'launch': launch, 'argv': command, 'returncode': code})
        status = status or code
        (receipts / 'launches.json').write_text(json.dumps(outcomes, indent=2) + '\n')
    return status


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    plan = sub.add_parser('plan')
    plan.add_argument('--plan-file', required=True, type=Path)
    plan.add_argument('--github-output')
    for name in ('authenticate', 'run'):
        command = sub.add_parser(name)
        command.add_argument('--wheel-directory', required=True, type=Path)
        command.add_argument('--install-root', required=True, type=Path)
        command.add_argument('--receipt-dir', required=True, type=Path)
        if name == 'run':
            command.add_argument('--plan-file', required=True, type=Path)
            command.add_argument('--group', required=True)
    args = parser.parse_args(argv)
    if args.command == 'plan':
        data = make_plan()
        args.plan_file.parent.mkdir(parents=True, exist_ok=True)
        args.plan_file.write_text(json.dumps(data, indent=2) + '\n')
        matrix = {'include': [{'group': row['group']} for row in data['groups']]}
        selection.write_github_outputs(args.github_output, {'matrix': json.dumps(matrix, separators=(',', ':')),
                                      'group_count': str(len(data['groups']))})
        print('%d MPI groups cover %d exact contracts' % (len(data['groups']), data['contract_count']))
        return 0
    try:
        if args.command == 'run':
            data = validate_plan(args.plan_file)
            groups = [row for row in data['groups'] if row['group'] == args.group]
            if len(groups) != 1:
                raise ValueError('MPI worker does not own exactly one declared group')
        python, environment = authenticate_install(
            args.wheel_directory, args.install_root,
            args.receipt_dir / 'execution-auth' if args.command == 'run' else args.receipt_dir,
            existing=args.command == 'run')
        if args.command == 'run':
            return run_launches(groups[0], python, environment, args.receipt_dir)
        return 0
    except (OSError, ValueError, RuntimeError, KeyError, TypeError, subprocess.CalledProcessError) as error:
        print('MPI group failed: ' + str(error), file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
