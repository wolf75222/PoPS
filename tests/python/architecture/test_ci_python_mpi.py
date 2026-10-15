"""Source-only proofs of complete bounded MPI allocation and installed-wheel gates."""
import copy
import hashlib
import json
import os
import sys
import xml.etree.ElementTree as ET

import pytest

from scripts import ci_python_mpi as runner


def fake_timeout_environment(directory, *, failed_path='', failed_status=0):
    directory.mkdir(parents=True, exist_ok=True)
    launcher = directory / 'mpiexec'
    launcher.write_text('#!/bin/sh\nprintf "Open MPI source-policy stub\\n"\n')
    launcher.chmod(0o755)
    timeout = directory / 'timeout'
    timeout.write_text('#!' + sys.executable + '\n' + r'''
import json,os,pathlib,sys,xml.etree.ElementTree as ET
assert sys.stdin.read()==''
path=sys.argv[-1]
with open(os.environ['LAUNCH_LOG'],'a') as output:
    output.write(json.dumps({'path':path,'argv':sys.argv[1:]})+'\n')
if path==os.environ.get('FAIL_PATH'):
    raise SystemExit(int(os.environ['FAIL_STATUS']))
reports=[arg for arg in sys.argv if arg.startswith('--junitxml=')]
if reports:
    root=ET.Element('testsuite')
    count=2 if path.endswith('test_initial_field_ghost_mpi.py') else 1
    for index in range(count):
        if '::' in path:
            owner,node=path.split('::',1)
            ET.SubElement(root,'testcase',name=node,classname=owner.removesuffix('.py').replace('/','.'))
        else:
            ET.SubElement(root,'testcase',name='source_stub_case%d'%index)
    ET.ElementTree(root).write(reports[0].split('=',1)[1])
''')
    timeout.chmod(0o755)
    return {**os.environ, 'PATH': str(directory), 'LAUNCH_LOG': str(directory / 'launches.jsonl'),
            'FAIL_PATH': failed_path, 'FAIL_STATUS': str(failed_status)}


def test_exact_groups_cover_all_declared_contracts_and_nodes_with_existing_caps():
    plan = runner.make_plan()
    assert (len(plan['groups']), plan['contract_count'], plan['orchestrator_count'],
            plan['entrypoint_count']) == (114, 115, 29, 86)
    assert plan['parent_launch_count'] == 139
    launches = [launch for group in plan['groups'] for launch in group['launches']]
    assert len({(launch['path'], launch.get('parent_node', '')) for launch in launches}) == 139
    assert len({launch['path'] for launch in launches}) == 115
    pairs = [row for row in plan['groups'] if row['group'].startswith('worlds-')]
    assert len(pairs) == 25
    for pair in pairs:
        assert len(pair['launches']) == 2
        assert [row['path'][-7:] for row in pair['launches']] == ['mpi1.py', 'mpi2.py']
        assert all(row['timeout_seconds'] == 900 and row['parent_cases'] == 1 for row in pair['launches'])
    for group in plan['groups']:
        assert sum(row['timeout_seconds'] + 30 for row in group['launches']) <= 35 * 60
        for row in group['launches']:
            if row['kind'] == 'ranked':
                assert row['timeout_seconds'] == 1200 and row['nproc'] == 2
            elif group['group'].startswith('driver-'):
                assert row['timeout_seconds'] == 1500
    assert all('test_public_ns_conduction' not in row['path'] for group in plan['groups']
               for row in group['launches'])


def test_allocation_refuses_an_incomplete_mandatory_world_pair():
    manifest = copy.deepcopy(runner.selection.load_manifest())
    suite = next(row for row in manifest['python']['suite'] if row['name'] == 'pops_python_integration_mpi')
    suite['mpi_orchestrators'] = [row for row in suite['mpi_orchestrators']
                                 if not row['path'].endswith('test_public_typed_newton_mpi2.py')]
    with pytest.raises(ValueError, match='both mandatory world drivers'):
        runner.make_plan(manifest)


@pytest.mark.parametrize('mutation', ('remove_field', 'dual_omission', 'unknown', 'duplicate',
                                     'wrong_function', 'wrong_world', 'unowned_wrapper', 'unknown_field'))
@pytest.mark.parametrize('family', ('amr', 'original', 'newton'))
def test_partition_contract_cannot_omit_or_invent_native_coverage(mutation, family):
    manifest = copy.deepcopy(runner.selection.load_manifest())
    suite = next(row for row in manifest['python']['suite'] if row['name'] == 'pops_python_integration_mpi')
    prefix = {'amr': 'test_public_amr_profile_mpi',
              'original': 'test_public_amr_original_field_mpi',
              'newton': 'test_public_typed_newton_mpi'}[family]
    owners = [row for row in suite['mpi_orchestrators'] if prefix in row['path']]
    row = owners[0]
    parent_function = row['partitions'][0]['parent_node'].split('[', 1)[0]
    first_id = row['partitions'][0]['id']
    if mutation == 'remove_field':
        row.pop('partitions')
    elif mutation == 'dual_omission':
        for owner in owners:
            owner['partitions'].pop()
    elif mutation == 'unknown':
        row['partitions'][0]['id'] = 'unknown'
        row['partitions'][0]['parent_node'] = parent_function + '[unknown]'
    elif mutation == 'duplicate':
        row['partitions'].append(copy.deepcopy(row['partitions'][0]))
    elif mutation == 'wrong_function':
        row['partitions'][0]['parent_node'] = 'test_foreign[' + first_id + ']'
    elif mutation == 'wrong_world':
        row['partitions'][0]['parent_node'] = parent_function.replace('mpi1', 'mpi2') + '[' + first_id + ']'
    elif mutation == 'unowned_wrapper':
        other = next(owner for owner in suite['mpi_orchestrators'] if 'test_public_captured_diffusion_mpi1' in owner['path'])
        other['partitions'] = copy.deepcopy(row['partitions'])
    else:
        row['partitions'][0]['watchdog'] = 9999
    with pytest.raises(SystemExit, match='MPI (?:orchestrator )?partition'):
        runner.make_plan(manifest)


def test_partition_parent_xml_requires_exact_owned_module_and_selected_node(tmp_path):
    launches = [launch for group in runner.make_plan()['groups'] for launch in group['launches']
                if 'parent_node' in launch]
    assert len(launches) == 30
    for launch in launches:
        root = ET.Element('testsuite')
        case = ET.SubElement(root, 'testcase', name=launch['parent_node'],
                             classname=launch['path'].removesuffix('.py').replace('/', '.'))
        path = tmp_path / 'parent.xml'
        ET.ElementTree(root).write(path)
        runner.require_parent_receipt(path, 1, launch=launch)
        for attribute, wrong in (('name', 'test_wrong[tag_selection]'), ('classname', 'unowned.module')):
            original = case.attrib[attribute]
            case.set(attribute, wrong)
            ET.ElementTree(root).write(path)
            with pytest.raises(RuntimeError, match='exact manifest parent node'):
                runner.require_parent_receipt(path, 1, launch=launch)
            case.set(attribute, original)


@pytest.mark.parametrize('mutation', ('missing', 'duplicate', 'unowned', 'revision', 'manifest', 'watchdog'))
def test_worker_refuses_modified_or_stale_allocation_before_installation(tmp_path, monkeypatch, mutation):
    plan = runner.make_plan()
    if mutation == 'missing':
        plan['groups'].pop()
    elif mutation == 'duplicate':
        plan['groups'].append(copy.deepcopy(plan['groups'][0]))
    elif mutation == 'unowned':
        plan['groups'][0]['launches'][0]['path'] = 'tests/unowned.py'
    elif mutation == 'watchdog':
        plan['groups'][0]['launches'][0]['timeout_seconds'] = 3600
    else:
        plan['source_revision' if mutation == 'revision' else 'manifest_sha256'] = '0' * 40
    path = tmp_path / 'plan.json'
    path.write_text(json.dumps(plan))
    def unexpected_auth(*args, **kwargs):
        raise AssertionError('Invalid allocation must never enter native authentication/execution')
    monkeypatch.setattr(runner, 'authenticate_install', unexpected_auth)
    assert runner.main(['run', '--plan-file', str(path), '--group', plan['groups'][0]['group'],
                        '--wheel-directory', str(tmp_path / 'wheel'), '--install-root', str(tmp_path / 'prefix'),
                        '--receipt-dir', str(tmp_path / 'receipts')]) == 1


def test_plan_cli_exports_the_same_complete_dynamic_matrix(tmp_path):
    path, outputs = tmp_path / 'plan.json', tmp_path / 'outputs'
    assert runner.main(['plan', '--plan-file', str(path), '--github-output', str(outputs)]) == 0
    plan = json.loads(path.read_text())
    values = dict(line.split('=', 1) for line in outputs.read_text().splitlines())
    matrix = json.loads(values['matrix'])
    assert matrix['include'] == [{'group': row['group']} for row in plan['groups']]
    assert values['group_count'] == '114'


def wheel_artifact(tmp_path):
    directory = tmp_path / 'wheel'
    directory.mkdir()
    wheel = directory / 'pops-policy-test.whl'
    wheel.write_bytes(b'source-only artifact policy, never a native binary')
    receipt = {'dimension': 2, 'backend': 'mpi', 'wheel_sha256': hashlib.sha256(wheel.read_bytes()).hexdigest(),
               'native_sha256': 'native-policy-token', 'SDK': 'sdk-policy-token'}
    (directory / 'assembly-receipt.json').write_text(json.dumps(receipt))
    return directory


@pytest.mark.parametrize('failed_stage', ('pip', 'installed-wheel-proof', 'installed-native-proof', 'installed-sdk-proof'))
def test_authentication_stops_on_each_failed_stage_without_running_a_group(tmp_path, monkeypatch, failed_stage):
    wheel = wheel_artifact(tmp_path)
    calls = []
    names = iter(('venv', 'pip', 'installed-wheel-proof', 'installed-native-proof', 'installed-sdk-proof'))
    def process(command, **kwargs):
        name = next(names)
        calls.append((name, command, kwargs))
        return type('Result', (), {'returncode': 7 if name == failed_stage else 0})()
    monkeypatch.setattr(runner.subprocess, 'run', process)
    with pytest.raises(RuntimeError, match=failed_stage):
        runner.authenticate_install(wheel, tmp_path / 'prefix', tmp_path / 'receipts')
    assert calls[-1][0] == failed_stage
    assert all(kwargs['stdin'] is runner.subprocess.DEVNULL for _, _, kwargs in calls)
    proof = next((command for name, command, _ in calls if name == 'installed-wheel-proof'), None)
    if proof:
        assert '--wheel' in proof and '--expect-dim' in proof
    native = next((command for name, command, _ in calls if name == 'installed-native-proof'), None)
    if native:
        assert '--expect-mpi' in native and '--expect-parallel-hdf5' in native


def test_existing_installation_rechecks_every_wheel_native_and_sdk_guard(tmp_path, monkeypatch):
    wheel = wheel_artifact(tmp_path)
    prefix = tmp_path / 'prefix'
    prefix.mkdir()
    calls = []
    def process(command, **kwargs):
        calls.append(command)
        if any('verify_installed_native.py' in item for item in command):
            kwargs['stdout'].write(json.dumps({'sha256': 'native-policy-token',
                                               'has_mpi': True, 'has_parallel_hdf5': True}))
        return type('Result', (), {'returncode': 0})()
    monkeypatch.setattr(runner.subprocess, 'run', process)
    python, environment = runner.authenticate_install(wheel, prefix, tmp_path / 'receipts', existing=True)
    assert python == prefix / 'bin/python'
    assert len(calls) == 3
    assert 'prove_installed_wheel.py' in calls[0][2]
    assert 'verify_installed_native.py' in calls[1][2]
    assert calls[2][-1] == 'sdk-policy-token'
    assert environment['POPS_REQUIRE_MPI_TESTS'] == environment['POPS_REQUIRE_NATIVE_TESTS'] == '1'
    assert 'PYTHONPATH' not in environment and 'POPS_INCLUDE' not in environment


def test_checkout_artifact_duplicate_wheel_and_altered_bytes_are_refused_before_pip(tmp_path):
    wheel = wheel_artifact(tmp_path)
    with pytest.raises(ValueError, match='outside checkout'):
        runner.authenticate_install(wheel, runner.ROOT / 'bad-installed-prefix', tmp_path / 'receipts')
    (wheel / 'pops-duplicate.whl').write_bytes(b'extra')
    with pytest.raises(ValueError, match='one exact retained wheel'):
        runner.authenticate_install(wheel, tmp_path / 'prefix', tmp_path / 'receipts')
    (wheel / 'pops-duplicate.whl').unlink()
    (wheel / 'pops-policy-test.whl').write_bytes(b'changed')
    with pytest.raises(ValueError, match='differ from the assembly receipt'):
        runner.authenticate_install(wheel, tmp_path / 'prefix', tmp_path / 'receipts')


@pytest.mark.parametrize('failure', ('missing', 'duplicate', 'skip'))
def test_parent_receipt_does_not_accept_missing_duplicate_or_skipped_world_driver(tmp_path, failure):
    root = ET.Element('testsuite')
    ET.SubElement(root, 'testcase', name='first')
    ET.SubElement(root, 'testcase', name='second')
    receipt = tmp_path / 'parent.xml'
    ET.ElementTree(root).write(receipt)
    runner.require_parent_receipt(receipt, 2)
    if failure == 'missing':
        root.remove(root[-1])
    elif failure == 'duplicate':
        root[-1].attrib['name'] = root[0].attrib['name']
    else:
        ET.SubElement(root[0], 'skipped')
    ET.ElementTree(root).write(receipt)
    with pytest.raises(RuntimeError, match='every mandatory parent case'):
        runner.require_parent_receipt(receipt, 2)


def test_required_workflow_produces_one_wheel_and_all_matrix_groups_with_same_boolean():
    workflow = (runner.ROOT / '.github/workflows/ci.yml').read_text()
    mpi = workflow.split('\n  mpi:\n', 1)[1].split('\n  mpi-python:\n', 1)[0]
    worker = workflow.split('\n  mpi-python:\n', 1)[1].split('\n  # PREWARM OPENMP NATIF.', 1)[0]
    gate = workflow.split('\n  gate:\n', 1)[1].split('\n  mpi:\n', 1)[0]
    assert 'scripts/ci_python_mpi.py plan' in mpi
    assert '--dimension 2 --backend mpi' in mpi
    assert 'wheel_directory="$RUNNER_TEMP/pops-mpi-python-wheel"' in mpi
    assert 'name: gate-mpi-python-wheel' in mpi
    assert 'scripts/ci_python_mpi.py authenticate' in mpi
    assert 'pops-mpi-producer-installed/bin/python" scripts/run_m4_gate.py' in mpi
    assert 'Test Python MPI contracts + native lifecycle' not in mpi
    assert 'needs: [set-mode, mpi]' in worker and 'fail-fast: false' in worker
    assert "if: needs.set-mode.outputs.mpi_required == 'true'" in worker
    assert 'fromJSON(needs.mpi.outputs.python_mpi_matrix)' in worker
    assert 'scripts/ci_python_mpi.py authenticate' in worker and 'scripts/ci_python_mpi.py run' in worker
    assert 'timeout-minutes: 35' in worker and 'timeout-minutes: 8' in worker
    assert 'mpi-dim1, mpi-python, kokkos-openmp' in gate
    assert '--gate mpi-python "${{ needs.mpi-python.result }}"' in gate
    assert '${{ needs.set-mode.outputs.mpi_required }}' in gate
