#!/usr/bin/env python3
"""Plan or qualify exact manifest tests requiring real CUDA/HIP device storage.

CPU CI records this declared hardware exclusion. Only --execute on an authenticated
installed GPU runtime can produce an executed receipt; no hardware is emulated.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ci_select_tests

ROOT = Path(__file__).resolve().parents[1]
CHILD_TIMEOUT_SECONDS = 14 * 60


def require_gpu_facts(capabilities, facts, dimension):
    if (capabilities.get('supports_gpu') is not True
            or type(capabilities.get('dimension')) is not int
            or capabilities['dimension'] != dimension
            or capabilities.get('real_bytes') != 8
            or facts.get('has_kokkos') is not True
            or facts.get('kokkos_device') not in ('cuda', 'hip')
            or facts.get('field_memory_space') not in ('device', 'managed')):
        raise RuntimeError('GPU qualification requires genuine CUDA/HIP with device/managed Field storage')


def authenticate_gpu(dimension):
    import pops
    from pops._native_selector import select_native_dimension
    from pops.codegen.abi import module_header_signature
    from pops.codegen.toolchain import pops_header_signature, pops_include
    from verify_installed_native import verify_installed_native

    native = select_native_dimension(dimension)
    origin = verify_installed_native(expect_dimension=dimension)
    prefix = Path(sys.prefix).resolve()
    if not Path(pops.__file__).resolve().is_relative_to(prefix) or not origin.is_relative_to(prefix):
        raise RuntimeError('GPU qualification requires the genuine installed Python and native package')
    if module_header_signature() != pops_header_signature(pops_include()):
        raise RuntimeError('GPU qualification SDK and native header identities differ')
    capabilities = dict(native.module_capabilities('production'))
    facts = dict(native.runtime_environment_report())
    return {'package': str(Path(pops.__file__).resolve()), 'native': str(origin),
            'capabilities': capabilities, 'facts': facts,
            'header_signature': module_header_signature()}


def require_case_receipt(path, expected):
    if not path.is_file():
        raise RuntimeError('GPU execution omitted its case receipt')
    cases = list(ET.parse(path).getroot().iter('testcase'))
    received = Counter(case.attrib['name'].split('[', 1)[0] for case in cases)
    identities = {(case.attrib.get('classname'), case.attrib['name']) for case in cases}
    if received != expected or len(identities) != len(cases):
        raise RuntimeError('GPU execution case inventory differs from the exact manifest plan')
    if any(case.find(tag) is not None for case in cases for tag in ('failure', 'error', 'skipped')):
        raise RuntimeError('GPU execution did not receive every mandatory case')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--hardware', required=True, choices=('gpu',))
    parser.add_argument('--receipt-dir', required=True, type=Path)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args(argv)
    entries = [row for row in ci_select_tests.manifest_python_hardware_entrypoints(
        ci_select_tests.load_manifest()) if row['hardware'] == args.hardware]
    if not entries:
        parser.error('the hardware profile has no manifest-owned tests')
    receipt = {'schema': 'pops.hardware-qualification@1', 'hardware': args.hardware,
               'status': 'planned', 'entrypoints': entries,
               'case_count': sum(sum(row['expected_cases'].values()) for row in entries),
               'cpu_execution': False, 'child_timeout_seconds': CHILD_TIMEOUT_SECONDS}
    args.receipt_dir.mkdir(parents=True, exist_ok=False)
    receipt_file = args.receipt_dir / 'hardware-plan.json'
    receipt_file.write_text(json.dumps(receipt, indent=2) + '\n')
    if not args.execute:
        print(receipt_file)
        return 0
    try:
        for index, row in enumerate(entries):
            identity = authenticate_gpu(row['native_dimension'])
            (args.receipt_dir / f'native-{index}.json').write_text(json.dumps(identity, indent=2) + '\n')
            require_gpu_facts(identity['capabilities'], identity['facts'], row['native_dimension'])
            environment = os.environ.copy()
            environment.pop('PYTHONPATH', None)
            environment.update(POPS_NATIVE_DIM=str(row['native_dimension']),
                               POPS_REQUIRE_NATIVE_TESTS='1', PYTHONNOUSERSITE='1')
            xml = args.receipt_dir.resolve() / f'cases-{index}.xml'
            bootstrap = (
                'from pops._native_selector import select_native_dimension; '
                f'select_native_dimension({row["native_dimension"]}); '
                'import os,sys,pytest; sys.path.insert(0,os.getcwd()); '
                'raise SystemExit(pytest.main(sys.argv[1:]))'
            )
            command = [sys.executable, '-I', '-u', '-c', bootstrap,
                       '-q', '-ra', '--maxfail=1', '--junitxml=' + str(xml), row['path']]
            result = subprocess.run(command, cwd=ROOT, env=environment, capture_output=True,
                                    text=True, check=False, timeout=CHILD_TIMEOUT_SECONDS)
            (args.receipt_dir / f'launcher-{index}.log').write_text(result.stdout + result.stderr)
            if result.returncode:
                raise RuntimeError('GPU execution failed with status %d' % result.returncode)
            require_case_receipt(xml, row['expected_cases'])
        receipt['status'] = 'executed'
    except Exception as error:
        receipt.update(status='failed', reason=str(error))
        receipt_file.write_text(json.dumps(receipt, indent=2) + '\n')
        print('GPU qualification failed: ' + str(error), file=sys.stderr)
        return 1
    receipt_file.write_text(json.dumps(receipt, indent=2) + '\n')
    print(receipt_file)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
