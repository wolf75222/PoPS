from pathlib import Path
import hashlib
import json
import shlex
import shutil
import subprocess
import time
import xml.etree.ElementTree as ET

root = Path('/Users/romaindespoulain/dev/tmp/PoPS-api040-integrated-main-20261004')
packet = Path('/Users/romaindespoulain/dev/tmp/PoPS-cpp-private-kernel-nonregression-8c1f-20261009')
out = packet / 'remaining-world1-e27d'
build = packet / 'build'
source = '8c1f55701cab7c60b3519bd343e3b4c1f193570f'
metadata_source = 'e27d065a5eb7c086ced60b751a2468f729361ff0'
assert subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip() == metadata_source
subprocess.run(['git', '-C', str(root), 'diff', '--quiet', 'HEAD'], check=True)
changed = subprocess.check_output(['git', '-C', str(root), 'diff', '--name-only', source, metadata_source], text=True).splitlines()
assert set(changed) == {'.github/workflows/ci.yml', 'scripts/ci_pytest_timings.py', 'tests/python/architecture/test_ci_impacted_selection.py', 'tests/python/support/stage_fields_public_fixture.py', 'tests/python/test_durations.json'}
previous = json.loads((packet / 'phase-result.json').read_text())
assert previous['Source'] == source and previous['exit'] == 0
out.mkdir(mode=0o700)
names = ['test_amr_tensor_fac_provider', 'test_amr_program_field_publication', 'test_field_nullspace']
pins = [{'path': str(build / 'bin' / name), 'bytes': (build / 'bin' / name).stat().st_size, 'sha256': hashlib.sha256((build / 'bin' / name).read_bytes()).hexdigest()} for name in names]
label = 'cpp-target:(' + '|'.join(names) + ')$'
command = ['ctest', '--test-dir', str(build), '--output-on-failure', '--no-tests=error', '-j', '1', '-L', label, '--output-junit', str(out / 'ctest.xml')]
prefix = '/Users/romaindespoulain/miniforge3/envs/pops'
script = 'set -euo pipefail\nset +u\nsource /Users/romaindespoulain/miniforge3/etc/profile.d/conda.sh\nconda activate ' + prefix + '\nset -u\nunset PYTHONPATH PYTHONOPTIMIZE\nexport FI_PROVIDER=tcp OMP_NUM_THREADS=2 OMP_PROC_BIND=false\n' + shlex.join(command) + '\n'
(out / 'actual-commands.sh').write_text(script)
invocation = {'compiled_Source': source, 'current_metadata_Source': metadata_source, 'unchanged_compiled_core_proof': changed, 'binaries': pins, 'argv': command, 'disk_before': shutil.disk_usage(out)._asdict(), 'scope': 'Actual repository CTest registration for the three remaining already-built C++ targets. No new build or relabelling of existing build Source.'}
(out / 'invocation.json').write_text(json.dumps(invocation, indent=2) + '\n')
start = time.monotonic()
with (out / 'phase.log').open('x') as stream:
    result = subprocess.run(['rtk', 'proxy', 'bash', str(out / 'actual-commands.sh')], cwd=root, stdout=stream, stderr=subprocess.STDOUT)
assert all(hashlib.sha256(Path(pin['path']).read_bytes()).hexdigest() == pin['sha256'] for pin in pins)
xml = ET.parse(out / 'ctest.xml').getroot() if (out / 'ctest.xml').is_file() else None
receipt = {'compiled_Source': source, 'current_metadata_Source': metadata_source, 'exit': result.returncode, 'seconds': time.monotonic() - start, 'counts': {key: int(xml.attrib.get(key, 0)) for key in ('tests', 'failures', 'errors', 'skipped')} if xml is not None else None, 'log_sha256': hashlib.sha256((out / 'phase.log').read_bytes()).hexdigest(), 'xml_sha256': hashlib.sha256((out / 'ctest.xml').read_bytes()).hexdigest() if xml is not None else None, 'binaries_unchanged': True, 'scope': 'Actual process and registered test closures. Separate from GPU, Python scientific suite and GitHub CI.'}
(out / 'phase-result.json').write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps(receipt), flush=True)
raise SystemExit(result.returncode)
