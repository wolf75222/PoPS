from pathlib import Path
import argparse
import datetime
import hashlib
import json
import os
import subprocess
import time

parser = argparse.ArgumentParser()
parser.add_argument('--phase', required=True, choices=['guarded-variants', 'original-pde', 'retry-and-native8', 'public10', 'mpi2-representative', 'mpi2-all', 'coupled-matched-world1'])
args = parser.parse_args()
out = Path(__file__).resolve().parent
main = Path('/Users/romaindespoulain/dev/tmp/PoPS-api040-integrated-main-20261004')
prefix = Path('/Users/romaindespoulain/miniforge3/envs/pops')
admission = json.loads((out / 'native-build-admission.json').read_text())
source = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=main, text=True).strip()
assert source == admission['Source']
subprocess.run(['git', 'diff', '--quiet', 'HEAD'], cwd=main, check=True)
native = Path(admission['installed_native']['path'])
assert hashlib.sha256(native.read_bytes()).hexdigest() == admission['installed_native']['sha256']
old = main / 'docs/development/api_040/evidence/current_988_f8d_20261009'


def old_nodes(name):
    values = json.loads((old / name).read_text())['pytest_args']
    return [value for value in values if value.startswith('tests/python/')]


phase_nodes = {
    'coupled-matched-world1': ['tests/python/integration/runtime/test_field_publication_instances_runtime.py::test_installed_three_instance_solved_provider_reads[cells0-False-False]'],
    'guarded-variants': old_nodes('guarded-variants-invocation.json'),
    'original-pde': ['tests/python/integration/runtime/test_imex_nonautonomous_field.py::test_public_nonautonomous_imex_field_reads_explicit_time'],
    'retry-and-native8': old_nodes('original-retry-and-native8-invocation.json'),
    'public10': old_nodes('public-methods-rollback-capacity-invocation.json'),
    'mpi2-representative': ['tests/python/integration/runtime/test_imex_nonautonomous_field.py::test_public_nonautonomous_imex_field_reads_explicit_time', 'tests/python/integration/runtime/test_public_diffusion_field_predictor.py::test_public_diffusion_predictor_is_field_rhs[euler]'],
    'mpi2-all': ['tests/python/integration/runtime/test_public_diffusion_field_predictor.py', 'tests/python/integration/runtime/test_public_diffusion_field_late_refusal.py', 'tests/python/integration/runtime/test_guarded_field_diffusion_publication.py', 'tests/python/integration/runtime/test_field_publication_instances_runtime.py::test_installed_three_instance_solved_provider_reads[cells0-False-False]'],
}
destination = out / args.phase
assert not destination.exists(), 'Use a new phase output; retained runs must not be overwritten'
destination.mkdir(mode=0o700)
env = os.environ.copy()
for key in ['PYTHONPATH', 'PYTHONOPTIMIZE', 'PYTEST_ADDOPTS']:
    env.pop(key, None)
env.update(PYTHONDONTWRITEBYTECODE='1', FI_PROVIDER='tcp', OMP_NUM_THREADS='2', OMP_PROC_BIND='false',
           POPS_NATIVE_DIM='2', POPS_REQUIRE_NATIVE_TESTS='1', POPS_INCLUDE=str(main/'include'),
           Kokkos_ROOT=str(prefix), POPS_KOKKOS_ROOT=str(prefix),
           POPS_GUARDED_FIELD_SOURCE_SHA=source, POPS_GUARDED_FIELD_NATIVE_SHA=admission['installed_native']['sha256'])
mpi = args.phase.startswith('mpi2-')
driver = main / 'docs/development/api_040' / ('run_installed_mpi_checks.py' if mpi else 'run_installed_checks.py')
command = [str(prefix/'bin/python'), str(driver), '--output', str(destination/'run')]
if mpi:
    command += ['--ranks', '2', '--dimension', '2', '--threads', '2']
for node in phase_nodes[args.phase]:
    command += ['--test', node]
record = {'Source': source, 'Native': admission['installed_native'], 'SDK': admission['SDK'],
          'command': command, 'phase': args.phase, 'start_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
          'scope': 'Actual new installed cohort; retained Source988 commands supply test selection only, no receipt or result inheritance.'}
with (destination/'producer-invocation.json').open('x') as handle:
    json.dump(record, handle, indent=2); handle.write('\n')
started = time.monotonic()
with (destination/'producer.log').open('x') as handle:
    result = subprocess.run(command, cwd=main, env=env, stdout=handle, stderr=subprocess.STDOUT)
subprocess.run(['git', 'diff', '--quiet', 'HEAD'], cwd=main, check=True)
assert hashlib.sha256(native.read_bytes()).hexdigest() == admission['installed_native']['sha256']
record.update(exit=result.returncode, seconds=time.monotonic()-started,
              log_sha256=hashlib.sha256((destination/'producer.log').read_bytes()).hexdigest(),
              Source_after=subprocess.check_output(['git','rev-parse','HEAD'],cwd=main,text=True).strip())
with (destination/'producer-result.json').open('x') as handle:
    json.dump(record, handle, indent=2); handle.write('\n')
print(json.dumps({'phase': args.phase, 'exit': result.returncode, 'seconds': record['seconds'], 'output': str(destination)}))
raise SystemExit(result.returncode)
