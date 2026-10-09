"""Local shell snippets only; no operational R5, Kokkos, CUDA, Native or MPI execution."""
import json
import os
from pathlib import Path
import subprocess

out = Path(__file__).resolve().parent
base = Path('/Users/romaindespoulain/dev/tmp/PoPS-root-gh200-um-runtime-preparation-98804c68-20261009/root-reviewed-resume-r4')
context = (out / 'runtime_context_r5.sh').read_text()
exports = '\n'.join(line for line in context.splitlines()
                    if line.startswith(('export Kokkos_ROOT=', 'export POPS_KOKKOS_ROOT=')))
assert exports.splitlines() == ['export Kokkos_ROOT="$ROOT/kokkos-unified-install"',
                               'export POPS_KOKKOS_ROOT="$Kokkos_ROOT"']
old_guard = next(line for line in (base / 'build_native_r4.sh').read_text().splitlines()
                 if line.startswith('[[ "$Kokkos_ROOT"'))
new_guard = next(line for line in (out / 'build_native_r5.sh').read_text().splitlines()
                 if line.startswith('[[ "$Kokkos_ROOT"'))
assert old_guard == new_guard
env = dict(os.environ, ROOT='/LOCAL_SYNTHETIC_OWN_NAMESPACE')
env.pop('Kokkos_ROOT', None)
env.pop('POPS_KOKKOS_ROOT', None)
rows = []


def check(name, snippet, expected, environment=None):
    result = subprocess.run(['bash', '-c', 'set -euo pipefail\n' + snippet],
                            env=env if environment is None else environment,
                            capture_output=True, text=True)
    assert (result.returncode == 0) == (expected == 0)
    rows.append({'case': name, 'actual_local_exit': result.returncode,
                 'snippet': snippet, 'stdout': result.stdout, 'stderr': result.stderr})


check('R4 unset Kokkos_ROOT refuses at unchanged guard', old_guard, 1)
check('R5 exports then unchanged exact own-root guard', exports + '\n' + new_guard
      + '\n[[ "$POPS_KOKKOS_ROOT" == "$Kokkos_ROOT" ]]', 0)
check('R5 symbols exported to child environment', exports + '\n' + new_guard
      + "\nbash -c '[[ \"$Kokkos_ROOT\" == \"$ROOT/kokkos-unified-install\" && \"$POPS_KOKKOS_ROOT\" == \"$Kokkos_ROOT\" ]]'", 0)
check('Foreign Kokkos root cannot pass unchanged own-root guard', exports
      + '\nKokkos_ROOT=/LOCAL_SYNTHETIC_FOREIGN_NAMESPACE/kokkos\n' + new_guard, 1)
check('Inherited foreign discovery values are explicitly replaced by own-root exports', exports
      + '\n' + new_guard + '\n[[ "$POPS_KOKKOS_ROOT" == "$Kokkos_ROOT" ]]', 0,
      dict(env, Kokkos_ROOT='/LOCAL_SYNTHETIC_FOREIGN_NAMESPACE', POPS_KOKKOS_ROOT='/LOCAL_SYNTHETIC_FOREIGN_NAMESPACE'))
with (out / 'synthetic-kokkos-environment-proof.json').open('x') as stream:
    json.dump({'rows': rows, 'scope': __doc__, 'guard_unchanged': True}, stream, indent=2)
    stream.write('\n')
print(json.dumps({'synthetic_cases': len(rows), 'actual_R5_execution': False}))
