# Physical global plan fixture portability - 30 September 2026

This change repairs tests only, on parent
`f7378be89e10de170b37f7158bb91031c754829b`. Production Python, headers, native
libraries and the shared environment are unchanged.

The previous four golden plan identities and complete Module manifest hashes
included authentic absolute authoring paths and callsite provenance. Moving the
fixture changes that provenance and therefore the snapshot artifact hash and
plan identity, even when IR and emitted C++ are identical. Replacing those
hashes with another checkout's hashes would recreate the same failure.

The four ordinary unit cases now construct one actual input and compare its
live and detached resolved plan: exact plan identity, complete canonical plan
payload, complete snapshot artifact JSON, Module hash and full manifest, and
both full System/AMR C++ strings. They retain the historical IR, manifest
version 10/11 and eight C++ digest pins. No provenance is stripped or rewritten.
These unit cases require neither Git history nor another checkout and contain
no skip or subprocess path.

The original `sol61_global_authority_parent_parity.json` remains byte-identical
(SHA256 `6e088b12ec19cbaa84c47f36edd84d90900320573d1343b973ee037cd626f6b0`).
Its location-dependent hashes remain historical evidence.

## Explicit historical comparison

`tests/review/check_sol61_global_plan_parent_parity.py` requires the local Git
object `368055dbe1f4c1f5fad4a11791508ae2b0520f03` and a named candidate package
root. It extracts only the parent's Python source into a temporary directory.
Two fresh interpreters execute the same fixture file and the same authoring
callsite, changing only the imported package root. Import identity is checked.
The entire resulting receipts must be equal, including full Module manifest,
canonical plan payload and snapshot provenance, rather than selected hashes.
Missing history is an explicit error in this review command; it never fetches,
installs, builds or substitutes an installed package.

For all four combinations of physical global presence and periodicity, changing
the actual source coefficient from 0.3 to 0.47 changes eight independent receipt
fields: IR, Module hash, complete Module manifest, plan identity, complete plan
payload, complete snapshot JSON, and both emitted C++ digests. All 32 guards
were observed. This negative control is source authoring, not a synthetic native
execution.

The fresh receipt is
`sol61_global_plan_fresh_parent_receipt_20260930.json`, SHA256
`7a27fdc3a90c80765b06e65e005c4f6ef1cc35ce6be63251314c71e50cfa2027`.
Its paths and provenance are genuine to this checkout; they are not portable
unit golden replacements.

## Validation

Using `/Users/romaindespoulain/miniforge3/envs/pops-api040/bin/python`:

```sh
PYTHONPATH=python python -m pytest -q \
  tests/python/unit/codegen/test_resolved_physical_global_authority.py \
  tests/python/unit/codegen/test_physical_global_primitive_authority.py
env -u PYTHONPATH python tests/review/check_sol61_global_plan_parent_parity.py \
  --repository . --candidate-package-root python \
  --receipt docs/development/api_040/sol61_global_plan_fresh_parent_receipt_20260930.json
```

The source unit selection passed **32 tests**. The explicit comparison received
four complete parent/candidate receipts and all 32 changed-source guards. Ruff
passed for the three changed/new Python files. These checks receive source
resolution, detachment and actual code emission; they do not claim installed
runtime, Kokkos, MPI, GPU or GitHub CI reception.
