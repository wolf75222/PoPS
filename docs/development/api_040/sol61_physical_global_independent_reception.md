# Independent physical Global source reception

This review starts from `afcff38407cd32e05752cace9947ba66166a1f4e`
(parent `8eabb8e2`). Production fixes received separately are
`97248f5e09be654ab262c96838608b30c5b1509b` and
`26acc01d6de41dc64f98fc22c9ab945ae4f1d2d5`. This review changes only tests
and this report. Source packages are exact `git archive` copies; no installed
package, prototype solver, JIT, native execution, or shared environment mutation
is used as evidence.

## Three counterexamples and their reception

1. A genuine source SSA was reissued using `_replace_value`, with its issued
   Global port replaced by a distinct same-owner clone carrying equal sealed
   metadata. Initial emission accepted it. The unchanged counter-test now refuses
   with `registry-issued`, before the source kernel can be emitted. The original
   authoring binder already checked this; the missing check was in the codegen
   binder. Fix: `97248f5`.
2. Manifest v11 accepted the Global declaration version `1.0`. The fixed decoder
   requires an exact integer and rejects this unchanged counter-test. Fix:
   `97248f5`.
3. Manifest v11 accepted the units image `[["charge", true, 1]]`: Python's numeric
   equality concealed the noncanonical exponent. The decoder now checks the list
   image and exact integer numerator/denominator before dimensional decoding.
   The unchanged counter-test refuses on `26acc01`, after failing on `97248f5`.

These are typed declaration and binding refusals, not alterations to the
physical equation or normalization of forged input.

## Independent witnesses

`tests/review/test_sol61_physical_global.py` defines the physical model before
constructing a Program: two components named `z,a`, a declared dimensionless
Global quantity, and the original named source `S_i = -0.47 Q U_i`. The source is
selected from its physical Equation. A candidate integral capture is bound to
the issued block port. The Program applies the selected source without another
factor of Q. An independent scalar evaluation checks the original physical body
at U=(1.3,-2.1), Q=0.82, before and after model freezing.

The real Uniform and AMR code generators emit one capture and one authorized
consumer; the consumer occurs before the genuine Kokkos cell traversal. SSA
dependencies include the capture at the exact state point. Missing, unused,
foreign-block, same-owner cloned, foreign-Program, wrong-unit and wrong-point
bindings refuse without changing Program counters or values. Reissued stale
capture point/scope and binding version/owner refuse during emission. A genuine
change to gamma changes model, Program and capture identities and the emitted
body. These are source checks; counting emitted consumers does not independently
prove native exactly-once exterior accounting or rollback/retry.

The independent unsupported-consumer witnesses use real Module/operator-first
declarations. Binding refuses for the default native source provider, grid and
field operators, and both implicit source APIs before SSA publication. Direct
unbound Global expression emission also refuses; no fallback into a direct
Equation/FieldProblem/flux is qualified by this extension.

## Commands and results

Run the independent tests against the exact final source archive, rather than
the installed SDK:

```sh
env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 \
  /Users/romaindespoulain/miniforge3/envs/pops-api040/bin/python -c '
import sys
sys.path.insert(0, "<archive-26acc01>/python")
import pops, pytest
print(pops.__file__)
raise SystemExit(pytest.main([
    "-q", "tests/review/test_sol61_physical_global.py",
    "-p", "no:cacheprovider", "--disable-warnings"]))'
```

Result: **23 passed**, 8.29 seconds. Ruff passes for both independent Python
files. The affected author source/codegen/capture/import-graph suite is run
separately; it is not counted as independent scientific evidence.

The independent Uniform and AMR emitted C++ both pass real Clang Dim2 syntax
with Kokkos, MPI, OpenMP, parallel-HDF5 and the genuine ProgramContext headers.
They were emitted against `97248f5`; `26acc01` changes only the manifest units
decoder and its tests. No object was linked or executed. Reproduce with the
existing environment includes and `-fsyntax-only -std=c++20 -DPOPS_NATIVE_DIM=2`
plus the repository's native feature flags, preincluding respectively
`pops/runtime/program/program_context.hpp` and
`pops/runtime/program/amr_program_context.hpp`.

`tests/review/sol61_global_legacy.py` runs in separate processes against parent
and final source, covering schema10 scalar, two-component and reversed-component
Modules. All three Manifest JSON byte digests, module hashes and semantic
identities match exactly. Use the same helper file for both sides: operator
provenance includes the declaration's source location, so reformatting the
helper between runs changes manifest bytes independently of production code.

```sh
env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 <python> \
  tests/review/sol61_global_legacy.py <archive-parent> > parent.json
env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 <python> \
  tests/review/sol61_global_legacy.py <archive-26acc01> > final.json
cmp parent.json final.json
```

Manifest v11 roundtrip/ABI copying and semantic projection are exercised by the
affected codegen suite and inspected in `_module_manifest.py` and
`identity/semantic.py`; unextended modules retain schema10. Root owns the
integrated conditional ProgramIR8 route and release contract reconstruction.

Local working evidence is under
`outputs/sol61-global-source-independent-review-20260930/`: the three source
archives, emitted Uniform/AMR C++, and legacy JSON receipts. The final legacy
comparison uses `legacy-parent-final.json` and `legacy-final.json`; older JSON
files predate helper formatting and remain preserved.

## Limits

This is a bounded source, mathematical body, codegen and C++ syntax reception.
It does not receive native source evaluation, MPI/GPU execution, distributed
failure consensus, integral/exterior ledger accounting, restart/retry, or the
scientific M14 result. Those require the authentic rebuilt package and native
receipts. The existing typed integral capture's native authority is reused,
not replaced by a Python mock. Arbitrary mutation of private object internals
outside the exercised reissue seams is not claimed exhaustively guarded.
