# Accepted diagnostics: installed Native fixture, pending reception

This test-only follow-up starts at Native source `0298d696ab5400c36d8f937f4ae3be87c2554e2d`
and contains the separate Python allocation-order correction `521cca58a9719dc0843065ac0af0bfaf9aa242fe`.
It changes neither production headers nor physical descriptors. The production codec/capacity
contracts remain `pops.program-diagnostics.checkpoint-archive@1` and
`pops.program-diagnostics.checkpoint-capacity@1`.

The actual public Case declares a two-component source balance, a stationary zero flux and
periodic Cartesian2D geometry. A Program applies `q0_next=(1-dt)*q0` and
`q1_next=(1+dt/2)*q1`, stores a depth-two accepted history, and records a genuine collective
`dot_all(q,q)` under `global-energy`. The AMR realization uses a two-level synchronous hierarchy,
an explicit StateTransfer and partial refinement from its actual initial profile. Its coarse
active mask must contain both active and covered cells. The fixture has no nonlinear solve or
new physical qualification: it isolates durable diagnostics and the joined restart transaction.

`tests/python/integration/runtime/test_program_diagnostic_checkpoint_runtime.py` contains two
unmarked source admission cases and four marked Native cases (Uniform/AMR times two families).
`tests/python/support/program_diagnostic_native_receipts.py` shares their real Case, bind,
observations and receipt construction. Native cases select the actual Dim2 module before
resolution, use the genuine typed execution context, and compile through
`compile_resolved_plan_once` when MPI is present. All checkpoint/fault paths descend from the
rank-zero broadcast `collective_directory`. Native calls and local assertions have separate
collective boundaries; no rank, DSO or native field operation is replaced.

The first family checks the initialized states, an initial accepted checkpoint/restart before
any run, an accepted step, an exact-byte-capacity checkpoint, fresh bind/restart with a
deliberately stale destination key, and a second accepted
step compared byte-for-byte with continuous execution. States, history arrays, initialization
flags, fill counts, duration bits, sample identities, clock/step and the complete native
diagnostic image are retained. The raw sink is the existing actual native
`record_program_diagnostic` API on the bound executor. It supplies a rank-local value, an empty
name, embedded NUL/Unicode, negative zero, a subnormal, infinity and a quiet-NaN payload.
The independent image reader compares float64 IEEE bits instead of NaN float equality. Invalid
UTF-8 names cannot be authored through this Python string API; their codec coverage remains in
the separate extracted C++ tests. The global sink's bits must agree across actual MPI ranks;
the local sink's bits deliberately differ.

The second family exercises the following on those same real runtimes:

- reserved raw namespace refusal; invalid capacity on rank zero, arithmetic overflow, and
  different valid rank proposals when MPI size exceeds one;
- configuration during a real native step transaction, followed by actual rollback;
- successful exact encoded-byte boundary versus one-byte-too-small capture, with no checkpoint
  publication or accepted-state changes on refusal;
- fully harness-resealed malformed offsets, duplicate rank authority, a reordered rank table
  when MPI has several ranks, and an impossible native entry count whose valid fixed header
  lets it reach the real native reader;
- a private test fault at the existing executor apply seam **after** its genuine native
  state/history/map restore returns, before the real apply vote. Only rank zero raises.
  The joined transaction must compensate all accepted images on every rank;
- harness-resealed legacy absence of both diagnostic arrays. It must clear the whole stale map,
  preserve the physical state/history/clock, and resume to the same physical second-step result.
  Raw keys must remain absent until explicitly recorded again.

These mutations are test inputs, never externally authenticated native positives. The harness
uses the real runtime envelope/consumer identities solely to reseal its owned negative copies;
it does not create an owner seal, modify a donor, infer a resource budget from NPZ bytes, or relax
a physical guard. Capacity covers archive/transport bytes, not peak Native map/encoding RAM.
Missing or dynamically named retained source requires explicit capacity by the production
contract; this fixture chooses capacity explicitly for its legitimate raw records and tests the
too-small alternative rather than fabricating missing compilation evidence.

Each executed phase writes an actual rank-owned NPZ and JSON sidecar with its hashes, lifecycle,
history metadata, diagnostic header/bits, Native dimension/path/DSO hash/ABI/capabilities,
artifact/bind/platform/component/retained-plan evidence, and actual retained Program C++ if
present. The copied local C++ has its own SHA/path; this evidence alone does not prove its
compilation into the DSO. The helper and test source hashes are included. ROOT must independently
pin the installed source manifest, native identity, JUnit results and every rank's sidecars to
receive a positive result. Shared storage and actual MPI execution remain ROOT responsibilities.

## Checks performed here

On 2026-10-01, the existing `pops-api040/bin/python` with explicit `PYTHONPATH=python` admitted both
real Cases through validate/resolve and emitted their actual Uniform/AMR Program C++. The first
draft used incorrect emitter import names and attempted to subscript a StateHandle; source
checks exposed both and the fixture now uses the current codegen modules and ValueExpr component
projection. Final source admission: **2 PASS, 4 Native cases deselected**, with Ruff and diff
checks. No native case, compilation, bind, MPI rank, checkpoint, JIT, setup or installed-environment
mutation was executed here. These are executable future fixtures, not saved-state evidence.

ROOT's installed Serial and MPI2 runners can select this file after the coherent rebuild:

```text
env -u PYTHONPATH <installed-python> -m pytest \
  tests/python/integration/runtime/test_program_diagnostic_checkpoint_runtime.py \
  -m native_loader -q --junitxml=<run-directory>/pytest.xml \
  --basetemp=<run-directory>/pytest-tmp
```

For MPI, use the established ROOT per-rank pytest runner and its separate rank JUnit files; do
not let two ranks write one XML. Preserve receipts even when a test is red. A source admission
pass is not Native Kokkos/MPI qualification, and this fixture does not receive AMR solver
convergence, GPU execution, rank-count-changing restart, or a complete scientific corpus model.
