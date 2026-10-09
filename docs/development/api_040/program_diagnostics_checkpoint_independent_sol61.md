# Independent reception of Program diagnostic checkpoint @1

Reviewed production/source freeze: `e86d07cc77c9e4b43868dae46feb1cbe5bc364b3`,
parent `01cc51b6127d20ab37c551ca14f4e38611b95add`. Review checkout:
`/Users/romaindespoulain/dev/tmp/pops-sol61-diagnostics-independent`.
This commit adds only this document and two independent probes. It changes no
production, installation, headers, runtime artefacts or owner evidence.

## Received source and host evidence

The standalone probe includes the real `program_diagnostics_checkpoint.hpp` and
extracts the exact frozen `System::restore_checkpoint_program_diagnostics` and
`AmrSystem::restore_checkpoint_program_diagnostics` definitions. It runs in
binary32/binary64, both with and without `NDEBUG`: **132 active checks per run,
528 total**. Lane, fence, table owner and collective failure propagation are
explicit host substitutes, not a Native System or MPI execution. Checks use an
exception-based assertion helper that remains active with `NDEBUG`.

`POPSDIA1` has an exact 40-byte header containing width, rank, rank count and
entry count. The tested tables preserve signed zero, a NaN payload, empty,
embedded-NUL and non-UTF8 names, and distinct rank ownership. Every proper
truncation of the real writer's 96-byte image, count/name overflow, foreign
rank/width, trailing byte, duplicate key and reserved balance prefix refuses.
The reader replaces no live table. The real restore bodies leave its original
bitwise table intact on producer allocation failure, malformed image, fence
failure and a simulated peer preparation failure; swap happens only after the
vote. Empty legacy input clears the table rather than retaining stale values.

The Python probes use the real public `RuntimeInstance` capacity method, real
`consensus`, exact `CheckpointResourceBudget`, and real joined budget builder.
Only transport/native accepted-boundary execution is replaced by two host
participants. One invalid rank (bool, zero, negative or overflowing integer)
votes before publication; the invalid rank never calls its Native accessor.
The other participant still reaches the preparation vote. The existing
consensus intentionally surfaces an overflow as collective `RuntimeError`
with the exact local `OverflowError` recorded. Distinct proposals, one pending
accepted-boundary refusal, consumer publication and changed cached authority
preserve both old budgets. A second-child refusal preserves every MultiLayout
child; positive reconfiguration uses the real aggregate budget and reserves
`N*ranks + 8*(ranks+1)` additional uncompressed bytes per child.

Independent image construction tests exact offset dtype/order/cardinality,
rank authority, incomplete key pairs and per-rank capacity before Native
validation. Rank-local payload bits need not match. Source AST checks ensure
exactly one diagnostic restore after accepted exchange/history replay for
Uniform and AMR. Existing accepted snapshots copy the diagnostic map, then
publish it by the existing prepared accepted-restore swap. This is source
evidence of the outer rollback join; the complete Native restart rollback is
still a Root reception obligation.

Real `CompiledSimulationArtifact` construction and verification receive local
retained C++ evidence. Changed/removed source invalidates the frozen record;
different source residence text yields distinct local evidence while component
evidence and canonical artifact identity remain identical. This evidence is
outside bind/content identity. It authenticates retained local text, and does
not independently prove the compiler's C++ to DSO execution association.

## Concrete resource-order defect in e86 (preserved historical probes)

The two historical probes load the helper directly from `git show e86:...`.
Their passing result records the defect's presence; it is **not** reception of
a corrected resource-order guard.

1. With chosen `capacity_per_rank=40` and a valid 66-byte rank image, both
   participants pass the preparation vote and enter byte allgather. Only the
   subsequent serialization vote refuses capacity. No payload is published,
   but remote images have already been transported/materialized beyond N.
2. Restart preflight converts both full rank slices to `bytes` before testing
   the per-rank capacity. Both 66-byte copies are observed before the refusal.

The existing NPZ reader still authenticates/bounds the enclosing archive with
the live aggregate resource budget. These counterexamples establish ordering
of specific additional allocations, not an unbounded archive or a scientific
acceptance bypass. The author and Root received the issue and assigned a
separate correction: check local size at the pre-transport vote, and inspect
headers/offsets without copying full oversized slices before the specific
capacity guard. That follow-up must be received independently.

Native serialization still constructs a vector before Python can inspect its
length. This contract is an archive/transport capacity choice, **not a cap on
RSS or all producer allocations**. Enforcing N inside that allocation would
require a separate capacity-bearing Native writer interface. No such Native
signature change is claimed or performed here.

## Commands and limits

From this private checkout:

```sh
rtk proxy env -u PYTHONPATH -u POPS_NATIVE_DIM PYTHONPATH=python PYTHONDONTWRITEBYTECODE=1 /Users/romaindespoulain/miniforge3/envs/pops-api040/bin/python -B -m pytest -q --tb=short tests/review/test_sol61_diagnostics_independent.py
rtk proxy env -u PYTHONPATH -u POPS_NATIVE_DIM PYTHONPATH=python PYTHONDONTWRITEBYTECODE=1 /Users/romaindespoulain/miniforge3/envs/pops-api040/bin/python -B -m ruff check tests/review/test_sol61_diagnostics_independent.py
rtk git diff --check
```

Reception: **26 source/host cases PASS**. Four cases compile only the small
real codec header plus extracted methods with system `c++ -std=c++20 -O0`.
No native package, MPI, Kokkos runtime, AMR hierarchy, NPZ filesystem restart,
shared environment change, heavy translation unit build or JIT was executed.
Actual Native one-rank failure/fence, pending outcome, multi-child rollback,
AMR accepted geometry/flux state and source/DSO reception remain with Root.

## Separate reception of capacity-order fix 521cca58

Authored fix `521cca58a9719dc0843065ac0af0bfaf9aa242fe`, parent Root
`0298d696`, was cherry-picked as `ee7978cf185986b16491e327109bcc38b3fc3a7a`
onto the independent historical receipt `6cb791941b580d82c0fa0d5d7b2ca0952aea2f0f`.
The tested helper is byte-exact to the authored 521 blob. This reception adds
only tests/documentation; the authored fix changes the Python helper, its
author test and its author document, with no Native API/header changes.

**33 source/host cases PASS in 5.94s**, Ruff PASS. This comprises the original
26 checks and seven new checks using the same independent image construction,
real consensus and two-host-participant transport adapters:

- A valid 66-byte image against chosen N=40 on one or both participants is
  rejected at the first preparation vote. Neither participant reaches byte
  transport; neither publishes payload keys. Every participant still votes.
- Positive bounded capture uses the real helper and real consensus in two
  phases and transports exactly the two distinct rank images. Both archive
  images contain exact uint8 bytes and int64 offsets.
- Per-rank or aggregate excess refuses before any `_exact_native_image` call,
  hence before even the fixed header conversion. The Native validator is not
  invoked for the rejected owner image.
- Standalone admissible array validation copies exactly 40 bytes per rank.
  Owner preflight checks the chosen capacities first, then copies exactly the
  selected bounded rank body for Native validation. Opaque/NUL/non-UTF8 names
  and NaN bits remain unchanged in that selection.

The two historical e86 proofs still execute the old helper from its pinned
Git source; they continue to establish the original late guards. They are
not rebased to new behavior or used as evidence of the corrected positive.
The real codec and extracted restore checks remain unchanged (528 active
host assertions). No Native/MPI, Kokkos or full physical restart execution is
attributed to this Python fix reception. The capacity is an archive/transport
bound; Native writer allocation before the Python size check is still outside
that bound and is not claimed to be limited by N. No new defect is demonstrated
within the corrected scope.
