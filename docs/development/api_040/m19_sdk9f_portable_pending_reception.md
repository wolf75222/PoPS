# M19 SDK9f portable reception, 2026-10-01

This is a new reception of actual saved finite Reduce/Lift executions. It does
not reuse historical M19 seals and does not qualify Vlasov/BGK or a field solve.
The scientific guard remains `2e-12`. No PoPS module, native library, JIT, build,
or installed environment mutation is used by this offline work.

## Actual origins and pending authority

The private reader base is `dbc9e1cd`. Donors are below
`/Users/romaindespoulain/dev/tmp/pops-api040-native-reception-evidence-20261001`:

* `installed-sdk9f57-product-initial-diagnostic-nonreg-serial-dim2`: six passing
  M19 tests, observed source `b167556663a0736c22699be6976f17c81e26d05b`.
* `installed-sdk9f57-product-entropy-feedback-nonreg-mpi2-dim2`: thirteen passing
  tests per rank, observed source `cfd5b01ed6c409879a823e9f93a9f93b5f5198cb`.

Both use the actual Dim2 native SHA
`c8a45b013042f53a76ca6754a0285203aa5a9fda6a18f1d580e181dec99642fe`
and SDK `9f57def42c90cfc0e7ee127afa5050d95f27ecbc65b4ed198f788a097e4b4bad`.
The original native build source `0abbe25395a44e570d8d5525693b8e2dcbf4d387`
is a separate ROOT-attested slot, not inferred from current HEAD or a digest.

The portable directory is `m19-sdk9f-offline-pending-20261001` under the same
base. Its `origin-bytes.zip` is 28,515,200 bytes, with SHA
`fb0b0d13c767ec1b0ca1470621e27d4d35478656263ec1082725b9a41d15c053`.
The pending inventory SHA is
`676a171583a8739b99dc4e50c493b94ca22b88ea916c30ffbe70821423799ff1`.
It records 2,892 captures, 2,869 distinct original paths and 1,406 content blobs:
1,128 installed Python/header origins, all 1,128 matching Native-checkout source
origins, the actual native DSO and SDK manifest, 610 campaign files and 24
fixture/helper source references. Every source-map hash matched; no fallback
to another checkout or environment was necessary. Reads used a bounded length
from `fstat` on the open FD and required unchanged device/inode/size/timestamps.
Locks, pycache and symlinks are not evidence and were excluded.

The snapshot preserves all actual component DSOs and sidecars, three provider
CPPs per case, provider manifests/binaries, NPZ/checkpoints, identities, logs,
results and complete XMLs. There is no retained System-model CPP or original
resolved-plan payload to invent: `generated_cpp` and `cpp_dso_links` remain
null. The actual compiled-plan record and recomposed artifact payload retain
their existing, separately stated authority. The provider CPP invokes the
actual common `physical_support_transfer.hpp` lowering.

## New archived batch protocol @3

`sol61.m19-owner-pins@1/@2` keep their exact six-case JUnit admission.
`sol61.m19-owner-pins@3` is an explicitly archived protocol. It adds
`other_junit_cases` (exact classname/name pairs) and a closed list of
`supporting_evidence` leaves. Every ZIP member must be pinned, every pinned
logical origin is read only from that ZIP, and duplicate/symlink/escape/hash
or unpinned-member cases are refused. No live donor lookup occurs in reception.
Original absolute names preserve origin identity; the ZIP can be relocated by
an explicitly ROOT-approved backing-path update.

The complete XML must contain exactly the six M19 cases plus the sealed list
of other cases. Any failure, error or skip anywhere in that batch is refused;
suite counters and the six M19 rank/native/path properties remain strict.
The seven MPI extras are M18 entropy plus six public Integral feedback tests.
Their presence and passing status are batch evidence only, not independent
M19-reader scientific qualification of those mechanisms. Both MPI XMLs point
to the same collective `rank0-tmp` receipt directories; a second physical
archive or two nonempty partitions is not fabricated.

@3 additionally cross-checks receipt state/checkpoint digests against their
exact file pins. The existing physical states, nested checkpoint envelopes,
typed array hashes, artifact/bind/run/restart domains, mapping counts, clocks,
layout membership, ownership and bit-exact accepted/restored states are still
checked. The oracle uses the initial saved population, exact `Fraction(4,nv)`
weights, signed weights `(-1)^j*(j+1)` and constant extension; no runtime or
author numerical oracle supplies expected results.

Only ROOT may approve either pending manifest. The two external slots per mode
are (1) the SHA of `MODE-owner-pins.pending.json`, and (2) the external SHA of
an approval containing exactly:

```json
{"schema":"sol61.m19-root-approval@3","approved_by":"ROOT","pins_sha256":"EXACT_PENDING_MANIFEST_SHA"}
```

The pending Serial manifest SHA is
`8fe3cf1aa2befa14c72aaad4dad9379c0799dfcf20ff6689281e78d599199bf5`;
the MPI manifest SHA is
`112d0f585aecb2fdb717bf03baea03c8a7c2ee73e7dbe5e2926d2d6be68a6203`.
ROOT must verify the observed/build source distinction, exact roots and origins,
four actual System DSOs and three actual provider CPPs per case, full mixed XML
and closed backing before issuing approvals. These files currently remain
pending; neither a checksum nor a locally generated seal authorizes ROOT.

## Independent results and negative copies

The internal, unapproved scientific pass receives all six cases and four phases
per mode. The maximum original-equation error is `2.842170943040401e-14` in
both modes. Actual owner reports are distributed. This result is deliberately
labelled `unapproved_archive@3`; full reception requires `receive()` with both
external SHA values. No new native execution is claimed.

The negative harness starts from those real archives, changes only copies and
recomputes changed file pins, receipt hashes, nested CP array hashes and restart
identities. It checks uniform reduction, signed reduction, lift, population,
one-ULP restored state, clock, duplicate owner, mapping cadence, physical
provenance and a failure in an MPI extra test. Numerical changes to accepted,
restored and replayed states also update their matching physical CP arrays,
so the original-equation guard is reached rather than just a stale hash.
The one-ULP restart countermodel isolates exact-byte restart admission while
remaining below the scientific error tolerance.

These are `negative-test-only` envelopes, not authentic ROOT approvals. Their
refusal demonstrates mathematical/protocol discrimination *after* local
resealing; it does not prove that a malicious owner or changed ROOT seal is
authentic. The original portable ZIP/manifests and donor campaign hashes are
checked unchanged. The final replay refuses nine copies per mode plus one
extra-test XML failure in MPI (19/19 refusals). Restart refuses specifically
`restored state is not bit-identical to accepted`, independently of the
scientific tolerance. All 610 captured campaign files were re-read with stable
FD metadata and unchanged SHA after snapshot. Source/protocol tests receive
158 PASS in 0.98 seconds; Ruff and `git diff --check` pass. Historical protocols and native results remain intact.

## Commands

From the private checkout, source/protocol tests:

```sh
rtk proxy env -u PYTHONPATH /Users/romaindespoulain/miniforge3/envs/pops-api040-ir17/bin/python -m pytest -q tests/review/test_sol61_m19_sdk9f_batch_contract.py tests/review/test_sol61_m19_saved_reception.py tests/review/test_sol61_m19_archived_backing.py tests/review/test_sol61_m19_product_oracle.py
```

The real copied-countermodel command (use a fresh output directory):

```sh
rtk proxy env -u PYTHONPATH /Users/romaindespoulain/miniforge3/envs/pops-api040-ir17/bin/python tests/review/test_sol61_m19_sdk9f_batch_contract.py --pins /Users/romaindespoulain/dev/tmp/pops-api040-native-reception-evidence-20261001/m19-sdk9f-offline-pending-20261001/serial-owner-pins.pending.json --pins /Users/romaindespoulain/dev/tmp/pops-api040-native-reception-evidence-20261001/m19-sdk9f-offline-pending-20261001/mpi2-owner-pins.pending.json --output /Users/romaindespoulain/dev/tmp/pops-api040-native-reception-evidence-20261001/m19-sdk9f-offline-pending-20261001/real-countermodels-final
```

After ROOT approval, run `sol61_m19_saved_reception.py check --pins PATH
--pins-sha256 ROOT_SHA --approval PATH --approval-sha256 ROOT_SHA` separately
for Serial and MPI. The reader receives source/CPP/component/CP evidence within
its documented domains; no GPU, full kinetic PDE, new AMR or compile-graph
reconstruction is qualified by this offline receipt.
