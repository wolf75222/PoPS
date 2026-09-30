# Independent execution-owner metadata review

Reviewed source: `81249ef170bba6e8176b3ff54b42e58ae275e24d`, parent
`32235` as supplied by ROOT. This review changes tests/docs only. No native
execution, generated C++, scientific state, JIT, installation or build was
performed. Temporary `.so`/`.cpp` files contain explicitly opaque protocol
bytes; none are evidence of an actual native package or accepted state.

The independent capture probes execute the exact helper from that Git object,
not a rebaselined future helper. Their accepted counterexamples demonstrate
historical gaps, rather than requiring a corrected helper to keep accepting
them. The ordinary author tests and existing assembler tests exercise the
checkout sources. A future corrective implementation needs fresh rejection
tests on its new source, independent of this historical review.

## Received source contracts

- The source commit comes from the executing fixture's actual Git checkout.
  It is an observed HEAD, not an inferred binary build commit. Python entry,
  SDK manifest and selected native module paths/hashes are read directly;
  dual/target paths come from the executing artifact's model objects.
  Package/SDK/native hashes do not establish their build source. The earlier
  fixture provenance hashes and external ROOT approval remain necessary.
- `generated_sources` alone supplies C++ paths. No glob, filename guess,
  regeneration or installed-binary-hash lookup occurs. Absent compiler
  metadata stays `null` even when a plausible C++ file exists nearby.
  Retained Program C++ does not establish a complete System source map.
  The separate proposed `keep_model_cpp` driver port was not reviewed here.
- Rank-origin comparison precedes elected-C++ rehash, destination parity,
  agreed-record parity and publication. All three simulated ranks cross
  exactly eleven collective boundaries; rank0 opens the exclusive output
  after the first ten. Non-writing ranks still enter the final vote. These
  are source protocol observations, not MPI execution or empty-rank native
  qualification. Initial/confirmation divergence and changed elected C++
  refuse before opening the sidecar.
- The sidecar is outside the ten-phase directory, which keeps its thirty
  phase files plus provenance file. Only rank0 writes with mode `x`.
  The existing no-overwrite test remains received. The capture schema has
  no accepted result or ROOT approval field. Submitting that schema as its
  own external approval fails the exact approval contract.
- The assembler still checks its closed schema, explicit canonical roots,
  regular files, hashes, duplicate generated-path inventory, native evidence
  and System ABI7/hash evidence. A foreign explicit C++ path can be captured
  without guessing, but cannot pass the assembler's runtime-root boundary.

## Concrete P2 metadata gaps

**Symbolic origin erased before admission.** `_owner_file` first calls
`resolve(strict=True)`. An explicitly selected native-module or generated-C++
symlink to a regular file inside the permitted root is accepted and recorded
as its target. The assembler then sees a canonical regular path and accepts
the record: it cannot detect the erased alias. This does not prove wrong
bytes or a forged native result, but it fails the requested refusal of
symbolic origins. Preserve the raw origin long enough to refuse aliases;
rejecting only the already-resolved path is insufficient.

**Duplicate compiler inventory silently erased.** Two identical entries in
one `program.generated_sources` list collapse into one dictionary entry.
The capture and assembler accept that reduced record. The assembler does
reject an explicitly duplicated list when it is actually presented.
Therefore its downstream duplicate guard cannot authenticate the compiler's
original list. This counterexample uses duplication within one Program;
it makes no assertion about whether shared paths across separate layout
Programs should be prohibited. A correction should define that distinction
and validate each original inventory before any permitted union.

**Only elected C++ is rehashed before publication.** Changing the native or
dual-System file after initial capture, at the consensus seam, publishes
a sidecar containing its previous digest. Unlike C++, those leaves are not
reopened before the final vote/write. The strict assembler later rejects
the changed file. This is a failure of the requested pre-publication refusal,
not a demonstrated bypass of external approval or scientific reception.
Rehashing all elected origins before the final agreement would close this
specific seam; it would not make mutable files atomically immutable.

No P1 scientific/native acceptance bypass was demonstrated. These P2 gaps
must not be presented as already refused by the capture helper. They also
do not justify changing the physical oracle, M18 residual/population guards,
external owner pins, actual source/build identity or native outcome protocol.

## Reproducible source/host receipt

From the private checkout `PoPS-sol61-m18-owner-counterreview`:

```sh
rtk proxy env -u PYTHONPATH /Users/romaindespoulain/miniforge3/envs/pops-api040/bin/python -B -c 'import sys; sys.path.insert(0,"python"); import pytest; raise SystemExit(pytest.main(["-q","-p","no:cacheprovider","tests/python/unit/runtime/test_m18_execution_owner_capture.py","tests/review/test_sol61_m18_execution_owner_independent.py","tests/review/test_sol61_m18_owner_assemble.py"]))'
rtk proxy env -u PYTHONPATH /Users/romaindespoulain/miniforge3/envs/pops-api040/bin/python -m ruff check tests/review/test_sol61_m18_execution_owner_independent.py
rtk git diff --check
```

Actual result: **74 PASS in 1.53 s**, including fifteen independent protocol
checks, sixteen author capture checks and forty-three existing assembler
checks. Ruff PASS. The first independent run had fourteen PASS and one
test-only diagnostic mismatch (`approval` versus the actual
`ROOT has not approved this exact pending template`); the final expectation
uses that actual diagnostic. No refusal guard was relaxed.

Native M18 Serial/MPI2 is ROOT's pending obligation: execute the real
twenty-cell interior solve, immutable outside target's two refusals and
fresh safe-rebind with actual native/JUnit/package/source evidence; approve
actual owner pins externally, then receive the original independent entropy
oracle. This review alone qualifies none of those executions.
