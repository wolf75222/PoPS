# Pending native M19 finite-product reception

This receiver prepares future ROOT Serial/MPI2 campaigns. It performs no native
execution, imports no PoPS package, and creates no accepted-state NPZ or ROOT
approval. Its only mathematical scope is a finite product support, reduction over
velocity, and constant lift. It does not receive Vlasov, BGK, a transport solution,
or a Poisson field solve.

The genuine fixture is
`tests/python/integration/runtime/test_m19_product_support_runtime.py`. It selects
six cases `(nx,nv,width)=(4,3,3),(2,5,1),(7,3,5)`, each with both map declaration
orders. These dimensions bound this *reception inventory*, not the public API.
Each actual case directory must contain exactly 13 regular files: provenance and
four state/receipt/checkpoint triples for initial, accepted, restored, replayed.
There must be six successful JUnit cases per rank, one Serial JUnit or two distinct
MPI2 JUnits. Failed/error/skipped cases, duplicate cases/properties, changed native
identity, wrong source realm, rank or saved directory are refused. A failed native
campaign cannot become a complete reception by collecting only successful phases.

## Two external seals

`tests/review/sol61_m19_saved_reception.py assemble` hashes actual inputs and emits
only pending pins. ROOT must independently approve those exact bytes and supply
both SHA256 values outside the manifest. The checker does not create an approval.
An approval file has exactly:

```json
{"schema":"sol61.m19-root-approval@1","approved_by":"ROOT","pins_sha256":"ACTUAL_EXTERNAL_PENDING_SHA256"}
```

The string `ROOT` is not a cryptographic signature by itself. The independently
supplied approval SHA and pins SHA are the trust boundary. User-supplied self
approval, recomputed attacker hashes, or a manifest field claiming approval do
not supply that external authority.

The pending `sol61.m19-owner-pins@1` contains exactly `schema`, `mode`, `ranks`,
`archive_root`, `execution_origins`, `junit`, `cases`. Mode/ranks is exactly
serial/1 or mpi2/2. Each case key is the actual pytest parameter ID, such as
`False-4-3-3`, and contains dimensions, reverse boolean, actual directory, the
closed file pins and native SHA. `assemble` obtains reverse from genuine JUnit
parameter/directory associations, never a guessed artifact token. It refuses
aliases, duplicate directories and foreign dimensions. All file pins use exact
`{path,sha256}` leaves. Parent/file symlinks and `..` aliases are refused before
resolution. Case/JUnit files remain under the approved archive root.

Execution origins are supplied by ROOT from real execution/cache/SDK metadata,
not reconstructed by the mathematical receiver. The exact
`sol61.m19-execution-owner@1` object has:

- `schema`, observed `source_commit`, `native_build_source_commit` (actual 40-hex
  commit or null), and `roots` with canonical installation/runtime/source paths;
- actual `python_package`, `sdk`, `native` and `common_lowering` file leaves under
  the installation root;
- `sources`: the two actual fixture/helper file leaves under the source root,
  matching the provenance source path/hash dictionary exactly;
- `system_packages`: each of the six case IDs maps to four distinct actual System
  package file leaves under the runtime root; their hashes must equal the four
  distinct binary hashes recorded for the four compiled blocks;
- `provider_sources`: each case ID maps to three distinct actual retained provider
  source leaves under the runtime root. Their bytes must include the actual common
  `physical_support_transfer.hpp` include and `apply_physical_support_transfer`
  consumer call. This source-route check does not prove complete C++ semantics;
- `generated_cpp`: each case ID maps to actual distinct retained `.cpp` leaves,
  or null if those paths were not recorded/retained. Missing paths are not guessed
  from a DSO basename, hash lookup or sibling glob;
- `cpp_dso_links`: each case ID maps to null or actual owner-supplied links with
  exactly `cpp`, `dso`, `build_receipt`. The first two reference pinned retained
  CPP/System leaves; the receipt is an actual pinned runtime file. No build receipt
  schema/compile graph is invented by this receiver.

All origins are reread and their SHA checked. Observed fixture HEAD and native
build commit are distinct fields; one does not establish the other. Missing
Python/SDK/native/System/provider/common-source files are refusals, not inferred
identities. Null CPP/link/build-source fields are explicit gaps and still allow
the bounded `finite_product_saved_states_authsource_scope`. Even with supplied
build-receipt bytes, this version reports `strong_cpp_dso_link_qualified=false`:
it has not independently interpreted a complete compilation/link graph.

## Independent equations and actual envelopes

The checker reads the saved initial population, without importing or extracting
`input_and_expected`. It calls the independent no-PoPS Fraction reference
`tests/review/sol61_m19_product_oracle.py`:

- `R_uniform f[c,x] = sum_j (4/nv) f[c,x,j]`;
- `R_signed f[c,x] = sum_j (-1)^j (j+1) f[c,x,j]`;
- `L E[c,x,j] = E[c,x]` for every component and fibre.

The absolute fixture acceptance bound remains `2e-12`; it is not expanded by the
checker. Shape/dtype/finitude, every component, accepted/restored/replayed
original equations and bit-identical accepted/restored arrays are checked.
Source population is retained through zero-flux stages. Signed weights are
complete authored moment weights, not a positive norm measure. Lift is not an
inverse: `R L = (sum_j w_j) I`. Initial auxiliary sentinels are input data, not
substituted by the author oracle.

Physical map descriptors in the actual compiled plan must contain exactly the
three independent support maps: velocity/position domains, retained position
support, native axes, dimensionless weights, velocity interval `[-2,2]`, uniform
and signed quadratures, constant extension and no inverse closure. Map counters
have three exact IDs and equal 0/1/1/2 across the four phases. Clocks are exactly
0/.01/.01/.02 and macrosteps 0/1/1/2. Native-axis box ownership must cover each
source/destination cell once collectively or be genuinely fully replicated;
rank-local overlaps, partial duplicates, missing cells and escaped boxes fail.
Empty ranks are permitted. The ownership evidence is global snapshots plus
reported boxes; it does not independently inspect each process's local arrays.

Native provenance uses the genuine `metadata_encoding=json-with-bytes-hex.v1`
codec introduced by ROOT. Bytes have exactly `{bytes_hex: lowercase_even_hex}`;
no `str()` fallback or alternate tag is accepted. The receiver decodes these
bytes and independently recomputes the artifact v1 aggregate identity from the
actual retained CompiledPlanRecord's `resolved_plan_identity`, target, platform,
and component payloads. The retained `compiled_plan` is a CompiledPlanRecord,
not the original ResolvedSimulationPlan payload. Its body is ROOT-owner-attested;
`original_resolved_plan_payload_scope=not_stored` remains explicit. Rehashing that
record as a ResolvedSimulationPlan would be a false upcast. Artifact aggregate
authentication is distinct from original plan-body authentication and a compiled
source/DSO link.

Each top checkpoint is a real `multi_layout_uniform` NPZ containing three
nested uniform-layout NPZs. The reader bounds compressed archives, forbids pickle
and duplicate/foreign ZIP paths, verifies strict JSON, exact envelope fields,
array typed digests, restart CBOR identities, ABI/artifact/bind/clock links and
initial-vs-run provenance. It verifies all four child blocks, component names,
exact saved state bytes and original three-layout topology (integral and weighted
share one child). The outer mapping report must equal the phase receipt. It does
not execute native restart or prove Newton/Kokkos/MPI algorithms from arrays.
The original fixture's real restart execution remains separately required.

## ROOT commands after successful real campaigns

Supply six `--case-directory` arguments and one/two `--junit` arguments, in rank
order. ROOT execution origins must already contain actual canonical paths/hashes.
No helper command manufactures this execution metadata from an oracle.

```sh
python tests/review/sol61_m19_saved_reception.py assemble \
  --archive-root ACTUAL_ARCHIVE --mode serial \
  --case-directory ACTUAL_CASE_1 --case-directory ACTUAL_CASE_2 \
  --case-directory ACTUAL_CASE_3 --case-directory ACTUAL_CASE_4 \
  --case-directory ACTUAL_CASE_5 --case-directory ACTUAL_CASE_6 \
  --junit ACTUAL_RANK0_JUNIT --execution-origins ACTUAL_OWNER_ORIGINS \
  --output pending-m19.json

python tests/review/sol61_m19_saved_reception.py check \
  --pins pending-m19.json --pins-sha256 EXTERNAL_ROOT_PINS_SHA256 \
  --approval ACTUAL_ROOT_APPROVAL --approval-sha256 EXTERNAL_ROOT_APPROVAL_SHA256
```

Protocol/math tests use labelled synthetic arrays, opaque inventory files, and
in-memory envelope archives. No synthetic file is claimed to be a native result
and no positive saved-state NPZ is created. Attacks cover fully redigested wrong
signed moments/normalization/components/uniform measure/lift, missing/overlapping
owners, closed inventory/escape/aliases, rank/JUnit realm, byte tags, original
checkpoint members/clock/restart digest, origin drift and aggregate body mutation.

```sh
env -u PYTHONPATH /Users/romaindespoulain/miniforge3/envs/pops-api040/bin/python \
  -m pytest -q tests/review/test_sol61_m19_saved_reception.py \
  tests/review/test_sol61_m19_product_oracle.py
```

Native M19 reception remains pending. Source/math/protocol tests do not replace
six genuine Serial cases, six cases on each MPI2 rank, authentic saved states,
ROOT approval, native/GPU coverage, or a scientific Vlasov/BGK qualification.

Local reception: **120 source/math/protocol PASS** (62 new controls plus 58
existing independent oracle tests), Ruff and diff checks. Read-only inspection of
four actual *partial* ROOT provenance records from the still-failing/pending
`installed-m19-product-serial-dim2-sdk375f-codec-20261001` campaign confirms exact
artifact aggregate recomposition and physical-map descriptors. It receives no
phase state, no checkpoint or successful native case; no positive native receipt
is emitted from these partial records.
