# Accepted Program diagnostics checkpoint extension

The SDK7 captured-diffusion reception observed successful solves, histories, auxiliary values,
checkpoint and restart, but the restored `program_diagnostics()` map was empty. The native
accepted snapshots already own this map for rollback; the durable Uniform and AMR payloads did
not. This extension preserves that accepted resource without changing the physical equation,
solver controls, history policies or the AMR accepted-state/face-flux codec.

`pops.program-diagnostics.archive@1` adds two sealed NPZ members: a `uint8` vector named
`program_diagnostics_state` and signed `int64` offsets named `program_diagnostics_offsets`.
Each rank contributes one owned image, including empty maps. Its `POPSDIA1` header contains
native Real width, rank, rank count and record count as little-endian 64-bit words (40 bytes
including magic). Each record contains a 64-bit name length, the opaque native string bytes,
and eight bytes carrying the exact Real bits. Names are strictly ordered and unique. The native
writer's existing reserved `pops.balance-term` prefix remains forbidden. No UTF-8, NUL, finite
value or name-length restriction is added: signed zeros, NaN payloads, infinities and arbitrary
native string bytes round-trip. Float32 images additionally refuse nonzero upper value bits.

Rank ownership is checked in both the image and offsets. Diagnostics legitimately differ by
rank; neither capture nor restore requires equal maps. An archive carrying this extension needs
the same rank count. Historical archives without both members explicitly restore an empty map;
one missing member is corruption. Legacy AMR rank-count rematerialization remains available when
this rank-owned resource is absent. No remapping of local diagnostic values is invented.

Both native facades prepare a detached map, decode/allocate and drain their fence before the
owner-lane error vote. Python byte/type extraction runs through a plain producer function pointer
and context inside that native preparation; no std::function allocation occurs before its vote. Replacement uses a nonthrowing map swap inside the existing restart
transaction. System rejects pending native outcomes as well as active attempts/restart during
capture. AMR capture uses its local durable package-lane accessor, avoiding hidden collectives
inside the Python local preparation vote. Restore follows history replay and, for AMR, hierarchy
transformation, so temporary replay diagnostics cannot overwrite the accepted table. A later
failure retains the existing full accepted snapshot for rollback. Replacement clears stale keys;
it never merges them. The separate AMR Program/flux image is unchanged.

## Owner-selected resource capacity

`pops.program-diagnostics.checkpoint-capacity@1` introduces
`runtime.configure_checkpoint_diagnostics(capacity_per_rank=N)`. `N` is an exact positive Python
integer counting serialized bytes, not a universal maximum number of records. The live owner
chooses it collectively before capture or restart, including on a fresh runtime. Values must
fit the existing checked host byte-capacity arithmetic; booleans, coercions and overflow fail.
An invalid value on one rank is voted before peers compare proposals. Different valid proposals,
active native attempts/restart, pending consumer publication/recovery and active MultiLayout
mapping attempts refuse without replacing any authority. All child budgets are prepared before
the joined MultiLayout budget and wrapper are replaced.

The default inventories all unique literal `record_scalar` names in the verified compiler-retained
Program C++, plus the existing native `pops.frontier.duration` record. A full original field solve
emits three residual scalars and two evaluation counters, so their five names and bits are
included by their actual lowering. Overwrites/cadences do not increase map cardinality. The
per-rank default is exactly `40 + sum(16 + len(name_utf8))`; its rank gather reserves `N*ranks`
bytes and `8*(ranks+1)` offsets in the existing live NPZ/ZIP budget, including existing member
headers and manifest overhead. Missing retained C++ or dynamic names leave the capacity
explicitly unconfigured rather than assuming zero. Binding remains inspectable; capture of this
resource requires the public choice. Raw/external native records can exceed the default and need
an explicit larger reserve. The refusal identifies this method as the action to take.

The reserve is not loaded from the NPZ. Its authority is derived from the live base budget,
contract, owner rank count and explicit choice; cached byte counts and source inventory are
checked against it at capture/preflight/configuration. Both aggregate and individual rank limits
are enforced, including when one oversized shard would fit the aggregate. Public configuration
does not change the contents of an old checkpoint. It allocates no arbitrary global diagnostic
limit in C++.

The archive/transport reserve is checked locally and its admission voted on the owner lane
before `allgather_bytes`. A failing peer therefore cannot send an oversized image. Read-only
array validation copies only each fixed 40-byte header; owner preflight checks all shard lengths
and the aggregate capacity before copying the selected rank body for native validation. The
remaining post-gather checks defend transport results before joining the admitted images.

This contract does not bound total runtime RAM. The existing accepted native map and the local
native encoding can allocate their full contents before the serialized-length guard, and NumPy
archive decoding has the separate live NPZ/ZIP resource budget. In particular, a raw diagnostic
map larger than the selected reserve can create a large local encoded image, but that image is
refused before collective byte transport. Capacity is neither a native map allocation limit nor
a claim about peak memory during encoding, gathering, or archive construction.

## Retained source evidence and bind compatibility

`pops.compiled-program.source-evidence@1` is a frozen local attestation captured when
`CompiledSimulationArtifact` is constructed, covering retained block and Program text or explicit
absence. `verify()` recomputes and compares it; bind/capture never remints evidence from modified
`_generated_cpp`. This snapshot deliberately lives outside canonical artifact/component/bind
content projections and outside dataclass comparison. Paths/includes/comments may differ by
residence across MPI ranks. No ad-hoc source normalization is used. `_binary_evidence`, the
canonical component payload and artifact schema 2 remain unchanged, preserving the existing
installed component/bind content contract. A source digest alone does not prove C++ to DSO
compilation provenance. No old binary/checkpoint/SDK is silently upgraded: the new native methods,
bindings and SDK header closure require Root's coherent rebuild.

Read-only verification of the retained M19 rank-zero Program files reproduced hashes
`d7960969543a601b7b85ff51df5f8cbdb5b0def3cb52c217968c222aa29c4f00`,
`5251b02b4ced306d9a0d51c1683ddb028529e548f563020898e2fa7642983605` and
`4883f268b1f8a1b0d1e6a067371f9f0f9583460314d16351d521b14982faf664` under
`m19-product-bind-content-mpi2-dim2/rank0-tmp/test_native_product_reduce_lif0/pops-native-cache`
in Root's `/Users/romaindespoulain/dev/tmp/pops-api040-native-reception-evidence-20261001`.
The closed donor inventory has no authenticated rank-one Program text. Equal provider DSO hashes
in the separate pending inventory are not used as a substitute for that missing comparison.

## Bounded checks and remaining reception

The allocation-order follow-up at base `0298d696` passed 28 source/runtime checks and the three
existing extracted native codec/restore/real-header syntax checks on 2026-10-01. Four new tests
observe the actual Python helper's refused local/peer transport, unchanged payload, fixed-header
copies and admitted owner-only body copy. Transport and peer votes in those tests are explicit
substitutes; they do not qualify real MPI or Kokkos execution. No native header changed in this
follow-up.

On 2026-10-01, source-only Python used this private checkout through explicit `PYTHONPATH=python`
and the existing `pops-api040/bin/python`, without installing or executing its native package.
The two review files run with:

```
env PYTHONPATH=python PYTHONDONTWRITEBYTECODE=1 /Users/romaindespoulain/miniforge3/envs/pops-api040/bin/python -m pytest tests/review/test_sol61_program_diagnostic_checkpoint.py tests/review/test_sol61_program_diagnostic_checkpoint_native.py -q -p no:cacheprovider
```

The final coherent selection passed 27 tests in 16.11 seconds. Ruff and diff checks passed. The Python checks exercise the actual public method, budget/helper functions, genuine field
lowering and actual immutable artifact verifier. Platform metadata, owner storage, communicator
transport and MultiLayout aggregate transport are explicit source-test substitutes. The C++ host
checks compile the real codec and actual System/AMR capture/validate/restore bodies for Float64
and Float32; storage, fence, error votes and lane transport are substitutes. They cover opaque
names/bits, every truncated prefix, malformed lengths/counts/order/ownership, active gates, local
producer/fence failure, simulated peer failure, unchanged maps on refusals, exact replacement and
legacy clear. Two extracted real pybind chains also pass C++23 syntax against the actual Dim2
System/AMR facade headers, Kokkos/OpenMP and Python/pybind headers, without linking or execution.
The separate API-member-pointer TU passes with those real facade headers. Initial syntax attempts
lacked the Kokkos/OpenMP compiler flags and were corrected without source changes. These are not
Kokkos/MPI runtime acceptance.

Cost is linear in diagnostic bytes/entries: each rank holds its native map and prepared replacement;
checkpoint capture gathers only diagnostic images and creates the sealed arrays. The archive
budget bounds serialized resources; native map nodes also require ordinary per-entry allocation.
Root must replay real serial/MPI Uniform and AMR checkpoints/restarts, active-attempt refusal,
one-rank malformed input, rollback and MultiLayout cases on the rebuilt SDK. No PDE convergence,
AMR/GPU numerical qualification or original DD2 native success is inferred from these checks.
