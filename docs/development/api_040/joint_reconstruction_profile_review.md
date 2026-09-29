# Independent review: separate profile-counter companion

Scope: docs/tooling only, independent source review of the companion observing
the retained `3d06cab` and `cf6dace` snapshots. No native campaign, installation,
core change or heavy build was performed by this reviewer. Public bind/run remain
the execution route; accessing `runtime._executor` for this observation was
explicitly authorized. This is an internal observation seam, not a new public API.

## What the counters mean

| Native name | Unit and scope | What it does not establish |
| --- | --- | --- |
| `kernels` | Count of instrumented Uniform ProgramContext operations; some sites add the number of RHS requests | Total Kokkos launches, device kernels, FLOPs or hardware work |
| `kernel_launches` | Separately named AMR ProgramContext counter | An alias for `kernels` or an exhaustive Kokkos trace |
| `scratch_allocs` | Count of scratch allocations if actually emitted during the profile window | All allocator activity, bind allocations, or allocations not instrumented |
| `scratch_peak_bytes` | Historical wrapper contract: largest individual scratch buffer, in bytes, if emitted | Peak live bytes, sum of buffers, RSS or total device memory |
| `mpi_messages`, `mpi_reductions` | Counts only if actually emitted at instrumented sites | All MPI calls/bytes, or evidence of no communication when absent |

The two installed SDKs both implement Uniform `count_kernel_` by incrementing
`kernels`; AMR's corresponding service increments `kernel_launches`. A grouped
Uniform RHS increments by its request count. Neither site is a Kokkos-tools
callback. Searching both SDKs found no increment sites for the two scratch names
or the two MPI names; their Python wrapper declarations alone do not make these
metrics available. Presence must be established from the actual native report.
An explicitly present integer zero and an absent key are different outcomes.

## Common observation authority

These files were independently hashed in both installed snapshots and match:

```
runtime/_profile.py: 09596141a8e4aee95076285323e9d23a5dd3724383819d15282154e6a5fba6b1
runtime/_system.py: 1ed71e2f117ba0da7633553cb5c2d351a861677c4bb3eb3e9cca93142ab02de1
include/pops/runtime/program/profiler.hpp: 7f662f73ed871c3eba086efd4d42a2d3e7f9f3b41d8f429475cfe5db92c1877e
```

`System.profile(Profile.Advanced())` resets/enables on entry and snapshots/disables
on exit. Opening it after a fresh public bind excludes compilation, binding and
bootstrap from the counters. The observations cover the requested run, not the
lifetime of the RuntimeInstance. `PerformanceSummary.total_s` is a sum of timed
scopes; nested scopes prevent treating it as a unique end-to-end wall-clock time.
Profiled timings must not enter the unprofiled v1 ABBA ratios.

## Final tool review

The companion imports the exact frozen v1 case only after verifying its script
SHA-256 (`425077468c34a5c925dce24857d645b765cc66582b566691b1db09d3cac0ff08`).
It uses the same five-component scalar User advection, 48×48 periodic grid,
cell averages, explicit Euler expression, FixedDt schedule and final checks.
The companion does not benchmark a new joint reconstruction against a different
baseline algorithm. Each version runs two unprofiled fresh-instance warmups
and three newly bound profiled instances. Every profiled final array must exactly
equal the unprofiled warmup; the two versions must also agree at the unchanged
v1 tolerance. Profiled wall times are not exported as v1 timing samples.

Both snapshot gates were executed here in read-only mode: all 1079 baseline and
1080 candidate manifest members plus both native binaries passed. The candidate
is pinned to `cf6daceaa028df1343a303751ec9ae8f9f5ce6b3`, native SHA-256
`4b458dabd54357b8aa3a4659cee8363a9a4510e55aaa81706bfe03a445d27037`.
The workers use separate `python -I` processes and caches; ambient Python,
SDK-header and native-variant overrides are removed. Package path, native hash,
SDK/header ABI, doctor, CPU/float64/one-rank platform and actual loaded images
are checked. This review does not replace those run-time checks or manual
inspection of the recorded compiler flags before a causal resource claim.

The enabled structured native snapshot is retained inside the profile window,
along with the closing summary, raw counters and raw scopes. Each selected
counter carries its own availability; a field present in only one version or
only some samples is not presented as a paired measurement. Values are preserved
without division by missing/zero values and without synthesizing allocation,
memory or communication totals. The source-level labels for `kernel_launches`
and the equality check against unprofiled states were tightened during review.

The independent harness `evidence/joint_resource_probe_review.py` passed **9/9**
tool-only tests: absent versus explicit zero; partial memory fields and distinct
kernel names; largest-buffer units; refusal of malformed/negative counters;
symmetric zero observations; missing lane/sample admission; frozen-case hash
refusal; and wrong candidate identity. Ruff also passed. These tests use small
counter fixtures and mocked receipt loading, never a mock native benchmark.

```sh
env -u PYTHONPATH POPS_RESOURCE_PROBE=/absolute/path/to/joint_reconstruction_resource_probe.py \
  /absolute/path/to/python -I docs/development/api_040/evidence/joint_resource_probe_review.py
```

Verdict: **bounded source-level agreement** for this separate macOS/CPU observation
phase. The actual profile session, numerical equivalence, live loaded-library
agreement and available counter values remain to be received by root. No MPI
scaling, GPU measurement, total-memory peak or actual resource improvement is
qualified by this review. Final author revision reviewed:
`c60d9dd9493f0f8e06cbedc271967a1315897218`.
