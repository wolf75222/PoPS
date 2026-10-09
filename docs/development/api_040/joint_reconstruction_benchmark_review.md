# Independent review of the 3d06cab → T2 performance protocol

Scope: source review and filesystem/provenance counter-tests only, by Astra. No
benchmark campaign, compilation of PoPS, package installation, native timing, MPI
execution, or modification of either implementation was performed for this review.
The author owns `joint_reconstruction_benchmark.py` and its protocol document in
the `PoPS-numerical-bodies` checkout. This review does not qualify the new joint
reconstruction's performance; the baseline can only execute the shared scalar body.

## Comparability

The shared authoring function defines the same five-component conservative linear
advection, velocities `(0.7, -0.4)`, 48×48 periodic grid, Rusanov face flux, scalar
User stencil, explicit Euler expression, and 12 requested FixedDt steps of `1e-4`.
There is no alternative reconstruction or physics selected by the release label.
The sine/cosine initial data include the appropriate sinc factors for true cell
averages. Every sample binds a fresh instance and copies that same array. Only
bind and public `run` are timed; state gathering/checks occur afterwards. The code
checks accepted-step count and final time, finite states, periodic component sums,
and reproducibility before comparing complete final arrays across releases.

The fixed `rtol=atol=1e-12` state comparison and `1e-10` unweighted-sum tolerance
are numerical non-regression gates. They are not spatial-accuracy tolerances or
proofs of agreement with an analytic PDE solution. The sum is proportional to mass
on this uniform grid. No performance acceptance threshold is invented: ratios
remain descriptive and must accompany raw observations and dispersion.

The runtime order is A/B/B/A, with two warmups and five measured fresh bindings/runs
per worker: **ten observations per version, twenty total**. There are only two
process-level observations per version; the ten samples are not ten independent
process replicates. This arrangement limits simple monotone drift but does not
eliminate thermal/load effects. Cold JIT compilation is measured once per version,
in A/B order, with separate empty artifact caches. It is not an OS-cold-cache
comparison or a statistically resolved compilation-speed estimate. The short run
measures public dispatch/transaction overhead as well as numerical work.

## Independently verified artifact

The retained baseline at workspace `outputs/artifacts/3d06cab` was read without
importing it. All **1079** manifest members matched their hashes; the manifest
digest matched `identity.json`. The native extension was independently rehashed:

```
source:   3d06cabee9db4a31c4d07fa4e00dc5a40d164155
manifest: 2a76043ebff3bb5405d3408b39867856b9dd635d67a72ca9040d6535794300ce
native:   4574ed6096650aa708ad040fd6ee056414a3cfa1735a0440bd74220170265beb
```

`otool -L` showed no external PoPS runtime library. Kokkos, MPI, HDF5, OpenMP and
other dependencies are external images, mostly resolved through `@rpath`; the
same Python executable alone cannot prove that these images are identical.
The old receipt's doctor result was not represented as a new doctor run.

## Findings corrected during review

The first draft read process RSS before the runtime samples, overstated the ABBA
sample count, could overwrite an existing plan, and authenticated the requested
candidate revision only against its own receipt. Its file validator accepted
parent-path and symlink escapes from the snapshot. It also lacked a cross-version
execution-platform/dependency check, and a subprocess timeout discarded captured
output. All were sent directly to the author before any campaign.

The revised source pins baseline commit/native/manifest/count, requires clean
source receipts and an independently supplied expected candidate commit, resolves
the imported package/SDK against the chosen snapshot, checks header/native ABI and
doctor, and rejects nonempty result directories. It fixes the platform to CPU,
float64, production System and one MPI rank. It fingerprints actual dyld images,
excluding owned PoPS DSOs, and compares runtime workers. The generated DSOs are
hashed and checked against each lane's cold artifact. RSS is sampled after runs;
missing generated C++ and inaccessible native counters are explicitly unavailable.

## Independent counter-tests

`evidence/joint_benchmark_identity_review.py` exercises the validator/driver without
importing PoPS or running a worker. Its small files are filesystem fixtures, not
fake native qualification. The original path and symlink cases were red (both
accepted). The revised validator/driver passes ten checks: an accepted unmodified
fixture; rejected changed source/native/manifest; rejected parent/symlink escapes;
independently pinned candidate revision and baseline manifest; and preservation
of an existing receipt, plus refusal of a non-macOS host before loading packages.
The driver tests mock only receipt loading so they can
exercise admission without importing or compiling either package.

Reproduce against the script under review:

```sh
env -u PYTHONPATH POPS_BENCHMARK_SCRIPT=/absolute/path/to/joint_reconstruction_benchmark.py \
  /absolute/path/to/python -I docs/development/api_040/evidence/joint_benchmark_identity_review.py
```

These tests do not establish loaded-library equality, compilation-cache reuse,
successful binding, or numerical equivalence in an actual paired campaign.

## Final boundary

**Bounded agreement for the proposed macOS CPU campaign protocol**, after the
above corrections, on author commit `b4469c15a15ae3131bbfd535503f7fcd22b41e85`.
Final files reviewed and tested are identified by SHA-256:

```
joint_reconstruction_benchmark.py  425077468c34a5c925dce24857d645b765cc66582b566691b1db09d3cac0ff08
joint_reconstruction_benchmark.md  c188816dbcdff3f6443cfc55d4146378addb52ad766ddd034f389edbc32351dd
```

The final source also compares compiler/standard/runtime ABI fields (excluding the
intentionally changed header signature), records SDK version and retained compiler
provenance, and handles a worker timeout by killing its process group, draining the
worker, and preserving output plus a failed receipt. That timeout path was read,
not exercised against a real compiling child. The final ten independent tests
pass; Ruff passes for the independent harness.

Before interpreting an actual campaign, verify the exact installed candidate,
review `cflags`/`lflags`/compile commands in the retained provenance, and record
machine activity. These build-option comparisons remain manual; recording an ABI
key does not prove them. Missing SDK/provenance requires external verification,
not an assumption of equality. The final driver refuses non-macOS hosts before
artifact loading. This review grants **no non-macOS dependency-comparability
qualification**; a future portable campaign needs another loaded-image check.

The public case/run API and symmetric internal execution-context construction
were reviewed in source. End-to-end feasibility, same-lane warm cache reuse,
loaded-image equality and numerical equivalence still require the real paired
run after root's native reception. No timing ratio or claim of speedup is delivered
by this review.
