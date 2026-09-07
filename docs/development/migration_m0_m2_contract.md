# PoPS migration M0-M2 acceptance contract

This document freezes the acceptance boundary for M0, M1, and M2. It began as a contract and
evidence ledger template. The frozen M0 evidence update below records supplied runs without
promoting a failure or an unqualified cell to a pass.

Provenance for this contract:

- Repository baseline: `51bcdc2e2399dbc922c58c3008ae30a11332f1d8`.
- Supplied migration specification attachment: SHA-256
  `4b50b6dd75201c2175dcc8f987acb42d3c04898c6bca55fd12cee84324976e40`.
- Mathematical reference `Notes_on_the_Abstraction_of_PoPS_sans_annexe.pdf`: SHA-256
  `dacaa9eaa8bdd55abd7366756601e91a4f8e7dfdc425d3b4495107ac3e613dfd`.
- Source record: Linear document `PoPS migration specification - pinned source and provenance`,
  sections 1-14 and the M0-M2 issue packages in `outputs/linear-migration-ready.json`.

Initial contract authoring performed no build, test, benchmark, or scientific run. The frozen M0
update records subsequent evidence against the same pinned source. Every result must identify the
exact source, configuration, command, artifact, output, and oracle that produced it.

## Contract boundaries

The public workflow remains equation-oriented `Model`, `Case`, `DiscretizationPlan`, and
`Program`, with the lifecycle:

```text
validate -> resolve -> compile -> bind -> run
```

Provider packs, access plans, lowering reports, and execution plans are derived compiler
records. They do not become a second authored model or a second temporal authority. Scalar
tutorial authoring stays at the same decision and complexity level. Compatibility adapters are
allowed only for a checked supported subset until a tested replacement exists.

M0 must complete the reproducible baseline and freeze the normative profiles before accepted
M1/M2 implementation is claimed. M1 preserves qualified scientific identity and authoring
expressions. M2 resolves complete numerical coverage, typed accesses, native operations, and
legality. A declaration, type, source witness, or registered test is not runtime or numerical
proof.

The following invariants apply throughout M0-M2:

- Declaration, instance, quantity, expression, SSA value, and storage identities stay distinct.
  Support, embedding dimension, shape, representation, units/domain, and sampling stay distinct.
  Names, source locations, and timestamps are diagnostic or scheduling data, never physical
  identity.
- Signed balance occurrences, multiplicity, selected balance, application projections, physical
  maps, and identity or nonlinear accumulation survive capture and lowering. There is no silent
  omission, duplicate coverage, divergence rewrite, discrete chain-rule replacement, or invented
  inverse closure.
- Expression capture happens during authoring. Symbolic truth tests reject; guarded and eager
  expressions retain their different validity and effect requirements. Native signatures retain
  domain, reads, outputs, effects, failure, build provenance, and required derivative routes.
- A field equation, discrete operator, solved value, and observation are separate records. A
  missing provider or dependency fails explicitly; no invisible solve or `use latest` fallback is
  accepted.
- A joint evaluation may share work, but it does not erase mathematical multiplicity. Effects,
  guards, solves, transfers, and collectives remain ordering boundaries.

## Evidence vocabulary and record schema

Each acceptance row records these seven evidence dimensions independently:

| Evidence dimension | Minimum closure evidence | What it does not prove by itself |
| --- | --- | --- |
| `representable` | Typed authoring/IR witness and a stable identity or serialization | Lowering or execution |
| `validated` | Validation command, exit code, and diagnostic or accepted plan | Native compilation |
| `resolved` | Resolved record or report with source mapping, dependencies, effects, and disposition | A runnable artifact |
| `emitted` | Generated/native source or operation record and artifact identity | Successful bind or run |
| `executed` | Exact command, exit code, runtime configuration, and retained output/artifact hashes | Numerical correctness |
| `numerically_checked` | Declared oracle, tolerance, comparison method, and result | Performance or generality |
| `performance_characterized` | Paired method, warmup/repetition protocol, device completion, raw metrics, and uncertainty | Scientific accuracy |

Use one of these statuses for each dimension: `pending`, `pass`, `fail`, `unsupported`, or
`unavailable`. `unsupported` means the contract intentionally refuses the request and records a
phase-specific diagnostic. `unavailable` means the requested evidence cannot be obtained in the
current environment or checkout and records the reason. Neither status is a waiver.

The following JSON shape is the minimum evidence record. It is illustrative; values remain
`pending` until a run supplies them.

```json
{
  "schema": "pops.migration.evidence.v1",
  "claim_id": "M2.3.scalar.native.direct",
  "source": {
    "revision": "51bcdc2e2399dbc922c58c3008ae30a11332f1d8",
    "worktree": "<absolute path>",
    "dirty": "<true|false>",
    "source_attachment_sha256": "4b50b6dd75201c2175dcc8f987acb42d3c04898c6bca55fd12cee84324976e40",
    "math_reference_sha256": "dacaa9eaa8bdd55abd7366756601e91a4f8e7dfdc425d3b4495107ac3e613dfd"
  },
  "environment": {
    "os": "<Darwin/Linux>",
    "architecture": "<arm64/x86_64>",
    "cpu": "<model>",
    "memory_bytes": "<integer>",
    "threads": "<integer>",
    "environment_name": "<conda env>",
    "compiler": "<name and version>",
    "cmake": "<version>",
    "kokkos": "<version and execution space>",
    "mpi": "<none or implementation/version>",
    "hdf5": "<none or version>"
  },
  "configuration": {
    "dimension": "<1|2|3>",
    "backend": "<serial|openmp|mpi|cuda|hip>",
    "mpi_ranks": "<integer>",
    "layout": "<uniform|amr|multiblock>",
    "levels": "<declared levels>",
    "blocks": "<declared blocks>",
    "method": "<method and coefficients>",
    "restart": "<fresh|checkpoint|regrid-restart>",
    "output": ["<format and consumer>"],
    "tolerances": "<all numerical and acceptance tolerances>"
  },
  "commands": {
    "setup": "<exact command>",
    "build": "<exact command>",
    "validate": "<exact command>",
    "execute": "<exact command>",
    "compare": "<exact command or oracle>"
  },
  "artifacts": {
    "build_manifest": "<path and hash>",
    "native_artifact": "<path and hash>",
    "stdout": "<path and hash>",
    "stderr": "<path and hash>",
    "outputs": ["<path and hash>"]
  },
  "evidence": {
    "representable": {"status": "pending", "witness": "<path:line or report>"},
    "validated": {"status": "pending", "command": "<command and exit code>"},
    "resolved": {"status": "pending", "witness": "<report path/hash>"},
    "emitted": {"status": "pending", "witness": "<source/artifact path/hash>"},
    "executed": {"status": "pending", "command": "<command and exit code>"},
    "numerically_checked": {"status": "pending", "oracle": "<oracle/tolerance/result>"},
    "performance_characterized": {"status": "pending", "protocol": "<protocol/result>"}
  }
}
```

## Baseline configuration and normative profiles

One native artifact is one explicit compile-time dimension. The build scripts accept only
`Dim=1`, `Dim=2`, or `Dim=3`; a missing or conflicting dimension is an error. The matrix is the
product of the axes below wherever a profile requires the axis. A result for one cell never
closes another cell by analogy.

| Axis | Required values or rule | Initial disposition at this baseline |
| --- | --- | --- |
| Dimension | `1`, `2`, `3` | Dim2 is the M0 reference method; Dim1 and Dim3 are outside this baseline and are not qualified. Native metadata still declares 1..3. |
| On-node backend | CPU Serial and CPU OpenMP/Kokkos where the preset is available | Dim2 Kokkos OpenMP is the M0 reference; benchmark serial/threaded cells are recorded separately. |
| Distributed backend | MPI with the exact communicator and parallel HDF5 requirement | Dim2 MPICH/HDF5 build is recorded below; MPI2 runtime or benchmark cells are used only where the harness supports them. |
| GPU backend | CUDA/HIP only when a real device and required toolchain are present | Outside this M0 reference baseline and not qualified; unavailable on the current Darwin arm64 host. This does not imply unsupported native dimensions. |
| MPI ranks | `1`, `2`, and `4` where the manifest declares rank parity; use the exact case-specific set | Never infer rank behavior from one rank |
| Layout | Uniform, AMR, and multiblock only when the profile declares them | Record levels, refinement ratios, blocks, and regrid state |
| Method | The authored scalar SSPRK2, final multiphysics explicit/implicit, and final IMEX-AMR methods | Preserve coefficients, controller, timestep, and solver tolerances |
| Restart | Fresh run, checkpoint/restart, and regrid/restart where declared | Restart identity and retained history must be hashed |
| Output | HDF5, ParaView, diagnostics, and checkpoint formats used by the profile | Reopen and compare output identities and arrays |

The frozen M0 reference method is Darwin arm64, 16 GiB, and 8 threads, with environment name
`pops-migration-20260907`, Dim2 Kokkos OpenMP, and MPICH/HDF5 for the distributed build. Dim1,
Dim3, CUDA, and HIP are outside this reference-method baseline and are not qualified here. The
native metadata still describes dimensions 1..3; the M0 disposition must not be read as a native
support rejection. MPI2 cells are included only where the selected test or benchmark harness
supports them. ROMEO-only GPU campaigns remain unavailable locally.

The normative profiles are separate from the small adversarial cases below:

| Profile | Normative source and required scope | Conformance witnesses (diagnostic only) |
| --- | --- | --- |
| Scalar AMR | `examples/final/EXEMPLE_SPEC_FINALE_ADVECTION_SCALAIRE_COMPLET.py` and its test; include leaf-cell error, every-level synchronization, completed regrid, declared dimensions/backends/ranks, and restart/output cells | `docs/tuto/scalar_advection/06_openmp_amr_explicit_ssprk2.py`, `08_mpi_amr_explicit_ssprk2.py`, `10_openmp_amr_synchronous_ssprk2.py`, `13_openmp_amr_restart.py`, `16_openmp_cartesian3d_ssprk2.py` |
| Multiphysics | `examples/final/EXEMPLE_SPEC_FINALE_MULTIPHYSIQUE_CORE.py` and its test; include accepted outputs, restart, field dependencies, and all declared numerical checks | `test_multiphysics_core_example.py` small-cell subprocess and structured-refusal tests |
| IMEX-AMR | `examples/final/EXEMPLE_SPEC_FINALE_ADVECTION_IMEX_AMR.py` and its test; include reopened scientific formats, AMR lowering authority, rollback, continuation, and declared convergence/conservation cells | Test-level lowering and lifecycle assertions |
| Compiler/native | Lowering coverage and native-loader suites selected from the manifest; record source mapping, generated artifact, binding, and structured refusals | `tests/python/unit/codegen/test_module_lowering_coverage.py` and native-loader package tests |
| Performance | `benchmarks/manifest.toml` cases and its declared protocol; this harness covers `arith_halo` and `scalar_mg`, while ADC-700/ADC-757 are separate hardware campaigns | No performance claim from a template or a smoke run |

The pinned README documents `verification/manifest.toml` as the scientific campaign source and
documents its planner commands (`README.md:194-217`), but the pinned checkout currently has no
`verification/` directory. Mark the normative campaign cells `unavailable` with this exact reason
until the campaign source is restored or its replacement is explicitly reconciled. Do not replace
the normative matrix with `tests/test_manifest.toml`, a reduced case, or a smoke run.

## Frozen M0 evidence update

The following results are the supplied M0 reference-method evidence for the pinned source
`51bcdc2e2399dbc922c58c3008ae30a11332f1d8`. They are recorded separately from the acceptance
requirements above. Exact command transcripts, logs, and hashes remain the authoritative run
artifacts; the totals below do not fill missing evidence fields by inference.

| Item | Frozen configuration or result | Evidence disposition |
| --- | --- | --- |
| Reference host | Darwin arm64, 16 GiB, 8 threads; `POPS_ENV_NAME=pops-migration-20260907` | Configuration recorded; source and environment must remain tied to the run artifacts |
| Native reference path | Dim2, Kokkos OpenMP, MPICH/HDF5 | M0 reference configuration |
| Tutorial 06 | Scalar tutorial 06, default `32x32`, `t_end=0.1` | **Failed (`rc=1`)**: `a non-serial compiled artifact requires an explicit ExecutionContext at pops.bind`; retain the exit code and diagnostic as an M0 conformance failure, with no waiver |
| Tutorial 08 | Tutorial 08 at 2 MPI ranks | **Failed (`rc=1`)**: `coarse/fine interpolation stencil crosses a physical parent face; use a boundary-aware transfer provider`; retain the MPI exit code and diagnostic as an M0 conformance failure, with no waiver |
| Final scalar path | Full defaults, uninterrupted and restart variants | **Failed (`rc=1`)**: `coarse/fine interpolation stencil crosses a physical parent face; use a boundary-aware transfer provider`; the run did not produce accepted restart or numerical comparison evidence |
| Multiphysics path | Final multiphysics example with default cell count | **Failed (`rc=1`)**: `ConsumerManifest contains duplicate diagnostic quantities`; the run did not produce accepted output, field, or restart evidence |
| IMEX-AMR path | Existing final IMEX acceptance | **Failed (`rc=1`)**: `planned collective has no exact ConsumerGraph block/resource/operation/strategy/communicator owner`; the run did not produce accepted reopened-format, rollback, continuation, or AMR evidence |
| Contract tests | 15 selected contract files, 207 cases: 206 passed, 1 failed | Incomplete gate. `tests/python/unit/codegen/test_module_lowering.py:167-177` fails as the M2 invariant `test_emit_requires_lowered_module_provider_authority` (`DID NOT RAISE ValueError`); it is not waived or counted as a pass. |
| Native CTest suites | `test_aux_single_source` (9 tests) and `test_component_interfaces` (13 tests), 22/22 passed; log [baseline-native-tests.log](migration_evidence/m0/baseline-native-tests.log), SHA-256 `b94c290740788226c5aa1ab972c0baa8a006ecf8925f9dccae6ddd88fc86023b` | Pass for these selected native suites only; this does not close the failing Python contract gate or qualify the full numerical matrix |
| Native Dim2 MPI build | Compilation and wheel creation succeeded in 123.97 seconds; wheel-build maximum resident set size was 1,888,010,240 bytes | Build/emission resource evidence only; this does not qualify runtime, numerical behavior, or performance |
| Installed-wheel proof (initial attempt) | Proof rejected because post-install `install_name_tool` deleted RPATH after the manifest hash was recorded, leaving retained Dim2 bytes different from `variants.json` | Failed integrity/provenance evidence; the failure remains part of the baseline record |
| Installed-wheel proof (RPATH rerun) | Supported `CMAKE_BUILD_WITH_INSTALL_RPATH=ON`, same source pin, no source edits; 901-member installed-wheel proof; native extension SHA256 `c98574b452ab5ee138212a660c48fe638571f9b5e37d5c6b2eb708fa45ea069c`; wheel SHA256 `37d66611578bd0e4117171009c87b4605182764a60b2f141fa1f84befbfffb54`; MPI/HDF5, code-sign, and `doctor` checks healthy; cached configuration rebuild took 9.36 seconds with maximum resident set size 130,252,800 bytes | **Passed** for installed-artifact integrity and provenance under this configuration. This does not qualify runtime, numerical behavior, or performance |
| Native benchmarks | Full `benchmarks/manifest.toml` defaults: `arith_halo` and `scalar_mg`, warmups `2`, repetitions `7`; compare CPU serial/threaded with MPI2 where the harness supports it | Two four-record JSONL runs were produced at Dim2/OpenMP, concurrency 2, ranks 1 and 2; all record validations passed. Both runs are marked `source_dirty=true` because the benchmark harness received the minimal `ExecutionLane` API repair. They qualify the explicitly revised benchmark profile once that patch is recorded, not an unmodified clean-source performance claim for `51bcdc2`; no serial record is inferred |

The curated in-repository evidence bundle is indexed by
[migration_evidence/m0/index.json](migration_evidence/m0/index.json) (schema
`pops.migration.baseline-evidence.v1`, SHA-256
`9f32fe2da9ca0d719a2d084466c31f9db46b424621f968ae94545cf3be96bb86`). It includes the contract
failure log, the 22-test native CTest log, both benchmark JSONL files, the MPI2 launcher resource
log, and the five example logs plus report. The index records the source revision, native hash,
benchmark source delta, wheel-build resources, and unavailable metrics. "Complete" for this bundle
means that the observed failures, successful selected checks, and unavailable fields are retained;
it does not mean that every M0 behavior cell passed.

For review from the repository checkout, the retained benchmark records are
[baseline-benchmarks.json](migration_evidence/m0/baseline-benchmarks.json) and
[baseline-benchmarks-mpi2.json](migration_evidence/m0/baseline-benchmarks-mpi2.json); the example
report and logs are under `migration_evidence/m0/baseline-examples/`.

The benchmark output is
`/Users/romaindespoulain/dev/tmp/PoPS-migration-20260907-evidence/baseline-benchmarks.json`
(schema `pops.benchmark.v1`, four records, SHA-256
`9e72c81eb7573129d7d35019b391a0bf39e8ea211030fd440aecf67b458e3899`). Its metadata records
`git_sha=51bcdc2e2399dbc922c58c3008ae30a11332f1d8`, `source_dirty=true`, AppleClang
21.0.0.21000101, Release, Kokkos OpenMP, concurrency 2, MPI ranks 1, and double precision.
The initial benchmark build failed because `GeometricMG` now requires an `ExecutionLane`; the
failure log is
`/Users/romaindespoulain/dev/tmp/PoPS-migration-20260907-evidence/baseline-benchmark-build.log`
(SHA-256 `559b621bee332025b29a824ef4efd52cfbece62c74bc76618b0a5a4e385cae4a`). The recorded
repair adds `#include <pops/parallel/execution_lane.hpp>` and passes
`ExecutionLane::duplicate_world_collectively("pops.benchmark.scalar-mg")` to the existing
solver constructor in `benchmarks/src/scalar_mg_case.cpp`; it does not change the benchmark
method or numerical defaults. The repaired compile log is
`/Users/romaindespoulain/dev/tmp/PoPS-migration-20260907-evidence/baseline-benchmark-repaired-build.log`
(SHA-256 `964341b95faf35c07836aa4f1fadb7e98983c70c88c85ff6ef916882e856dd95`); it records a
successful `pops_benchmark` link and the duplicate-RPATH linker warning. The executable output
log is
`/Users/romaindespoulain/dev/tmp/PoPS-migration-20260907-evidence/baseline-benchmarks.log`
(SHA-256 `9e72c81eb7573129d7d35019b391a0bf39e8ea211030fd440aecf67b458e3899`). No compile wall
time, peak MRSS, host name, binary-size, or memory-traffic record is present in these benchmark
artifacts; those M0.4 metrics remain unavailable/pending. The resulting output is an explicit M0
benchmark revision at `51bcdc2` plus the recorded harness patch and qualifies that selected
profile once the patch is committed and its provenance retained. It must not be promoted to an
unmodified clean-source performance baseline.

| Benchmark record | Runtime configuration and validated result | Qualification boundary |
| --- | --- | --- |
| `arith_halo` / `saxpy_then_fill_boundary` | OpenMP concurrency 2, rank 1; median `0.000232854 s`; validation passed, `saxpy_error=0`, `lincomb_error=0`, `variant_difference=0`, tolerance `1.4210854715202004e-14` | Explicit revised M0 profile; source patch provenance required |
| `arith_halo` / `lincomb_then_fill_boundary` | OpenMP concurrency 2, rank 1; median `0.0002487295 s`; same zero-error validation | Explicit revised M0 profile; source patch provenance required |
| `arith_halo` paired comparison | ABBA median saxpy/lincomb time ratio `0.90498633858762778`; validation passed | No serial result is present; MPI2 paired result is recorded below |
| `scalar_mg` / `geometric_multigrid` | OpenMP concurrency 2, rank 1; median `0.008588666 s`, 19 iterations; residual `1.6208599851186278e-08` below limit `7.8911524462436492e-08`; manufactured max error `0.0002007008622438855` below resolution limit `0.03125`; validation passed | Explicit revised M0 profile; no unmodified clean-source claim |
| `arith_halo` paired comparison (MPI2) | OpenMP concurrency 2, rank 2; saxpy median `0.0002231665 s`, lincomb median `0.0002337705 s`, paired ratio median `0.8938312927152714`; validation passed | Explicit revised M0 profile; noisy host and no performance threshold |
| `scalar_mg` / `geometric_multigrid` (MPI2) | OpenMP concurrency 2, rank 2; median `0.315405875 s`, 19 iterations; same residual and manufactured-error validation | Explicit revised M0 profile; launcher MRSS is not aggregate worker MRSS |

The benchmark run used two warmup blocks and seven measured repetitions, with device fences and
MPI barriers as declared by the harness. The MPI2 output is
`/Users/romaindespoulain/dev/tmp/PoPS-migration-20260907-evidence/baseline-benchmarks-mpi2.json`
(the matching `.log` has the same SHA-256
`282e66048b06f54ec3e28974ddcfe5d21e30d54775d5ac2c029ef03fdfa37b10`). It contains the same four
records at `mpi_ranks=2`, all with `validation.passed=true`: `arith_halo` medians are
`0.0002231665 s` (saxpy) and `0.0002337705 s` (lincomb), with paired ratio median
`0.8938312927152714`; `scalar_mg` has median `0.315405875 s` and 19 iterations. The MPI2
launcher resource record is
`/Users/romaindespoulain/dev/tmp/PoPS-migration-20260907-evidence/baseline-benchmarks-mpi2-resources.log`
(SHA-256 `6abd2a6c0fdba39396a98b9f00e5a218f9b6e31eb8b796b977d171441dd1f9ec`) and reports
`15253504` bytes maximum resident set size for the launcher; it is not aggregate worker MRSS.
The host was noisy and `performance_threshold` is `null`. GPU cells are unavailable on this host;
no serial record is present, and missing kernel-count, memory-traffic, and aggregate worker-memory
metrics remain unavailable rather than zero.

`benchmarks/manifest.toml` declares the cases and protocol but does not enumerate a serial versus
threaded matrix. This M0 update records the selected OpenMP rank-1 and rank-2 cells; a serial
comparison remains unqualified until it is explicitly selected and run.

The example replay report is
`/Users/romaindespoulain/dev/tmp/PoPS-migration-20260907-evidence/baseline-examples/report.json`
(schema `pops.migration.m0.reference-examples.v1`, SHA-256
`c2ca6437a004daa04bf3ae789e72f8c3d2b71f632c1f504b34975aeabf6ae2a3`). It records source
revision `51bcdc2e2399dbc922c58c3008ae30a11332f1d8`, `dirty=false`, the installed native
extension SHA above, `POPS_NATIVE_DIM=2`, and `OMP_NUM_THREADS=2`; the host has eight hardware
threads. The runner strips `PYTHONPATH`, sets `PYTHONNOUSERSITE=1`, combines stdout and stderr in
each log, records `returncode`, timeout, elapsed seconds, command arguments, and a log SHA, and
returns nonzero when any profile fails. Therefore the eight-thread host description must not be
read as an eight-thread example execution claim.

The five retained logs under
`/Users/romaindespoulain/dev/tmp/PoPS-migration-20260907-evidence/baseline-examples/` and their
SHA-256 values are:

```text
scalar_tutorial_openmp.log  83923596ff17335079606274400ae04beaa44b42901bc14ee7623c37b92c2e96
scalar_tutorial_mpi2.log    ad96b167805bfdda451baaa715d8c1b30b841f39c73d0b77c6c5110b6bba5b5f
scalar_full.log              f421543dfd71224acf226f1283ff39f6aef0a931103c4e6fadcc40a8419c2eb0
multiphysics_full.log        62242f718744947f6c6dfda23ae3442db613f84ccf5d6796372371b87c6a76e2
imex_amr_full.log             b45cff38f6b7154262e13fb94073df4aef6f5ed940950308dce8e7dccd3dadc9
```

All five profiles completed with `rc=1` and `timeout=false`. The report is execution and failure
evidence; because every profile stopped at a contract diagnostic, it contains no accepted output
hashes, restart comparison, numerical oracle, or performance result. The runner source is
`scripts/run_migration_baseline.py` (SHA-256
`a31db2726e35f0960e38f31e28c354766bbfade35d9445ffe16895aad891c05d`); this helper is an evidence
driver and is not a replacement for the pinned repository source.

The successful installed-wheel proof rerun used the exact command below from the evidence
artifact, with the pinned source checkout as its working directory:

```bash
python -m pip wheel -v . --no-deps --no-build-isolation \
  --wheel-dir /Users/romaindespoulain/dev/tmp/PoPS-migration-20260907-evidence/wheels-fixed \
  -C build-dir=build/{wheel_tag}-dim2 \
  -C cmake.define.POPS_NATIVE_DIM=2 \
  -C cmake.define.POPS_HEAVY_MODULE_TU_POOL=2 \
  -C cmake.define.POPS_USE_MPI=ON \
  -C cmake.define.POPS_USE_HDF5=ON \
  -Ccmake.define.CMAKE_BUILD_WITH_INSTALL_RPATH=ON
```

The retained artifact is `pops-1.0.0-cp312-cp312-macosx_26_0_arm64.whl`; the proof output
reported 901 installed members and the hashes recorded above. The initial failed proof remains
historical evidence and is not overwritten by this corrected configuration.

The cold initial wheel-build resource record is
`/Users/romaindespoulain/dev/tmp/PoPS-migration-20260907-evidence/baseline-build.log` (SHA-256
`7c91fde04e10ec00db4ab99a681e58fb081e39d0139a3b31c46366541e9c510b`): 123.97 seconds and
1,888,010,240 bytes maximum resident set size. The cached RPATH rebuild record is
`/Users/romaindespoulain/dev/tmp/PoPS-migration-20260907-evidence/baseline-build-install-rpath.log`
(SHA-256 `0d35084361f7e87af69286eeb463d172ab4851e61cc04591457418198f92c3c2`): 9.36 seconds and
130,252,800 bytes maximum resident set size. These are wheel-build resources, not benchmark
runtime MRSS.

The 15-file test run is source-level contract evidence plus one explicit failure. In particular,
it does not close the M2 gate while `test_emit_requires_lowered_module_provider_authority`
fails, and it does not turn the selected cases into a full scientific qualification. Tutorial 06,
tutorial 08, final scalar, multiphysics, and IMEX are failed M0 conformance cells for the
diagnostics recorded above; reproducibility of a failure does not make its requested behavior
pass. These failures are the retained M0 baseline; remediation belongs to the narrow paths
declared for M2 and is not implied by this evidence document. The RPATH rerun closes
only the installed-wheel integrity/provenance check and does not change these execution or
numerical dispositions. The missing `verification/` checkout is still unavailable;
no verification result is fabricated. M0 uses the section 13 retained examples, selected contract
tests, and the benchmark harness listed here. The larger manufactured-solution and
method-of-manufactured-solutions gate belongs to future M3-M7 qualification work.

## M0-M2 acceptance and traceability

The dependency order is `M0.1 -> {M0.2,M0.3,M0.4} -> M1.1 -> {M1.2,M1.3} -> M1.4 ->
M2.1 -> M2.2 -> M2.3 -> M2.4`. Parent milestones are completion gates; a child may report its
own evidence separately.

### M0 child-package closure

The four M0 child packages in `outputs/linear-migration-ready.json` define the completion
boundary (`outputs/linear-migration-ready.json:544-656`). For this baseline phase, capture is
complete for the observed scope when each selected command, configuration, result (including a
failure), hash, and unavailable cell is archived in the evidence index. A reproducible `rc=1` is
valid baseline failure evidence when its exact command, configuration, log, and diagnostic are
archived, but it is not a pass for the requested behavior. Failed behavior and unavailable
numerical/restart cells therefore retain those statuses and remain inputs to the downstream M2
and R1 gates; baseline capture does not waive those requirements.

| Child package | Exact manifest acceptance boundary | Current M0 disposition |
| --- | --- | --- |
| `M0.1` | Freeze clean source bytes, toolchain, flags, native provenance, reproducible setup/build/doctor commands, seven evidence dimensions, unavailable cells, and numerical profiles (`outputs/linear-migration-ready.json:550-554`) | `complete for M0 baseline capture`: pinned source/worktree state, host/profile, build configuration, native and wheel hashes, evidence axes, and explicit unavailable dimensions are recorded in [migration_evidence/m0/index.json](migration_evidence/m0/index.json). The missing `verification/` checkout and any unrecorded source-byte/archive or full normative campaign cell remain explicit limitations; no qualification is inferred from them. |
| `M0.2` | Run the selected tutorials, final examples, compiler/native tests, and expected refusals; archive commands, exit codes, hashes, numerical outputs, and replayable output/restart comparisons (`outputs/linear-migration-ready.json:578-582`) | `complete for M0 baseline capture`: all five example profiles are archived as `rc=1`, the contract run is 206/207, and the selected native CTest suites are 22/22. Exact report/log evidence and diagnostics are present; accepted numerical/restart outputs are unavailable because the examples fail before producing them, and the named M2 failure remains. Remediation is tracked separately for the narrow paths declared by M2. |
| `M0.3` | Map every §12 adversarial case, cover the required identity/field/flux/failure cases, specify complete numerical profiles, and define R1's eight native paths while separating later scope (`outputs/linear-migration-ready.json:609-613`) | `complete for M0 inventory/freeze capture`: A1-A15, normative axes, required conformance outcomes, and R1's eight paths are mapped. The later manufactured-solution/convergence/conservation and R1 execution evidence remain pending future qualification; small conformance cases do not close them. |
| `M0.4` | Compare equivalent methods with identical numerical settings; record runtime, solves, rejection, kernels, communication, memory, compilation, binary, device completion, uncertainty, and replayable raw measurements (`outputs/linear-migration-ready.json:637-641`) | `complete for M0 baseline capture`: the explicitly revised Dim2/OpenMP profile has rank-1 and MPI2 four-record JSONL outputs with manifest defaults, passing recorded validations, and retained compile/source-patch provenance. The clean original baseline's compile failure remains valid evidence; serial, GPU, aggregate worker-memory, kernel-count, memory-traffic, and other unrecorded cells remain unavailable or unqualified. |

For the `M0.3` requirement to define R1's eight paths, the manifest names these paths explicitly
(`outputs/linear-migration-ready.json:892-904`): unchanged scalar AMR; field-coupled transport;
Euler-Poisson source; heterogeneous interaction; explicit diffusion; implicit diffusion;
nonconstant-coefficient scalar field; and an imported native primitive. M0 freezes this list and
their dimensions/backends/ranks/layout/restart/output policy. R1 execution and its complete
conservation, exchange, and failure/retry matrix remain downstream; no R1 path is implied to have
passed by the baseline examples or benchmark records above.

This interpretation keeps M0 useful as a reproducible baseline with inherited failures while
preserving the manifest's completion gate. `complete for M0 baseline capture` describes the
archive/freeze work for the observed scope; `pass` still applies only to the evidence dimension
and configuration actually checked (for example, installed-wheel integrity or benchmark
numerical validation). Neither label changes an underlying `fail` or `unavailable` cell into a
behavior pass or closes a downstream M2/R1 requirement.

| ID | Acceptance gate | Pinned source/test witness | Closure evidence required |
| --- | --- | --- | --- |
| M0.1 | Freeze source bytes, toolchain, flags, artifact provenance, axes, tolerances, and reproducibility policy. Map existing implementation to migration delta. | `notes/repo-inventory.md:5-25`; `README.md:65-113,194-217`; `scripts/setup_env.sh:42-71,233-277`; `scripts/build_python.sh:17-26,70-88,162-211`; `tests/test_manifest.toml`; `benchmarks/manifest.toml` | Exact SHA/worktree digest, dirty state, environment report, selected matrix cells, and explicit `pending`/`unsupported`/`unavailable` rows. |
| M0.2 | Reproduce selected scalar AMR, multiphysics, IMEX-AMR, compiler, field, solve, and native paths. Capture accepted outputs and structured refusals without changing tolerances to hide failures. | Scalar target/test from `notes/repo-inventory.md:29-34`; multiphysics `examples/final/EXEMPLE_SPEC_FINALE_MULTIPHYSIQUE_CORE.py:427-458,557-569,642-740`; `tests/python/examples/final/test_multiphysics_core_example.py:28-116`; `tests/python/unit/codegen/test_module_lowering_coverage.py:18-55` | The frozen M0 evidence update records all five example profiles as `rc=1` with exact log/report hashes and the 15-file contract-test total. The diagnostics are reproducible failure evidence, not behavior passes; no accepted numerical/restart outputs exist, and the one named M2 failure remains an incomplete gate. |
| M0.3 | Extract every adversarial case, assign its owner and expected accept/refuse outcome, and freeze the complete normative profile. Small cases diagnose mechanisms only. | `notes/spec-review.md:31-50`; `examples/final/EXEMPLE_SPEC_FINALE_MULTIPHYSIQUE_CORE.py:458-490`; `tests/python/examples/final/test_multiphysics_core_example.py:107-245` | Case-to-owner map below, declared dimensions/backends/ranks/layouts/methods/restart/output, manufactured-solution/convergence/conservation or refusal oracle, and explicit later-scope cells. |
| M0.4 | Compare equivalent methods with identical equations, discretization, timestep/controller, solver tolerance, and output. Record execution, solve/rejection, kernel, communication, memory, compile, binary, and device-completion metrics. | `benchmarks/README.md:1-34,51-70`; `benchmarks/manifest.toml:1-42,52-99` | The full manifest default scope and `2` warmups/`7` repetitions are frozen. Four validation-passing records are archived for each selected dirty OpenMP rank-1 and MPI2 rank-2 run; clean-source, serial comparison, GPU, uncertainty, and any missing metrics remain unqualified. |
| M1.1 | Preserve qualified quantities, support/type/domain, shape/representation, immutable expression identity, and repeated-instance separation. Represent joint unknowns/maps without claiming later execution. | `python/pops/model`, `python/pops/physics`, `python/pops/_ir`; `tests/python/examples/final/test_scalar_advection_final_example.py:26-67`; `test_multiphysics_core_example.py:69-93,211-245` | Distinct identities for homonymous/repeated quantities, serialized qualified references, explicit support mismatch rejection, and evidence fields separated. |
| M1.2 | Retain signed physical term occurrences and multiplicity; choose one physical balance explicitly; keep explicit/implicit partitions as views; preserve identity default and declared nonlinear accumulation. | `python/pops/physics/_board_rate.py:133-170`; `tests/python/examples/final/test_scalar_advection_final_example.py:37-42` | Accepted and refusal cases for signs/scales/duplicates, selected-balance record, discrete representation contract for `Q_kappa(q+) - U^n`, and no silent drop or chain-rule rewrite. |
| M1.3 | Give each operator application immutable identity, signature, projections, input domain, effects, and failure obligations. Capture once per specialization; symbolic truth tests reject; names such as `gamma` have no physical meaning without a declaration. | `python/pops/model/module.py`; `python/pops/codegen/module_lowering.py:158-174,300-317`; `python/pops/native_components.py`; `test_multiphysics_core_example.py:189-209` | Application/projection identity and evaluation multiplicity, eager/guarded legality, unrelated-`gamma` behavior, captured-vs-mutable parameters, and no Python callback in native execution. |
| M1.4 | Add fresh stage shorthand while preserving qualified Program authoring, exact time/stage identity, and scalar tutorial simplicity. Equal timestamps do not alias values. | `python/pops/time/_program`; `examples/final/EXEMPLE_SPEC_FINALE_ADVECTION_SCALAIRE_COMPLET.py`; `test_scalar_advection_final_example.py:52-67,124-146` | Distinct stage tokens/caches, authenticated handles, unchanged scalar authoring decisions, and compatibility evidence for the longer `StagePoint` spelling. |
| M2.1 | Resolve all selected signed occurrences exactly once unless an explicit decomposition proves otherwise. Preserve effects, orientation, measure, evaluation identity, temporal attachment, and reject unsupported joint-only partitions or rewrites. | `python/pops/codegen/module_lowering.py`; `python/pops/numerics`; `test_module_lowering_coverage.py:18-55`; current restricted rate behavior `python/pops/physics/_board_rate.py:145-170` | Complete coverage report, source-to-realization map, omission/double-count refusal, joint fitted-term disposition, and exchange records ready for accepted quadrature. |
| M2.2 | Derive typed reads, sampling, stencils, transfers, reductions, and communication from every expression and native input. Opaque reads are conservative; rank or shape equality never infers a map. | `python/pops/codegen`, `python/pops/numerics`, `python/pops/fields`; `test_multiphysics_core_example.py:107-116,211-245`; `tests/test_manifest.toml:339-348` | Access-plan and communication report covering coefficients, waves, boundaries, observations, composed halos, field-free paths, and explicit different-support maps. |
| M2.3 | Emit resolved operations directly through narrow native contracts; retain the legacy emitter only as a checked supported-subset adapter. Use fixed direct call targets, with no runtime physical-type lookup or Python callback. | `python/pops/codegen/module_lowering.py:318-352`; `include/pops/core/model/physical_model.hpp:58-91,204-213`; `python/pops/native_components.py`; native-loader suites in `tests/test_manifest.toml:350-408` | Source-to-native mapping, compile/bind/run artifact evidence, adapter legality, direct-call inspection, and parity checks for supported scalar and Poisson/screened-Poisson routes. |
| M2.4 | Explain equation-to-result provenance and effect-aware planning. Keep guards, failures, solves, transfers, and collectives as boundaries; preserve structured rejection and distinguish checked, assumed, missing, unsupported, and runtime obligations. | `tests/python/unit/codegen/test_module_lowering_coverage.py:29-55`; `scripts/ci_select_tests.py:1-15,1817-1875`; `tests/test_manifest.toml` labels and rank declarations | Explain report for each requested result, legality/refusal diagnostics, materialization/reuse/fusion decision, and measurement against M0 where performance is claimed. |

## Adversarial case traceability

These cases are intentionally small and isolating. They are conformance evidence for one
invariant, not substitutes for the normative numerical profiles. A case may have an existing
source or test witness while its runtime and numerical columns remain `pending`.

| Case | Trigger and required result | Owner | Existing witness | Expected status at M0-M2 boundary |
| --- | --- | --- | --- | --- |
| A1: repeated `rho` | Two model instances or blocks use homonymous `rho`; identities stay distinct, while valid identical kernels may share implementation. | M1.1 | `test_multiphysics_core_example.py:69-93,211-245` | Representable/validated target; runtime and numerical evidence pending |
| A2: same-time stages | Two stage values have equal physical time; stage and SSA identities do not collide in caches or output. | M1.4 | `test_scalar_advection_final_example.py:52-67,124-146`; `python/pops/time/_program` | Accept with fresh identities; numerical evidence pending |
| A3: unrelated `gamma` | A parameter named `gamma` is not EOS metadata unless explicitly declared as such. | M1.3 | `python/pops/codegen/module_lowering.py:158-174` | Accept ordinary parameter; EOS selection remains explicit |
| A4: signed balance | Author `-div(Fh) + div(Fd) + S`, preserving signs, scales, targets, and repeated occurrences. | M1.2/M2.1 | `python/pops/physics/_board_rate.py:133-170` currently rejects multiple/scaled terms | Migration target accepts retained IR; current narrow route is a known unsupported baseline |
| A5: omission or duplicate | A selected realization omits or covers an occurrence twice. | M2.1 | `test_module_lowering_coverage.py:18-45` | Structured coverage rejection; no artifact or numerical claim |
| A6: nonlinear accumulation | `Q_kappa(q+) - U^n` or nonlinear law is applied to an average without its representation/quadrature contract. | M1.2/M2.1 | `notes/spec-review.md:9-13,40-42` | Reject missing contract; accept only with declared discrete representation |
| A7: guarded failure | Symbolic truth test, eager invalid operand, or guarded division/sqrt/fallible operation is captured. | M1.3/M2.4 | `notes/spec-review.md:9-12,42`; `python/pops/model/module.py` | Truth-test rejection; guard preserves conditional effect and failure |
| A8: missing field map | A field/source/flux consumer lacks a typed provider or layout map. | M2.2; later M3.3 | `EXEMPLE_SPEC_FINALE_MULTIPHYSIQUE_CORE.py:458-490`; `test_multiphysics_core_example.py:107-116` | Refuse before plan/publication; no fallback or output |
| A9: field reuse invalidation | Density-only reuse is followed by a momentum, coefficient, boundary, or observation-rule change. | M2.2 representation; M3.3 execution | `test_multiphysics_core_example.py:149-183` field context witness | Invalidate the affected qualified object; execution pending until M3 |
| A10: heterogeneous projections | Unequal input/output shapes use one application with projections; projections do not re-evaluate or erase multiplicity. | M1.3/M2.3; later M4.1 | `python/pops/model/module.py`; `test_multiphysics_core_example.py:211-245` | Preserve application identity and repeated contribution; runtime pending |
| A11: joint flux coverage | A joint fitted drift-diffusion declaration supplies both terms, while an unsupported separate IMEX partition is requested. | M2.1; later M5.5 | `notes/spec-review.md:45-50`; `python/pops/codegen/module_lowering.py` | Cover both or reject the partition; never invent a decomposition |
| A12: shared gauge | A joint field problem has one compatibility/normalization rule; independent gauges would change the problem. | M1.1 representation; later M3.4 | `python/pops/fields/operator.py:139-170,243-250`; `tests/test_manifest.toml:589-592` | Joint representation required; native execution pending |
| A13: rank-local failure | A rank-local/native failure occurs before a required collective. All ranks reach coherent control flow and no partial state publishes. | M2.4 planning; later M6.2 | `tests/test_manifest.toml:261-264`; MPI temporal rollback entries `:135-156` | MPI execution pending; failure policy must be explicit |
| A14: atomic regrid/restart | Regrid or restart fails midway; accepted state, history, topology, and outputs remain unchanged. | M2.4 planning; later M6.4/M7 | `test_multiphysics_core_example.py:28-105,127-145`; IMEX test `:27-83` | Conformance witness only; full AMR/restart matrix pending |
| A15: different supports | A physical/phase-space read requires an explicit map/reduction; equal rank or shape cannot infer it. | M2.2 representation; later M7.3 | `notes/spec-review.md:8-15,80-85`; `test_multiphysics_core_example.py:107-116` | Explicit map or structured unsupported diagnostic; no 6D claim |

## Repository command ledger

The commands below are selected from the pinned `README.md`, `CONTRIBUTING.md`, scripts, and
manifests. The frozen M0 update identifies which reference paths and tests have supplied results;
all other commands remain replay commands. Substitute the exact matrix cell and retain
stdout/stderr plus all hashes.

Environment and dimensioned Python artifacts:

```bash
POPS_ENV_NAME=pops-migration-20260907 bash scripts/setup_env.sh --cpu
conda activate pops-migration-20260907
POPS_ENV_NAME=pops-migration-20260907 bash scripts/build_python.sh --dim 1
POPS_ENV_NAME=pops-migration-20260907 bash scripts/build_python.sh --dim 2
POPS_ENV_NAME=pops-migration-20260907 bash scripts/build_python.sh --dim 3
```

The build command must name exactly one dimension. Add `--mpi` only for a cell whose MPI and
parallel-HDF5 prerequisites are recorded. Do not reuse a serial artifact as an MPI or GPU result.

The captured M0 example replay used the read-only helper
`scripts/run_migration_baseline.py` with `--source` set to the frozen baseline checkout,
`--python` set to the dedicated environment's Python 3.12 executable, `--output` set to the
`baseline-examples` evidence directory, and `--phase baseline`. The helper's report stores the
fully resolved command for each profile; its actual environment also sets `POPS_NATIVE_DIM=2`,
`PYTHONNOUSERSITE=1`, `OMP_NUM_THREADS=2`, and removes `PYTHONPATH`. Replays must preserve those
settings and must retain nonzero exits rather than treating the helper's aggregate nonzero return
as an infrastructure-only warning.

C++ preset and selected Python paths:

```bash
cmake --preset serial
cmake --build --preset serial
ctest --preset serial --output-on-failure

python -m pytest tests/python/examples/final/test_scalar_advection_final_example.py
python -m pytest tests/python/examples/final/test_multiphysics_core_example.py
python -m pytest tests/python/examples/final/test_imex_amr_final_example.py
python -m pytest tests/python/unit/codegen/test_module_lowering.py \
  tests/python/unit/codegen/test_module_lowering_coverage.py
python -m pytest tests/python/unit/fields/test_field_operator_contract.py \
  tests/python/unit/fields/test_field_discretization_contract.py
python -m pytest tests/python/unit/time/test_solve_outcome_contract.py \
  tests/python/unit/time/test_field_solve_outcome.py
python -m pytest tests/python/integration/native_loader/test_external_component_package.py
```

Use the manifest-driven selector before expanding a changed-code test set. Its `cpp` and
`python` subcommands require `--changed-files`; preserve the generated explanation with the
evidence record. `CONTRIBUTING.md:43-50` requires affected selection to remain conservative and
to fall back to full coverage for unknown build inputs.

```bash
python scripts/ci_select_tests.py python \
  --changed-files build/changed-files.txt \
  --tests-file build/m0-python-tests.txt \
  --explain-file build/m0-python-explain.txt
python scripts/ci_select_tests.py cpp \
  --changed-files build/changed-files.txt \
  --explain-file build/m0-cpp-explain.txt
```

The README also documents the scientific planner, but its source directory is absent at this
baseline. Do not report these commands as a successful normative run until that availability gap
is resolved:

```bash
python scripts/check_verification_manifest.py
python scripts/run_verification.py \
  --suite pr \
  --dimensions 1 \
  --max-nodes 2 \
  --output build/verification/plan
```

For performance evidence, use the existing harness and retain its protocol metadata. The harness
uses device fences and MPI barriers, aggregates the maximum rank time, validates outside timed
regions, and reports robust statistics. It covers only the cases declared in its manifest.

```bash
cmake -S benchmarks -B build/benchmarks -DCMAKE_BUILD_TYPE=Release
cmake --build build/benchmarks --target pops_benchmark -j
build/benchmarks/bin/pops_benchmark --case=all --output=benchmarks.jsonl
```

`benchmarks/README.md:51-70` and the `romeo_armgpu`, `adc700_program_cutover`, and
`adc757_heterogeneous_numerics` manifest sections describe hardware-dependent campaigns. A CPU
run, missing device inventory, or local placeholder does not create GPU or performance proof.

The generic repository command above is a replay recipe, not the exact captured M0 command. The
captured records use the dedicated `migration-benchmarks` build, explicit Dim2/OpenMP settings,
the recorded harness patch, and separate rank-1 and MPI2 output files. Use each JSONL metadata
object and the retained build/output logs for the exact command and source-dirty state; do not
infer a serial or MPI2 result from the generic command alone.
