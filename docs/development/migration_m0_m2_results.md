# PoPS M0–M2 migration results — candidate7 evidence report

**Evidence refresh: 2026-09-08.** M0 / [ADC-900](https://linear.app/romain7522/issue/ADC-900) is
**Done for baseline capture**. M1 / [ADC-901](https://linear.app/romain7522/issue/ADC-901) execution is
**complete within the frozen contract**, including the final green qualified Roe/HLL
automatic-differentiation replay. M2 / [ADC-902](https://linear.app/romain7522/issue/ADC-902) execution is
**complete within the frozen contract**, including the final compiler/identity and benchmark gates.
The final evidence index is complete. The migration project is
[PoPS Programming Model Migration — M0–M8](https://linear.app/romain7522/project/pops-programming-model-migration-m0-m8-cee53d2e326a).

This report records the current candidate7 evidence and the boundaries that remain after M0–M2
execution closure. The frozen M0 boundary and exact twelve-leaf inventory remain in
[`migration_m0_m2_contract.md`](migration_m0_m2_contract.md) and `notes/issues-root.json` in the
companion conversation checkout. Local commits are not presented as pushed GitHub commits.

## Scope and qualification rule

The public workflow remains equation-oriented `Model` / `Case` / `DiscretizationPlan` / `Program`:

```text
validate -> resolve -> compile -> bind -> run
```

The acceptance contract tracks seven independent evidence levels. A source witness does not prove
resolution, a resolved plan does not prove emission, an artifact does not prove execution, and a
run does not prove numerical correctness or performance.

| Evidence level | Required witness | Current reading |
| --- | --- | --- |
| `representable` | Typed authoring/IR record with stable identity or serialization | Present for qualified quantities, signed balances, applications, effects, resolved operations, access plans, and provider identities covered by the retained source tests. Joint fields, broad different-support maps, and later native routes remain bounded. |
| `validated` | Command, exit code, and accepted diagnostic or plan | Present for the listed source, native, provider, public-binding, and example lanes. Refusals and failed candidates remain recorded as such. |
| `resolved` | Resolved report with mappings, dependencies, effects, and disposition | Present for selected `ResolvedOperationPlan`, coverage/access, field, and provider reports. `explain` is resolution evidence only. |
| `emitted` | Generated/native source plus authenticated artifact identity | Candidate7 Dim2 wheel and selected native targets are authenticated by the retained proof and manifests. Unsupported operations still have to refuse before unusable artifacts. |
| `executed` | Exact command, configuration, exit code, and retained output | Present for candidate7 validation/native/example lanes, the final compiler/identity replay, and the completed candidate5 scalar profile. |
| `numerically_checked` | Declared oracle, tolerance, comparison method, and result | Present for the local MMS, primitive, parity, restart, and projection checks named below. No broad convergence or conservation claim is inferred from these lanes. |
| `performance_characterized` | Paired protocol, warmups/repetitions, device completion, raw metrics, uncertainty | Eight valid records and six strict paired comparisons are retained. The observations carry no threshold, speedup, or zero-overhead claim. |

## Source and artifact matrix

| Record | Source/state | Configuration | Evidence and boundary |
| --- | --- | --- | --- |
| M0 reference | `51bcdc2e2399dbc922c58c3008ae30a11332f1d8`, clean pinned baseline | macOS arm64, Dim2, Kokkos OpenMP, MPI/HDF5 environment; public Dim1/Dim3 science and GPU cells unavailable | [`migration_evidence/m0/index.json`](migration_evidence/m0/index.json), SHA-256 `9f32fe2da9ca0d719a2d084466c31f9db46b424621f968ae94545cf3be96bb86`. Failed and unavailable cells remain visible. |
| Candidate7 production | `bd583faf196f3c1faeedec489e04d24e959ed00b`, clean qualification8 checkout | Dim2, Kokkos OpenMP/Serial, MPI and parallel HDF5, `OMP_NUM_THREADS=2`, private candidate7 environment | Source for the authenticated candidate7 wheel and the five provider scripts. |
| Candidate7 wheel/proof | `m2-build-seventh-proof.json` | Installed Dim2 package, Kokkos and MPI present | [`migration_evidence/m1_m2/m2-build-seventh-proof.json`](migration_evidence/m1_m2/m2-build-seventh-proof.json): 915 installed members; wheel SHA-256 `4bc8fda4395f510f9e56710dbce6e769030730d48752ef67fc0eaa1c2ac49792`; native SHA-256 `d1eb7da32a5c0a7b2e7433dfdd309d03d4b19a8afaab5f426d03ad5ee875f518`; installed tree SHA-256 `9feb2a12f311ef57bcfcd33de9264c442ad47bdbb3ce93d70fa0a198dc543e04`; variants manifest SHA-256 `e4524b31919672d3bba33770360a3005ab1becc385be5eecf7b6b0e7cc34052f`. |
| Candidate7 native lane | Native summary record named `candidate6-native-summary.json` | Release, Dim2, clang++, Kokkos Serial/OpenMP, MPI enabled | [`migration_evidence/m1_m2/candidate6-native-summary.json`](migration_evidence/m1_m2/candidate6-native-summary.json): 156 serial passes, 3 MPI-only skips, 4/4 MPI CTest, and 7 effective MPI gtests at ranks 2/3. Its fingerprint proves the C++ tracked bytes are unchanged between native source `a760245b004008cd464ad44ef7b5c629c3e728db` and production source `bd583faf196f3c1faeedec489e04d24e959ed00b`; only the recorded Python projection guard differs. The basename is historical; it is not a candidate6 completion claim. |
| Candidate7 public InputAux lane | Test-only slot fix `39e03a8`; integration HEAD `625e01b`; production package/native bytes remain candidate7 `bd583...` / `d1eb...` | Dim2 installed artifact, public bind and field publication | [`migration_evidence/m1_m2/candidate7-auxiliary-public-bind-clean.log`](migration_evidence/m1_m2/candidate7-auxiliary-public-bind-clean.log) and XML: **9 passed in 75.58 s**. Exact `ComponentKey` binding, positive runtime input projection, native field-output ownership, and bare/foreign/derived/missing refusals are covered. |
| Candidate7 helper lane | Candidate7 installed native artifact | Dim2 OpenMP, source-kernel reuse probes | The final retry records ordinary `2.003255542 s` / `198416` bytes / `1` source implementation / `2` evaluations and guarded `3.327155250 s` / `235328` bytes / `2` implementations / `2` evaluations. These cache-affected observations are not cold timings or a speedup result. The initial `45.741679 s` / `46.165868 s` measurements remain historical beside the pre-fix stale diagnostic failure. |
| Candidate7 example source variants | Production `bd583...`; final multiphysics example-only snapshot inspection `7f577333dfc99fdbb642938b94fb26bbe3865b0d` | Same installed candidate7 native SHA `d1eb...` | The production example failure and the example-only snapshot fix are kept as separate records. The fix changes example inspection only; it does not change the installed native package. |
| Candidate5 scalar profile | `76319c0658187d9542c527870854321609a0f891`, clean qualification6 checkout | Dim2 Kokkos OpenMP, one rank, private candidate5 environment/native SHA `16342cd011e2dc8d002ee180971060ef5a3925367e4953595fc5ab2f49c8ce02` | [`migration_evidence/m1_m2/candidate5-examples/scalar/report.json`](migration_evidence/m1_m2/candidate5-examples/scalar/report.json) records the completed full scalar profile. It is not candidate7 source evidence. |
| Candidate5 → candidate7 scalar equivalence | `scalar-candidate5-to7-equivalence.json` / `.md` | Immutable candidate5 and candidate7 Python environments; Dim2 native loaders | Complete manual/preset native-loader and Program C++ are byte-identical, with identical science plan, reads, and parameters. Four conservative boundary-effect/provenance-ID changes are recorded; candidate7 full scalar execution and cross-binary/checkpoint compatibility are not claimed. |
| Final benchmark comparison | Candidate7 `bd583...` clean versus M0 `51bcdc2...` benchmark records containing the explicit `ExecutionLane` API repair | Dim2 rank-1/rank-2, same parameter/toolchain/protocol pairs | [`migration_evidence/m1_m2/candidate7-benchmark-comparison.json`](migration_evidence/m1_m2/candidate7-benchmark-comparison.json): 8 valid records, 6 strict paired comparisons, two warmups, seven repetitions, arithmetic ABBA ordering, repeated MG solves, and raw medians/MAD. |

The candidate7 wheel was produced after a clean native build already established the native bytes;
the retained packaging-only pass took `9.41 s` with maximum RSS `102727680` bytes. The earlier clean
native build took `118.81 s` with maximum RSS `1447313408` bytes. These are build observations, not
cross-host performance characterization.

The final curated index is present at
[`migration_evidence/m1_m2/index.json`](migration_evidence/m1_m2/index.json). It records five passed
profiles with separate source/native provenance, status `complete`, an empty pending list, and 66
byte-hashed files totaling `2,912,669` bytes.

## M0 baseline capture

M0 is complete as a reproducible archive of the observed reference configuration. It does not
convert baseline failures into behavior passes.

| Baseline item | Recorded result |
| --- | --- |
| Five reference profiles | All five returned `rc=1`: tutorial 06 lacked an explicit `ExecutionContext` for a non-serial artifact; tutorial 08 and full scalar hit an unhandled coarse/fine interpolation stencil; multiphysics rejected duplicate diagnostic quantities; IMEX-AMR lacked an exact owner for a planned collective. |
| Python contract selection | `206 passed, 1 failed` in the selected 15-file run. The failure was `test_emit_requires_lowered_module_provider_authority`. |
| Native selected suites | `22/22` passed for the selected auxiliary/component suites. |
| Installed-wheel proof | The initial RPATH mutation proof failed integrity; the supported `CMAKE_BUILD_WITH_INSTALL_RPATH=ON` rerun passed the installed-artifact integrity/provenance check. |
| Benchmark baseline | Eight valid dirty-profile records were archived for the revised Dim2/OpenMP rank-1 and MPI2 profiles with two warmups and seven measured repetitions. No clean-source, serial, GPU, aggregate-memory, kernel-counter, or memory-traffic result is inferred. |

Baseline hashes remain in the M0 index and companion evidence: example report SHA-256
`c2ca6437a004daa04bf3ae789e72f8c3d2b71f632c1f504b34975aeabf6ae2a3`; benchmark JSON SHA-256
`9e72c81eb7573129d7d35019b391a0bf39e8ea211030fd440aecf67b458e3899`; MPI2 benchmark JSON SHA-256
`282e66048b06f54ec3e28974ddcfe5d21e30d54775d5ac2c029ef03fdfa37b10`.

## Final compiler and benchmark closure

The final candidate7 compiler/identity replay is green on source `1c73f30`, with no production
source diff against `bd583...`: exit code `0`, `1107 passed`, `0 failed`, `30 deselected`, `2`
pytest property warnings, and `725.41 s`. The retained JUnit record contains `1107` cases. The
earlier `1106 passed, 1` stale TypeError-regex failure remains historical; the diagnostic-only fix
and final result are retained separately in
[`candidate7-codegen-identity-final-summary.json`](migration_evidence/m1_m2/candidate7-codegen-identity-final-summary.json),
its [JUnit XML](migration_evidence/m1_m2/candidate7-codegen-identity-final.xml), and its
[log](migration_evidence/m1_m2/candidate7-codegen-identity-final.log).

The final benchmark comparison contains eight valid records and six strict same-parameter,
toolchain, and protocol pairs: two warmups followed by seven measured ABBA blocks for arithmetic
and seven repeated cold MG solves. The table reports the JSON medians, candidate/baseline ratio, and median absolute deviation (MAD); the
ratios are observations without a threshold or speedup guarantee.

| Rank | Workload | Baseline median (s) | Candidate median (s) | Candidate / baseline | MAD baseline / candidate (s) | Samples |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | SAXPY then fill boundary | `0.000232854` | `0.000212708` | `0.913482` | `9.604e-06 / 2.7354e-05` | 14 |
| 1 | Lincomb then fill boundary | `0.0002487295` | `0.0002527715` | `1.016251` | `8.6665e-06 / 3.998e-05` | 14 |
| 1 | Geometric multigrid | `0.008588666` | `0.007943417` | `0.924872` | `2.32707e-04 / 4.875e-05` | 7 |
| 2 | SAXPY then fill boundary | `0.0002231665` | `0.000224979` | `1.008122` | `3.6479e-05 / 3.3375e-05` | 14 |
| 2 | Lincomb then fill boundary | `0.0002337705` | `0.000257438` | `1.101242` | `2.42495e-05 / 4.65635e-05` | 14 |
| 2 | Geometric multigrid | `0.315405875` | `0.239240833` | `0.758517` | `9.925416e-03 / 1.540501e-03` | 7 |

All six numerical validations in the comparison are finite and within their recorded error limits.
The benchmark is a Dim2 host observation: no GPU, hardware memory-traffic counters, aggregate
MPI-worker RSS, or zero-overhead conclusion is inferred. The raw comparison is retained at
[`candidate7-benchmark-comparison.json`](migration_evidence/m1_m2/candidate7-benchmark-comparison.json),
with executable, source and command provenance in
[`candidate7-benchmark-provenance.json`](migration_evidence/m1_m2/candidate7-benchmark-provenance.json).

## M1/M2 implementation evidence

M1 source witnesses include owner-qualified `Module.state_space` / `field_space` / `apply` /
`constitutive`, immutable application/effect records, signed `Balance.capture` and selected views,
and fresh stage identity under `python/pops/time/_program`. M2 source witnesses include
`ResolvedOperationPlan`, `build_resolved_operations`, `require_provider_packs`, `require_native`,
`explain`, `LoweringCoverageReport`, and `LoweringRejection`.

The direct native contract in `include/pops/core/model/physical_model.hpp` requires the exact typed
provider count and fixed call shape. Existing `PhysicalModelFor`/`CompositeModel` consumers remain
behind a checked supported-subset adapter. The aggregate adapter may conservatively load the union
of model/provider inputs; operation packs and standalone models retain exact inputs. A union load is
an access-plan guarantee and does not prove unused work was eliminated.

The public auxiliary-input boundary is exact. `pops.bind` accepts the declared runtime `InputAux`
value under its owner-qualified `pops.model.ComponentKey` (`owner_qid`, `space_kind`, `space_name`,
`component`). The candidate7 nine-test lane preserves the key and array, executes a positive
projection, and refuses a bare component string, a foreign owner, a derived key, or a missing input.
Native field outputs stay owned by the field provider; callers do not upload them as arbitrary
auxiliary values. This is the public authority exercised by
`tests/python/unit/runtime/test_auxiliary_public_bind.py`.

The candidate7 source updates retain exact Roe/HLL AD coordinates (`dbc922c`), selected fallible
effects (`61650a9`), exact InputAux binding (`8dbb0e2`), and the model-free projection guard in the
production `bd583...` source. The native fingerprint reference `a760245b` is kept separate from that
Python guard. Source fixes and source
tests do not close a native or example gate until the authenticated artifact executes the affected
path.

## Candidate7 validation lanes

| Lane | Result | Boundary |
| --- | --- | --- |
| Screened Poisson/MMS + primitive | **16 passed, 0 skipped** in `413.27 s` | [`candidate7-validation-report.md`](migration_evidence/m1_m2/candidate7-validation-report.md) and exact results/metadata. Full selected files ran without marker exclusions or reduced cases. |
| Provider-sensitive scripts | **5/5 passed**, no skip markers: multielliptic `133.468 s`, solve-fields-from-state `74.830 s`, predictor/corrector `124.770 s`, spectral predicate `2.341 s`, projection eig `54.531 s` | [`candidate7-provider-fixtures/summary.json`](migration_evidence/m1_m2/candidate7-provider-fixtures/summary.json) and `results.jsonl`; exact requested scripts only. |
| Native selected matrix | **156 passed**, 3 MPI-only tests skipped | The native summary and CTest XML record the selected 14-target matrix, not a whole-repository qualification. |
| Effective MPI | **4/4 MPI CTest**, **7 effective MPI gtests** passed at ranks 2/3 | MPI target records are retained beside the native summary. |
| LocalLinear/LocalNewton | **145 assertions passed, 0 skips/failures**: LocalLinear `25`, LocalNewton `120` | [`candidate7-native-time/results.json`](migration_evidence/m1_m2/candidate7-native-time/results.json) includes parity, analytic comparison, the full 8-case fault/action/rollback matrix, and fixed original `16x16`/`8x8` profiles. |
| Public auxiliary binding | **9 passed** in `75.58 s` | Exact public InputAux authority and field-output ownership; test-only slot fix only. |

## Authenticated reference profiles

The five normative reference profiles now have pass evidence across authenticated candidate
artifacts, with source provenance kept separate:

| Profile | Artifact/source | Result | Qualification boundary |
| --- | --- | --- | --- |
| Scalar tutorial OpenMP | Candidate7 / `bd583...` | Pass, one rank, `47.489559 s` | Execution evidence for the Dim2 Kokkos OpenMP tutorial; not a convergence campaign. |
| Scalar tutorial MPI2 | Candidate7 / `bd583...` | Pass, two ranks, `25.290489 s` | Execution evidence for the MPI2 replay; no broader MPI claim. |
| Full IMEX-AMR | Candidate7 / `bd583...` | Pass, one rank, `153.921117 s` | Retained manual/preset/restart outputs do not alone establish a manufactured-solution or conservation result. |
| Full multiphysics, default `N=8` | Candidate7 native package `d1eb...`; example-only snapshot fix source `7f577...` | **Pass, `74.45343079199665 s`** | First accepted step, HDF5/ParaView output, checkpoint, continuous and restarted second step, and bit-identical restart step 2 are retained. The source change fixes public `instance.layout_plan` inspection only; production package/native bytes are unchanged. The missing different-support mapping is retained as an explicit refusal. |
| Full scalar AMR | Candidate5 / `76319...` | **Pass, `3937.817645 s`** | Manual continuous/restarted and preset/SSPRK2 parity reached `t=0.4`, with `228/0` accepted/rejected steps per interval, bit-identical restart at step 456, AMR levels `(0,1,2)`, reflux + average-down, and regrid `92 -> 184`. Analytic check: at `t=0.193359`, L1 `3.331568e-4`, L2 `2.166155e-3`, Linf `3.548130e-2`, relative-L2 `1.700899e-2`. This remains candidate5 execution evidence; the scalar equivalence proof covers candidate7 source construction, not candidate7 full scalar execution or cross-binary/checkpoint compatibility. |

The earlier candidate7 production multiphysics record is retained separately: it reached one accepted
step and then failed stale example snapshot inspection with `KeyError: 'qualified_id'`. The final
example-only snapshot route corrected that inspection and produced the pass record above without
changing the installed package. The earlier candidate5 N24 projection diagnostic (`0` observed
versus `2` expected) is historical; candidate7’s authenticated nine-test public InputAux lane now
passes the positive projection and refusal cases. The candidate5-to-candidate7 scalar equivalence
record proves byte-identical generated native-loader and Program C++ for the inspected manual/preset
construction, while retaining the boundary-effect and provenance-ID differences.

## Twelve-leaf acceptance traceability

“Done” means the source/IR contract and selected witnesses satisfy the frozen M0–M2 contract. It
does not extend the qualification to bounded future generalization or unavailable hardware cells.

| Leaf | Current status | Evidence available | Scope limits and downstream work |
| --- | --- | --- | --- |
| **M0.1** | **Done** | Clean baseline pin, toolchain, artifact provenance, seven evidence levels, matrix axes, unavailable cells, candidate7 wheel/proof, and final 66-file index are frozen. | Baseline failures and unavailable cells remain part of the record. |
| **M0.2** | **Done for baseline capture; current reference profiles pass across candidate artifacts** | All five baseline failures remain archived. Candidate7 supplies three tutorial/IMEX passes and the final example-only multiphysics pass; candidate5 supplies the completed scalar profile. | Keep mixed artifact/source provenance explicit; do not collapse these into one candidate7 full-suite claim. |
| **M0.3** | **Done for inventory/freeze capture** | A1–A15, R1’s eight paths, exact refusal phases, and later joint/different-support scope are mapped in the contract. | Manufactured-solution/convergence/conservation and R1 qualification remain downstream. |
| **M0.4** | **Done** | Eight baseline records and the final strict paired comparison are retained. | Ratios, medians, and MAD remain observations with no threshold, speedup, or zero-overhead claim. |
| **M1.1** | **Done within the M0–M2 contract** | Qualified state/field identities, support/representation, repeated-instance separation, and joint/map records are covered by source and selected runtime tests. | Broader joint-field, support, dimension, backend, MPI, AMR, and restart combinations are future generalization scope. |
| **M1.2** | **Done within the M0–M2 contract** | Signed occurrences, multiplicity, explicit balance selection, partitions, and identity/nonlinear accumulation contracts are retained. | Wider discrete `Q_kappa(q+) - U^n` campaigns are future numerical qualification scope. |
| **M1.3** | **Done** | Exact Roe/HLL AD coordinates and source protocols are repaired; candidate7 LocalLinear/LocalNewton and public binding lanes pass. | Final private7 compiler/identity replay passes `1107/1107`, with `30` deselected and `2` property warnings; the earlier stale diagnostic failure remains historical. |
| **M1.4** | **Done within the M0–M2 contract** | Fresh `Program.stage` and qualified commit paths are covered; the full scalar manual/restarted/preset parity profile reaches `t=0.4` on candidate5. | Candidate5 execution remains separate; candidate7 has source-equivalence evidence, not a full scalar execution or cross-binary/checkpoint compatibility claim. |
| **M2.1** | **Done for selected M0–M2 scope** | Resolved operations, signed coverage, structured refusals, effect records, and selected lowering tests pass. | Joint fitted transport/diffusion, unsupported partitions, and temporal/exchange quadrature remain bounded future generalization. |
| **M2.2** | **Done for selected qualification** | Candidate7 provider scripts pass; exact public InputAux binding passes 9/9; AMR equivalence and explicit mapping refusal records are retained; full multiphysics final example executes with output/checkpoint/restart. | Broader different-support maps remain bounded. |
| **M2.3** | **Done** | Candidate7 wheel proof, native 156/3 plus 7 effective MPI, narrow direct contracts, checked aggregate adapter, helper reuse/evaluation counts, and final compiler replay are recorded. | Native and scientific claims remain limited to the authenticated dimensions, backends, ranks, and profiles listed here. |
| **M2.4** | **Done for current effect/performance scope** | Explain/coverage, guard/failure/collective/solve boundaries, source effects, checkpoint/helper, native MPI, LocalLinear/LocalNewton, structured refusals, and paired benchmark are recorded. | No GPU, hardware counter, aggregate MPI RSS, or zero-overhead conclusion is inferred. |

## Closure and remaining limits

* The final candidate7 compiler/identity replay is green: `1107 passed`, `0 failed`, `30 deselected`,
  `2` property warnings, and `725.41 s`. The pre-fix stale diagnostic-regex failure remains a
  historical record, and source `1c73f30` is the diagnostic-only correction.
* The final benchmark is green for its recorded validation: eight valid records and six strict
  same-parameter/toolchain/protocol comparisons using two warmups and seven repetitions. Ratios and
  MAD are observations; no threshold, speedup, or zero-overhead claim is made.
* Broad historical runtime diagnostics still contain failures and 116 explicitly unrun nodes.
  They remain visible and are not folded into current targeted totals or treated as new M3–M8
  requirements.
* There is no GPU or public Dim1/Dim3 scientific qualification. Native dimension contracts are
  separate evidence and do not advertise public numerical support.

## Curated evidence

The final seven-level index is complete at
[`migration_evidence/m1_m2/index.json`](migration_evidence/m1_m2/index.json): five passed profiles,
status `complete`, an empty pending list, and 66 byte-hashed files totaling `2,912,669` bytes. It
enumerates the directly referenced wheel proof, native/compiler summary and XML, benchmark comparison,
scalar source-equivalence proof, validation reports, and their provenance without requiring a recursive
output listing.

## First unblocked work

The completed M0–M2 gates unblock these entry slices:

* **M3.1 / [ADC-903](https://linear.app/romain7522/issue/ADC-903):** one authoritative
  `FieldProblem` with unknown/equation tuples, typed dependencies and physical boundaries, joint
  normalization, and independent field storage.
* **M4.1 / [ADC-904](https://linear.app/romain7522/issue/ADC-904):** heterogeneous joint
  applications and constitutive outputs with one evaluation context and union dependencies.
* **M5.1 / [ADC-905](https://linear.app/romain7522/issue/ADC-905):** scalar diffusion with an
  explicit gradient, signed balance, and boundary semantics. Its entry depends on M1.2 and
  M2.1/M2.2/M2.3; it can start in parallel with the other entry slices with M2 complete. Later
  M5.3/M5.4 still depend on M6.1/M6.2.
* **M6.1 / [ADC-906](https://linear.app/romain7522/issue/ADC-906):** general `SolveRequest` /
  `SolveOutcome`, including `Q(q+) - U^n`, failure disposition, and one native solve authority.

These are downstream work items. Their representability may already have M1/M2 witnesses, but no
future phase is completed by this report. M3.1 is the recommended first slice; the other three entry lanes can proceed in parallel.
Later child dependencies still govern integration across phases.
