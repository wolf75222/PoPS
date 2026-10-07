# AMR auxiliary invalidations and cleanup publication

The coherent c0f18 diagnostic exposes a publication bookkeeping defect. During a
two-level bootstrap, a Field consumer publishes its physical auxiliary closure at
levels 0 and 1. The old global invalidation queue retains those provider identities.
Cleanup then evaluates the Analytic provider again at an unqualified topology
diagnostic point, and its original physical-Clock guard rejects the request.
The [actual diagnostic reception](/Users/romaindespoulain/dev/tmp/root-amr-aux-clock-coherent-diagnostic-reception-20261007.json)
records callback points and the bootstrap failure phase; it does not capture a full
native call stack or a successful AMR step.

The integrated Source correction is a common publication mechanism, independent of physical
equations, model names and component positions. It tracks an outstanding invalidation
for each provider and actual live hierarchy level, with topology/materialization
identity and a checked revision. A public dirty-provider list is the union of those
obligations. It cannot prove that a particular level has published a value.

Only complete, accepted native publication can acknowledge the captured obligation.
Ancestor candidates used for fine sampling are rejected and do not acknowledge the
ancestor's accepted image. A later invalidation retains its revision. Accepted
Field outputs and actual coarse/fine transport consequences invalidate their declared
dependent images. Candidate bookkeeping, carrier storage, registry metadata and
invalidation incidence participate in the same collective commit and rollback.
Input and topology invalidation writers refuse a pending Aux publication before
mutating accepted data; they do not silently overwrite a live candidate's authority.

Cleanup selects explicit invalidation roots and their prerequisite closure through
`ExactAuxiliaryRegistry::begin_invalidated_publication`. Clean accepted prerequisites
are reused with their accepted provenance. The diagnostic point does not trigger a
policy-only reevaluation of an unrelated clean physical provider. Ordinary consumer
publication retains the authored freshness policy. A truly dirty physical provider
without physical authority still fails the original guard.

| Contract | Version and meaning |
|---|---|
| `pops.amr.auxiliary-invalidation-incidence` | 1: provider/level/topology/materialization/revision obligations |
| `pops.amr.auxiliary-refresh-selection` | 3: exact publication purpose, roots, suppression mode and incidence |
| `pops.amr.exact-level-auxiliary-selection` | 3: exact consumer closure and level selection |
| Registry invalidation drain | 1: explicitly invalidated roots, clean accepted prerequisites |
| POPSAUX3, Uniform checkpoint9, AMR checkpoint12 | Wire representations retained; AMR capture and restore use each level's actual invalidated providers |
| Field input2, System package8, NativeABI13 | Existing data layouts and ABI values retained |

POPSAUX3 already represents `invalidated_providers` separately for each level. Capture
must record the actual incidence; restore validates each level and reconstructs the
union instead of requiring artificially equal lists. The accepted non-Input and
never-published guards remain. Historical equal-list images remain representable.

The new registry method changes an SDK header. The signed-header signature
is `967e6afd549028368716cf484593d8424775d415daacba937e03822565da93e8`,
over the same 383 signed headers. Old Header4b artifacts are incompatible with that
signature. Unchanged numeric ABI values do not qualify the old Native or component
DSOs: official reconstruction, relinking, runtime identity and an actual old-artifact
refusal remain required.

At this documentation checkpoint the R3 correction is integrated after
[independent Source review](/Users/romaindespoulain/dev/tmp/PoPS-independent-amr-aux-publication-protocol-review-20261007/candidate-r3-final-source-review.json),
SHA `56d99eda5cadf1af45df1236015fd0c55f7340f27e135e3730f37d491d4a3f81`.
Six independent registry tests are integrated but unexecuted. Their dimension 1/2/3
instantiations exercise registry selection and accepted metadata, not complete AMR
or backend qualification. The unchanged 13-case Field tranche, original installed
nonautonomous PDE and own-checkpoint retry must receive new native results after
integration. Partial AMR incidence, rejected ancestors, Field re-invalidation,
coarse/fine transport, asymmetric checkpoint roundtrip, failure rollback, regrid,
empty MPI ranks and temporary accepted halo restoration require actual runtime
witnesses. None is closed by a Source hash or a pair of standalone registries.

The [immutable R3 admission](/Users/romaindespoulain/dev/tmp/PoPS-amr-aux-incidence-admission-r3-20261007/admission.json)
retains the exact two changed production files and all 383 signed-header pins.
Earlier author R1/R2 patch and Source-receipt files were overwritten in their working
directory and are not independently rehashable; this
[preservation limit](/Users/romaindespoulain/dev/tmp/PoPS-amr-aux-incidence-admission-r3-20261007/preservation-limit.json)
is explicit. The runtime negatives, coherent diagnostic and R3 immutable admission
are retained separately. No original user work is removed.
