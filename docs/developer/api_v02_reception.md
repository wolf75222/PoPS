# API 0.2: equations, composition and native reception

PoPS elaborates Python equations and numerical programs into typed graphs,
emits C++, compiles an artifact and executes it through its existing native
System or AmrSystem, Kokkos kernels and MPI collectives. Equations in the
examples are executable definitions. Independent NumPy oracles inspect the
result after execution; they do not supply the native trajectory.

This page follows the completion work after the original v15d2 reception.
**Final-source reception and GitHub CI are still in progress.** The
[machine-readable record](api_v02_reception.json) keeps successes and failures
with their own identities. The [original reception](api_v02_reception_v15d2.json)
remains historical evidence; its results do not certify later changes.

The source012 reception has passed 51 selected native C++ cases on MPI1,
MPI2 and MPI3 (306 rank observations), covering continuation images, Field
migration, retained face sources and hierarchy recovery. The official Dim2/MPI
build and installed identity check pass. Its Python-authored receivers exposed
failures in hierarchy reconstruction, checkpoint migration between MPI sizes,
Bool composition in a rollback predicate, and the depth160 preparation budget.
The initial failed receipts remain available; subsequent reader/oracle fixes
are recorded separately and do not turn an unexecuted correction into a pass.
A separate source009m H100 reception
passes 15 CUDA cases; it does not qualify source012 or establish MPI speedup.
The larger kinetic performance case exhausted its declared storage budget;
new CPU/GPU pilots with an explicit larger budget have not yet supplied accepted
timings. The depth160 source011 and source012 receivers reached their unchanged
900-second watchdog before native execution. The source013 Python preparation
passes in 652 seconds with unchanged depth; its native execution is still pending.

The source013 and source014 official Dim2/MPI builds also pass in both isolated
environments with unchanged source manifests and verified native identities.
Source013 receives 25 selected C++ contracts on MPI1/2/3 (150 rank observations);
source014 receives five diffusion/publication and atomic-promotion contracts on
those ranks (30 observations). Corrected test fixtures are separately hashed and
linked to the unchanged native core; the initial failures remain in the archive.
The source014 regrid campaign passes the late-refusal rollback case and exposes
two reader assertions after successful regridding. The reader correction and
its nine issuer counterexamples pass. Both complete receivers then pass in a
40.79-second replay with compilation forbidden and unchanged SDK/generated DSOs,
including checkpoint/restart and the next step; only the two readers are replaced.
Changed-rank restart now reaches MPI1 to MPI2 and identifies a missing live
flux-budget contract during post-requalification; this remains an open defect.

The large CPU/GPU pilots were stopped after approximately 90 minutes without a
completed measured repetition. Native stack samples locate both in distributed
continuation garbage collection, after compilation. These instrumented runs are
diagnostics, not performance measurements. Their exact jobs, terminal states and
stack traces are preserved. The next qualification must measure the corrected
collector and must still receive independent Field supports, persistent
representation obligations, nested implicit compositions and final-source CI.

## Source015 checkpoint

The official Dim2/MPI incremental build passes with an unchanged source manifest.
Eight Python-authored independent Field, typed-error capsule and interval-region
receivers pass in the actual installed PoPS runtime. Four temporal hierarchy
receivers also pass, including checkpoint/restart and rollback of all histories.
The next nested Path receiver fails scratch ownership; the separate original
typed-recovery receiver fails exact level-layout validation. Neither correction
is qualified by these earlier successes.

Ten selected C++ contracts pass on MPI1/2/3 (60 rank observations), covering
independent observations, assembly scratch, implicit barrier reentrance, CONT6
images, migration, temporal restart and loader transaction ordering. Six corrected
CFL, ProjectAndRecheck and complete-vector pairing fixtures pass on those ranks
(36 observations) against the unchanged native core. The overall fixture campaign
still fails: replacing an independent Uniform Field on an empty MPI rank exposes
a missing storage-incarnation check. Failed receipts remain failed.

The next isolated ROMEO CPU and H100 pilots each finish two complete numerical
oracle/checkpoint repetitions, then hit the unchanged 2400-second watchdog during
the third. Jobs 747939 and 747940 exit124. Instrumented stacks still identify
distributed continuation collection. These partial diagnostic runs establish
neither final-source GPU qualification nor a scaling improvement.

## Source016 checkpoint

The official source016d Dim2/MPI build passes in 290.77 seconds with unchanged
source inputs and authenticated installed module/SDK identities. Earlier attempts
fail packaging or a missing AMR method definition; their receipts remain failed.
The source016d nonsymmetric global Schur receiver executes in 208.59 seconds and
passes its unchanged complete-state and original-system oracle. The 25 device
kernels retain the same definitions; the compiler change factors generic host
allocation protocols and bounds host inlining. This local duration is not a
controlled speedup comparison with the earlier GitHub timeout.

Twenty-six selected C++ cases pass on MPI1/2/3 (156 rank observations), including
owned dynamic parameter carriers, collective AMR replacement, empty-rank storage
incarnations, retained observations and distributed collection. The overall C++
campaign fails three fixture groups and remains recorded as failed. Separate
replays against the unchanged source016d native core pass four cases after
fixing their Product schema and Kokkos initialization (24 rank observations),
then the package-cardinality case after initializing MPI before constructing
System (six rank observations). No native rejection or numerical assertion is
relaxed by these fixture repairs.

Both Python-authored AMR input-field receivers pass, including value refresh,
checkpoint/restart and rejected-attempt rollback. The 33/65/129-parameter cases
fail because bind classifies their byte budget as a competing execution context;
the adapter correction has 20 source tests but awaits new installed reception.
Both independent Uniform Field receivers execute their first step and then fail
checkpoint serialization because the exact provider prototype is not supplied.
That composition is still open. Final-source reception, the remaining histories
and representation compositions, ROMEO performance, and GitHub CI remain open.

CI setup now installs the same MPI packages and recommendation policy in the
SDK producer and its consumers. Exact header authentication remains enabled.
The source registration review passes 165 cases and finds one stale duration
mirror; that metadata correction passes the unchanged failing assertion in a
separate replay. The 125 Python groups cover all 1166 selected files exactly
once, with unchanged watchdogs. These source checks are not GitHub results.

## Source017 and source018 integration

Source017a passes the official Dim2/MPI build. Native Python receivers pass the
33-, 65- and 129-parameter cases, both independent Uniform Field checkpoint and
rollback cases, and the straight-branch Path composition. The two independent
regrid fixtures still accidentally declare their unknown as an input auxiliary;
the fold Path receiver exposes a trace keyed by SSA identity without its dynamic
invocation. Those three failed cases remain recorded as failed. Their corrections
are included in source018b. Their subsequent reception is recorded below.

The independent Uniform Field C++ checkpoint fixture separately passes MPI1/2/3
after installing its actual block map before provider configuration, against the
unchanged source017a native core. Source016d also completes six integral-feedback
receivers and thirteen of fourteen implicit/frontier receivers. The latter include
all eight constant, variable, diagonal and nonlinear-accumulation diffusion cases
with backward Euler and IMEX, and all four typed-recovery cases. Its remaining
failure is the fold Path composition; these successful subsets do not qualify it.

Source018b builds the actual PoPS Dim2/MPI module in 199.36 seconds, with unchanged
source inputs and an authenticated Kokkos/MPI/parallel-HDF5 identity. The earlier
018 and 018a build failures are preserved. The integrated source adds exact Field
support and representation metadata to History rotation, rollback and checkpoint;
Python-authored nonlinear coordinate transforms retain their actual read bindings;
and immutable distributed arrays avoid repeated scalar-payload garbage-collection
walks while preserving resource accounting and conservative handling of mutable
entries. Numerical, MPI and ROMEO performance reception is still in progress.

Source018b subsequently receives both nested Path examples in actual C++,
including their accepted face-integral comparisons. Nonlinear representation,
workspace rejection, continuation-history rollback and mapped condensation also
pass. The history/regrid campaign exposes three distinct defects: a numeric
preflight incorrectly applied to structural history retention, frozen history
descriptors omitted by the Uniform installer, and representation lookup requiring
a State owner for authenticated independent provider storage. Restart also
creates an unnecessary allocator for a domain-free legacy continuation image.
The failures remain indexed; source018c includes their corrections.

Thirty-five selected C++ cases pass on MPI1/2/3. Five additional cases pass after
correcting their fixtures against the unchanged source018b core: the Path
quadrature splits one duration into two halves, deferred histories obtain their
actual provider-issued storage and DSO identity, and mapped solver routes are
installed before their package. Four admissibility cases pass on MPI1/2/3 with
the new owned-capture header, including an actual Kokkos kernel with 65 parameters;
borrowed views and host containers remain rejected.

The official source018c Dim2/MPI build passes in 282.22 seconds with unchanged
sources and authenticated Kokkos/MPI/parallel-HDF5 identity. Five new C++ authority
and continuation contracts pass on MPI1/2/3. Native Uniform independent-history,
domain-free continuation restart, and nonlinear-coordinate-history rollback pass.
The nonlinear-coordinate restart receiver also passes after correcting its slot
order to the existing rotation contract; it still compares every slot, metadata,
checkpoint and subsequent accepted state.

The AMR independent-history receivers expose a separate restart defect: numerical
samples were restored, but the manager omitted frozen slot descriptors and their
declared capacity. Source018d materializes both from the installed contract before
collective publication. Its official Dim2/MPI build passes in 278.48 seconds,
but the two receivers still fail: provisional hierarchy construction precedes
that materialization and also needs the complete cold metadata table. Source018e
initializes this table from the installed contract before publishing the forest.
Its official build passes in 324.32 seconds. Both complete independent-history
regrid/restart receivers now pass in 153.03 seconds on that SDK, including the
subsequent step, without a source overlay.

The independent solved-field regrid witness also exposed two numerical choices
in its Python definition. Its local affine equation has no spatial Jacobian, so
it now declares identity preconditioning. Its central-difference step is one:
the affine derivative has no truncation error, while the previous step of 1e-6
caused cancellation above the unchanged 1e-13 stopping criterion. The failed
receipts are retained. Both complete native regrid receivers pass against the
unchanged source018d SDK with the recorded Python witness correction: checkpoint
and next step, then late rejection and complete forest rollback (298.72 seconds).

T10's explicit computed-frontier clipping policy passes its principal native
case. Two parameter variants expose cancellation in the Python relaxation
formula near the final time, including after an exact first-step restart. They
remain recorded as failures; the runtime's final-time guard and numerical
tolerances remain unchanged. The stable authored increment formula now passes
six native cases on source018e: principal macro-step, two exact restart variants,
one/two-level affine finalization and late rejection. The (0.75, 1.3) variant
retains its final step of 1.7763568394002505e-15; no near-frontier tolerance hides
that step. Original and restarted trajectories have identical states and receipts.

The direct C++ cold-history receiver also passes on MPI1/2/3 with the source018e
runtime unchanged. Its initial test-only assertion incorrectly expected empty
flux bytes; the corrected reader authenticates all four registered rings, two
empty-term slots per ring and exact end-of-input. Metadata omissions, extra rings,
incomplete restart publication and rollback checks are retained.

Six new signed-remainder ledger contracts pass on MPI1/2/3 against source018c
with a recorded header overlay. They authenticate source families, face coverage,
clock windows, affine publication and exact destination bases. These results do
not qualify the still-incomplete persistent deferred-flux protocol.

Nested implicit solves, composed commit delivery and persistent deferred AMR
flux expressions remain separate open work. Build success and source-only tests
do not establish their completion, nor final-source coverage of all 40 contracts.

The isolated CONT7 protocol passes four C++ contracts on MPI1/2/3 with a
source018e overlay. The first reception retains a signed-zero fixture failure;
the corrected fixture initializes exact IEEE bits before exercising the codec.
This protocol is not yet integrated with the physical AMR archive. The underlying
`Fab::set_val(-0)` defect is corrected separately: two native Kokkos tests pass
on MPI1/2/3, including Host/default storage, dimensions1/2/3, ghosts and copies.
Those CPU results do not establish CUDA or Float32 reception.

The depth160 diagnostic still times out during preparation after 450 seconds,
before native compilation. Iterative traversal and bounded-size encoding chunks
remove recursion limits but do not yet make this case practical. A proposed hash
reuse shortcut was removed after six counterexamples showed that it suppressed
observable second-pass metadata validation. The 32 targeted source tests now
pass with both protocol calls retained. The compiler must preserve extension
semantics while reducing repeated structural work; depth160 remains open.

CI scheduling now selects 135 groups, partitioning long parameterized receivers
into exact test units without dropping cases. The integration checks pass 231
source tests. The longer existing job classes are assigned using recorded case
costs; numerical tolerances and case trajectories are unchanged. The current
GitHub result still belongs to the previous commit and is not green.

## Contract of the language

The fixed engine understands values, regions, supports, ownership, effects,
communication, provenance and publication. It does not dispatch on corpus
identifiers or contain a table of physical models. A new physical law or
numerical method is a Python composition of these operations; adding a C++
scientific implementation is not the extension mechanism.

| Contract | Public constructs and executable examples |
|---|---|
| Values and regions | Bool, Int64, Float64, Product, Variant, bounded Index/Array; `GraphFunction`, map, scan, reduce, conditional and finite iteration. [Finite algebra](../../examples/api_v02/algebra_checks.py) |
| Instances and remapping | `BundleSchema`, typed geometry/maps/fields, versioned domain instances and explicit history reconstruction or rejection. [Remapping](../../examples/api_v02/bundle_remap_v14.py) |
| Full continuation | `ContinuationSchema`, Candidate, Assess, Advance, provisional children and root publication; the declared schema carries method memory and controller state. [Multiple supports](../../examples/api_v02/multisupport_children_v14.py) |
| Spatial realization | Occurrences, Use/Realize, owned deliveries and reads, interval-dependent equations and explicit recomputation after replacing a face contribution. [Nonlinear interval transport](../../examples/api_v02/method_nonlinear_interval_transport.py) |
| Relations and algorithms | Authored residual, constraints, algorithm, accuracy and independent final residual checks. [Global nonsymmetric Schur](../../examples/api_v02/methods_coupled_nonsymmetric.py) |
| Distributed algebra | Distributed domains, local maps, bounded requests/imports, reductions, scatters and live-version storage. [Six-dimensional kinetic transport](../../examples/api_v02/models_kinetic_nd.py) |
| Derivatives | `jvp`/`vjp`, declared pairings, finite-trace derivatives and controlled implicit relations. [Implicit derivatives](../../examples/api_v02/implicit_relation_checks.py) |
| Numerical admission | A Python-defined candidate predicate, including an energy inequality, becomes a native guard and retry decision. [Authored stability](../../examples/api_v02/method_authored_stability.py) |

`require_complete=True` checks that every component of the **selected equation
and composition contract** is present and tied to reviewed Python source. It
does not require a theorem about every physical regime. Missing components or
stale source hashes are errors. Hyperbolicity, constitutive assumptions and
closure validity belong to the selected mathematical model; they are not
missing API operators.

## The 28 models and 12 methods

The following constructors are selected by the corpus dispatcher. Their full
component lists, parameters, assumptions, companions and numerical receiver
paths are in [profile_contracts.py](../../examples/api_v02/profile_contracts.py)
and the machine-readable record. A row lists an executable contract, not a
PASS inferred from the model name. Reduced historical witnesses remain named
compatibility profiles and do not replace the selected constructor.

| ID | Selected contract | Python equations |
|---|---|---|
| M01 | periodic-advection | [`build_m01`](../../examples/api_v02/models_M01_M14.py) |
| M02 | godunov-burgers | [`build_m02`](../../examples/api_v02/models_M01_M14.py) |
| M03 | stiffened-euler | [`build_m03`](../../examples/api_v02/models_M01_M14.py) |
| M04 | oblique-tensor-diffusion | [`build_m04`](../../examples/api_v02/models_M01_M14.py) |
| M05 | two-dimensional-navier-stokes | [`build_m05`](../../examples/api_v02/models_M01_M14.py) |
| M06 | one-phase-stefan | [`build_m06`](../../examples/api_v02/models_M01_M14.py) |
| M07 | wet-dry-hydrostatic | [`build_m07`](../../examples/api_v02/models_M01_M14.py) |
| M08 | periodic-guiding-center | [`build_m08`](../../examples/api_v02/models_M01_M14.py) |
| M09 | magnetic-midpoint-schur | [`build_m09_midpoint`](../../examples/api_v02/models_magnetic_midpoint.py) |
| M10 | scharfetter-gummel-poisson | [`build_m10`](../../examples/api_v02/models_M01_M14.py) |
| M11 | ternary-maxwell-stefan-poisson | [`build_m11`](../../examples/api_v02/models_M01_M14.py) |
| M12 | five-ten-moment-maxwell | [`build_m12`](../../examples/api_v02/models_M12.py) |
| M13 | streamer-photoionization | [`build_m13`](../../examples/api_v02/models_M13.py) |
| M14 | cold-sheath-circuit | [`build_m14`](../../examples/api_v02/models_M14.py) |
| M15 | hyqmom-order-four | [`build_m15_orders`](../../examples/api_v02/models_M15_orders.py) |
| M16 | hyqmom15-b1 | [`build_m16`](../../examples/api_v02/models_M15_M18.py) |
| M17 | fan-li15-path | [`build_m17`](../../examples/api_v02/models_M15_M18.py) |
| M18 | discrete-entropy-m2 | [`build_m18_entropy`](../../examples/api_v02/models_M18_entropy.py) |
| M19 | distributed-vlasov-poisson-bgk | [`build_kinetic_nd`](../../examples/api_v02/models_kinetic_nd.py) |
| M20 | distributed-mixture-fokker-planck | [`build_kinetic_nd`](../../examples/api_v02/models_kinetic_nd.py) |
| M21 | full-f-geometric-nonboussinesq | [`build_gyrokinetic_nd`](../../examples/api_v02/model_gyrokinetic_nd.py) |
| M22 | gray-m1-material | [`build_m22`](../../examples/api_v02/models_M22_M25_M28.py) |
| M23 | hall-ohm-staggered-ct | [`build_hall_mhd_ct`](../../examples/api_v02/model_hall_mhd_ct.py) |
| M24 | bn-pressure-velocity-relaxation | [`build_m24`](../../examples/api_v02/models_M22_M25_M28.py) |
| M25 | gpr-thermomechanical | [`build_m25`](../../examples/api_v02/models_M22_M25_M28.py) |
| M26 | aggregation-global-convolution | [`build_m26`](../../examples/api_v02/models_M15_M28.py) |
| M27 | cahn-hilliard-fourth-order | [`build_m27`](../../examples/api_v02/models_M15_M28.py) |
| M28 | multidimensional-reynolds-ale | [`build_ale_nd`](../../examples/api_v02/model_ale_nd.py) |
| T01 | stage-constrained-ssprk2 | [`build_t01`](../../examples/api_v02/methods_T01_T12.py) |
| T02 | ars222-imex | [`build_t02_imex_second_order`](../../examples/api_v02/methods_coupled_nonsymmetric.py) |
| T03 | dirichlet-extended-strang | [`build_t03_dirichlet`](../../examples/api_v02/method_splitting_boundary.py) |
| T04 | global-nonsymmetric-schur | [`build_t04_nonsymmetric`](../../examples/api_v02/methods_coupled_nonsymmetric.py) |
| T05 | nested-entropy-newton | [`build_t05`](../../examples/api_v02/methods_T01_T12.py) |
| T06 | global-selected-relation | [`build_t06_selected_global`](../../examples/api_v02/methods_coupled_nonsymmetric.py) |
| T07 | variable-bdf-authenticated-history | [`build_t07`](../../examples/api_v02/methods_T01_T12.py) |
| T08 | recursive-amr-window | [`build_t08`](../../examples/api_v02/methods_T01_T12.py) |
| T09 | spatial-duhamel-interval | [`build_interval_transport`](../../examples/api_v02/method_interval_transport.py) |
| T10 | whole-macro-relaxation | [`build_t10_composite`](../../examples/api_v02/method_relaxed_composite.py) |
| T11 | network-mprk22 | [`build_t11_network`](../../examples/api_v02/methods_patankar_network.py) |
| T12 | versioned-amr-face-replacement | [`build_t12_amr`](../../examples/api_v02/method_flux_correction_amr.py) |

## Cross-composition reception

The AMR receivers examine the full accepted state and its histories, clocks,
geometry, field dependencies and face ledgers. They include partial refinement,
2:1 subcycling, actual coarse/fine face selection, late parent rejection,
checkpoint/restart and temporal reconstruction followed by another step.
The reconstructed flux history preserves exact rational coefficients and
source identities. Rounding for field arithmetic is explicit; reflux provenance
is not normalized with a tolerance to make an invalid identity pass.

Dedicated receivers strengthen the corpus's default examples: full mixed-tensor
transport for M04, the complete Poisson/advection trajectory for M08, moving
wet/dry data for M07, authored Bernoulli faces for M10, independent BN/GPR
oracles, and a T12 interface mask whose effect reaches uncovered coarse cells
through reflux. A time/step-only frontier check is reported as such and cannot
be relabelled as an equation assessment.

## Execution domain and resources

Array rank, number of components and structural depth are separate from the
native spatial dimension. Phase-space maps preserve coordinates, fibers,
boundaries and ownership; velocities are not smuggled into arbitrary physical
components. Cardinality, byte capacity, live versions and exact-coefficient
memory are declared resources. Exhaustion produces a typed failure before
accepted publication. Finite resources remain part of API 0.2's contract.

Structural storage, codecs and validation use iterative traversal rather than
fixed depth-128/256 rejection. Repeated validation schemas are shared without
skipping validations or caching a verdict by pointer. The deep source witness
shrinks from 119.85 MB to 6.80 MB while preserving 159 validation calls; native
compilation and execution require their separate receipt.

Finite and distributed scalar maps lower to actual Kokkos kernels. Compilation,
backend identity, device execution and MPI scaling are separate measurements.
The ROMEO campaign uses isolated H100 allocations and leaves existing jobs
untouched. GPU performance or speedup will only be reported from terminal jobs
with authenticated artifacts, device identities and numerical comparisons.

## What numerical checks establish

Authors can express residuals, invariants, positivity and energy conditions in
Python, compile them, and make rejection/retry depend on them. Native tests
exercise these decisions and their rollback effects. The compiler verifies
language and effect contracts; a mathematical claim needs its stated hypotheses
and evidence. A small residual alone is not a solution-error bound, and finite
experiments do not prove convergence for every equation in a physical family.
These are distinctions between executable contracts and mathematical theorems,
not reasons to refuse an otherwise expressible model.

## Reproduce

Use the repository setup/build workflow from [CONTRIBUTING](../../CONTRIBUTING.md)
and the [example guide](../../examples/api_v02/README.md):

```sh
bash scripts/setup_env.sh
env -u PYTHONPATH bash scripts/build_python.sh --dim 1 --mpi
env -u PYTHONPATH bash scripts/build_python.sh --dim 2 --mpi
env -u PYTHONPATH OMP_NUM_THREADS=1 KOKKOS_NUM_THREADS=1 \
  python -m examples.api_v02.campaign --output /new/absolute/reception-directory
```

Run setup once per worktree. Use the installed `pops` environment and retain
native package/SDK identities, generated C++/DSO hashes, process exit status
and numerical results. M15 selects Dim1; the selected multidimensional M28
uses Dim2. Dedicated receivers and MPI launch contracts are registered in
[tests/test_manifest.toml](../../tests/test_manifest.toml). The full GitHub gate
must pass on the actual PR head; local passes are not GitHub CI.
