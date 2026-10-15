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
