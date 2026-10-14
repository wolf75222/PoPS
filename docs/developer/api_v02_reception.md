# API 0.2: implemented mechanisms and recorded local reception

The `0.2-experimental` profile now provides the five finite composition
mechanisms below through the real PoPS compiler and runtime. The local **v15d2**
reception recorded 30 `accepted_with_scope` decisions and five
`closed_with_scope` decisions. These are explicit decisions of the integration
agent, not a cryptographic signature or human approval, and they do not certify
every scientific family in the corpus.

This page summarizes a reception **before the PR commit and before these two
documentation files were added**. Its source identity is a working-tree diff,
not the later commit SHA. **This local reception does not establish a GitHub CI
result**; local tests, native runs and required GitHub checks remain separate.
The [machine-readable summary](api_v02_reception.json) records the exact hashes,
counts, selected constructors and limitations without binaries or large logs.

## What was added

| Initial gap | Implemented interfaces and behavior | Executable source |
|---|---|---|
| Dependent Bundle and remapping | `BundleSchema`, `Remap`, instance-bound Array/Index domains; complete geometry, maps, fields and history transfer/reconstruction/rejection. | [Bundle inventory](../../examples/api_v02/bundle_inventory_v15.py), [remap](../../examples/api_v02/bundle_remap_v14.py) |
| Complete continuation and evolving children | `ContinuationSchema`, `Candidate`, `Assess`/`DecisionAssess`, `Advance`; fresh capabilities, provisional children, one-time consumption, reassessment after Transform, atomic root publication and restart. | [Children on several supports](../../examples/api_v02/multisupport_children_v14.py), [decision graphs](../../examples/api_v02/decision_assess_v14.py) |
| Closed spatial composition | Closed `GraphFunction` regions, Array/CSR operators, `Use`/`Realize`, `Problem`/`Algorithm`/`Accuracy`/`solve`; the original residual and declared constraints are checked after the supplied algorithm. | [Spatial nonlinear solve](../../examples/api_v02/scientific_spatial_v14.py), [interval transport](../../examples/api_v02/method_interval_transport.py) |
| Closed distributed algebra | `Distributed`, local regions, requests/imports, reductions/scatters with declared ordering, structural transport, live versions, bounded storage and checkpoint shards. | [Distributed graphs](../../examples/api_v02/distributed_program_checks.py), [distributed implementation](../../python/pops/time/_program/distributed_arrays.py) |
| AD and adjoints | `jvp`/`vjp`, explicit dualities, finite trace derivatives and separately controlled implicit derivatives; domains and selection remain part of the contract. | [Implicit relations](../../examples/api_v02/implicit_relation_checks.py), [inventory derivative](../../examples/api_v02/derivative_inventory_v15.py) |

The scientific type checks also survive `.value`, closed calls and compilation
detachment; equal SSA arguments do not invalidate a legitimate map application.
Declared ledgers are bound to the exact application, recipient, branch and
published path. `checked_inventory_decode` rechecks the original nonlinear
inventory; a tolerated defect is recorded once, not applied again.
`CurrentDependency` declares a retained relation such as a current field plus a
legitimate history. A freely constructed Product is not a certified ledger.
See the [inventory contract](inventory_provenance.md) and
[complete examples](../../examples/api_v02/inventory_publication_v15.py).

## From authored equations to execution

The author writes numerical bodies in Python. Closed functions elaborate typed
SSA graphs; native data-dependent loops and solves execute in emitted C++.
The route is `Case/Program -> validate/resolve -> C++ -> DSO ->
pops_install_program(System*) -> ctx.install -> System::step`, with PoPS
transactions and MPI collectives. Host NumPy/stdlib oracles compare results
afterward; they do not provide the published trajectory.

The [entropy closure example](../../examples/api_v02/models_M18_entropy.py)
authors quadrature, nonlinear closure, transport and time integration in Python.
An additional locally archived N10 witness authored a new law and signed-substep
method after the engine freeze, then exercised the same route on MPI2 with
checkpoint/restart. Its conception remains v15d; its actual execution is v15d2
through the explicit CI-metadata-only transition. This is a bounded extension
experiment, not a claim for arbitrary Python or a complete physical family.

C17 maps authored operations/uses/SSA to emitted instructions; it is **not an
execution trace**. Loaded DSO identities, results, process exits and native
context provide separate execution evidence. The archived Program command
strings are redacted; 43 model-block sidecars retain authenticated original
compile commands. Neither kind is reconstructed into a fictitious command.

## Recorded evidence and its origin

The evidence archive is local and is **not published with this PR**. Archive
keys in the JSON identify retained files; they are not public download links.
Hashes identify recorded bytes, not a signature or a substitute for reading the
archive. No private absolute filesystem paths or authentication data are needed
to use the repository examples.

| Category | Recorded result | Unit and boundary |
|---|---|---|
| Official build23 | Dim1 and Dim2 MPI | Installed Python, SDK and native identities; not scientific acceptance by itself. |
| Source | 428 new + 456 legacy | 884 source test cases, separate from native execution. |
| CI logic locally | 60 passed | Local catalogue/routing checks; not GitHub CI. |
| New native groups | 37 groups, 92 distinct cases | 116 case executions and 252 rank observations. |
| Legacy native groups | 14 groups | 232 case executions and 312 rank observations. |
| C++ contracts | Eight targets; 74/78 legacy CTest passed, four expected single-rank skips; 24/24 structural CTest | 105 legacy MPI instances and 93 structural instances; do not add these to Python case counts. |
| Model/method corpus | 40/40 selected profiles | 28 models +12 methods, MPI size1; 83 retained C++/DSO artifacts and 44 state arrays. |
| Final index | 58 received evidence rows, no missing row | 30 scoped decisions and five scoped gap closures, with original receipts retained. |

The source working-tree diff is
`b048508a7d0d8f3e2b30d3d398ab2b8c7b39b9618ef9ed118d092844cee12a17`.
The final index SHA-256 is
`58fdbbdf91718ee05c6042baa10241a0e5c8555dcbe657ff208a6a6fe46ff4da`;
the decision record SHA-256 is
`1bec8623c1009700a5fa489d5574ef09cdef75dc44003004175cc504a702cba1`.
Build23 and freeze-manifest hashes are in the JSON.

The original corpus receipt keeps `complete_api_02_conformance=false`: its 40
profiles alone cannot close the API contracts. The final decisions use the full
index, including separate positive and adversarial witnesses. Earlier failures
and preparations keep their original identities and are not relabelled.

## Reproduce on a new checkout

Follow [CONTRIBUTING](../../CONTRIBUTING.md) and the canonical
[API 0.2 example guide](../../examples/api_v02/README.md), including its complete
`pops.validate`/`resolve`/`compile`/`bind`/`run` example. From the repository root,
with the `pops` environment active and the machine's MPI/toolchain configured:

```sh
bash scripts/setup_env.sh                 # Once per worktree.
env -u PYTHONPATH bash scripts/build_python.sh --dim 2 --mpi
env -u PYTHONPATH bash scripts/build_python.sh --dim 1 --mpi
env -u PYTHONPATH OMP_NUM_THREADS=1 KOKKOS_NUM_THREADS=1 \
  python -m examples.api_v02.campaign --output /new/absolute/reception-directory
```

Use a fresh output directory. Both dimensions must be installed together: M15
and M28 select Dim1; the other profiles select Dim2. The `python` must be the
installed `pops` environment, without a foreign `PYTHONPATH`. Native evidence
requires the full process to exit successfully, including MPI finalization.
These commands produce a **new** reception, not a reproduction of the old hash.

Additional receiver selections are declared in the
[test manifest](../../tests/test_manifest.toml). Use the existing
[change-aware CI instructions](../development/change_aware_ci.md) and
[verification scope](../development/migration_verification_scope.md); do not
replace a failing runtime test by a source-only check or count a skip as a pass.
Documentation checks use `bash scripts/build_docs.sh`.

## The 40 selected profiles

Each link leads to the constructor actually selected by the corpus dispatcher.
The [coverage catalogue](../../examples/api_v02/coverage.py) retains unqualified
obligations. `require_complete=True` rejects a full-family claim while those
obligations remain; the default constructs the finite profile below. A profile
run is not a theorem of stability, positivity, AP behavior or continuum
convergence.

| ID | Selected profile / constructor | Important limitation |
|---|---|---|
| M01 | [Constant-velocity scalar advection](../../examples/api_v02/models_M01_M14.py#L110) | Periodic smooth data; no general boundary or convergence qualification. |
| M02 | [Burgers flux in two axes](../../examples/api_v02/models_M01_M14.py#L141) | Selected data are effectively one-dimensional; shocks and general entropy behavior are unqualified. |
| M03 | [Compressible Euler with a pressure-offset EOS](../../examples/api_v02/models_M01_M14.py#L200) | One EOS and smooth periodic profile; no all-state positivity or shock qualification. |
| M04 | [Advection with constant anisotropic diffusion](../../examples/api_v02/models_M01_M14.py#L228) | One positive-definite constant tensor; nonlinear and discontinuous tensors are unqualified. |
| M05 | [Compressible Navier-Stokes with temperature-dependent viscosity and heat conduction](../../examples/api_v02/models_M01_M14.py#L314) | Planar profile; balance/minimum checks alone do not distinguish a cyclic permutation of the state. |
| M06 | [Enthalpy diffusion with a latent-heat plateau](../../examples/api_v02/models_M01_M14.py#L386) | One piecewise-linear temperature law; no general Stefan-interface convergence claim. |
| M07 | [Shallow water over topography](../../examples/api_v02/models_M01_M14.py#L418) | Stationary lake-at-rest profile cannot distinguish a no-op; dynamic wetting/drying is not qualified. |
| M08 | [Guiding-center advection coupled to Poisson](../../examples/api_v02/models_M01_M14.py#L498) | Field constraints and balances do not constitute a complete trajectory oracle. |
| M09 | [Magnetized Euler-Poisson](../../examples/api_v02/models_M01_M14.py#L519) | Explicit SSPRK2 profile, not Hoffart midpoint/Schur or an AP/quasi-neutral qualification. |
| M10 | [Drift-diffusion-Poisson](../../examples/api_v02/models_M01_M14.py#L541) | Selected coefficients and finite mesh; constraints and balances are not a full trajectory oracle. |
| M11 | [Ternary Poisson-Maxwell-Stefan](../../examples/api_v02/models_M01_M14.py#L576) | Three selected species with a common flux constraint; arbitrary mixtures are unqualified. |
| M12 | [Five-/ten-moment two-fluid Maxwell coupling](../../examples/api_v02/models_M12.py#L106) | 2D3V kinematics with zero third central heat moment; not a fitted general closure. |
| M13 | [Drift-diffusion reactions and Helmholtz photoionization](../../examples/api_v02/models_M13.py#L27) | One photo group and chosen rates; data invariant in the transverse direction. |
| M14 | [Cold two-fluid sheath and a capacitive circuit](../../examples/api_v02/models_M14.py#L86) | Pressureless selected regime; no universal sheath or circuit model qualification. |
| M15 | [One-dimensional HyQMOM, order four](../../examples/api_v02/models_M15_orders.py#L162) | The corpus selects the smooth order-four profile; other orders/regimes need their own evidence. |
| M16 | [HyQMOM15 with magnetic rotation](../../examples/api_v02/models_M15_M18.py#L229) | Universal hyperbolicity is false for an admissible oblique state; this finite success does not restore it. |
| M17 | [Regularized Fan-Li15 with a nonconservative path](../../examples/api_v02/models_M15_M18.py#L131) | One raw-moment segment path and finite profile; no universal path or stability claim. |
| M18 | [Entropy closure M2 with six quadrature nodes](../../examples/api_v02/models_M18_entropy.py#L101) | Finite quadrature and selected nonlinear closure; not all orders or continuum velocity space. |
| M19 | [Vlasov-Poisson-BGK in 1x1v](../../examples/api_v02/models_M15_M28.py#L197) | Truncated reduced phase space; not 3x3v, stiff/AP BGK or an H-theorem qualification. |
| M20 | [Reduced multispecies Fokker-Planck in 1x1v](../../examples/api_v02/models_M20_M21.py#L8) | Chosen moment constraints and velocity discretization; no general multispecies theorem. |
| M21 | [Electromagnetic gyrokinetic slab](../../examples/api_v02/models_M20_M21.py#L129) | Fixed transverse symbol; not full-f 5D, transverse ExB transport or tokamak geometry. |
| M22 | [Grey M1 radiation coupled to Euler matter](../../examples/api_v02/models_M22_M25_M28.py#L42) | One closure and exchange model; no universal optically stiff/AP claim. |
| M23 | [2.5D Hall MHD](../../examples/api_v02/models_M22_M25_M28.py#L198) | Ohmic resistivity is disabled by default; no general Hall stability or boundary qualification. |
| M24 | [Seven-equation Baer-Nunziato without stiff relaxation](../../examples/api_v02/models_M22_M25_M28.py#L408) | Chosen interface pressure/velocity and path; the realization oracle shares constitutive expressions. |
| M25 | [GPR thermomechanics with distortion and thermal impulse](../../examples/api_v02/models_M22_M25_M28.py#L284) | The oracle shares constitutive expressions; stiff limits and universal determinant preservation are unqualified. |
| M26 | [Nonlocal aggregation-diffusion](../../examples/api_v02/models_M15_M28.py#L46) | One cosine interaction kernel; finite energy checks are not a continuum convergence theorem. |
| M27 | [Cahn-Hilliard with constant mobility](../../examples/api_v02/models_M15_M28.py#L150) | One quartic potential and discretization; no arbitrary mobility or unconditional fully discrete guarantee. |
| M28 | [ALE advection on moving volumes](../../examples/api_v02/models_M22_M25_M28.py#L338) | Default uniform state tests geometric conservation; not every nonuniform ALE problem. |
| T01 | [SSPRK2 with stage Poisson fields](../../examples/api_v02/methods_T01_T12.py#L77) | One coupled guiding-center profile; stiff order and arbitrary couplings remain unqualified. |
| T02 | [IMEX Euler: local decay and global backward diffusion](../../examples/api_v02/methods_T01_T12.py#L137) | Linear selected split; no general high-order IMEX or AP claim. |
| T03 | [Extended-time Strang splitting for u'=t-u](../../examples/api_v02/methods_T01_T12.py#L175) | One solvable ODE and endpoint contract; not arbitrary boundary-driven splitting. |
| T04 | [Coupled elliptic blocks with a Schur solve](../../examples/api_v02/methods_T01_T12.py#L231) | Selected symmetric positive-definite operator; not arbitrary nonsymmetric blocks or preconditioners. |
| T05 | [Outer implicit update with an inner entropy Newton closure](../../examples/api_v02/methods_T01_T12.py#L299) | Two-velocity finite closure; no singular-root or branch-selection differentiation claim. |
| T06 | [Nonlinear argument partition in two residuals](../../examples/api_v02/methods_T01_T12.py#L342) | Two scalar residuals distinguish old/new arguments; not a general partitioning theorem. |
| T07 | [Variable-step BDF2 with genuine retained history](../../examples/api_v02/methods_T01_T12.py#L360) | Selected decay equation and startup; AMR history transfer is not part of this profile. |
| T08 | [Subcycling/reflux on a partially refined hierarchy](../../examples/api_v02/methods_T01_T12.py#L429) | Default hierarchy is frozen and accepts; parent rejection and remap refusal are separate tests. |
| T09 | [Two-velocity spatial BGK with a Duhamel interval flux](../../examples/api_v02/method_interval_transport.py#L50) | Frozen local equilibrium; not a complete nonlinear UGKS/AP or arbitrary ADER qualification. |
| T10 | [Relaxed Heun rotation with actual reached time](../../examples/api_v02/methods_T01_T12.py#L538) | One quadratic invariant and relaxation law; arbitrary invariants are unqualified. |
| T11 | [Directed Patankar-Euler production/destruction](../../examples/api_v02/methods_T01_T12.py#L559) | Default circulation is zero; empty species and nonzero circulation are separate tests. |
| T12 | [Euler face-flux correction: Richtmyer to shared Rusanov](../../examples/api_v02/method_euler_correction.py#L150) | One correction pass and EOS; not a general MOOD loop, coupled-field recalculation or AMR reflux method. |

## Backend and interpretation limits

Finite arrays currently use host C++ loops and `std::vector` storage within
real PoPS built with Kokkos/MPI. That is not GPU execution qualification.
No multinode scaling or general performance claim is made. Dynamic VJPs formed
by columns may be correct and expensive.

Schemas have fixed structure with cardinalities varying inside explicit
capacities. The current codec/closure depth limit is 128 and distributed GC
traversal limit is 256. Ownership is contiguous on the current communicator;
arbitrary-rank-count restart and MPI hardware-failure recovery are not promised.
Legacy `Field`/`MultiFab`/`StencilFlux` adapters retain their own constraints.
The general finite Array/Bundle route does not silently remove those limits.

Topology changes, nondifferentiable selections and singular relations need an
explicit domain or rule. A small original residual is not automatically a
solution-error bound. The compiler does not choose correct physical boundary
conditions, gauges, remaps or analytic hypotheses for the author.

For the cross-contract witnesses, N07's four initial negative cases check
rejection and unchanged time/step, not each full image or exact payload; a
separate parent-rejection case checks the complete image. The C12 Bundle
receiver retains three inventory entries while remapping geometry/maps. The
actual 2->1 nonlinear-inventory/BDF2/history transfer is a separate received N01
composition, not a property inferred from that fixed-cardinality receiver.

The local scoped decisions close the finite mechanisms in their declared
backend domain. Universal API/scientific-family certification, GPU execution
and untested physical variants remain unqualified. New code or a new target
must be validated under its own exact source and native identities.
