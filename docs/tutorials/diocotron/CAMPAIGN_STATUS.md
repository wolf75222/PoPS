# Hoffart finite-volume campaign: active status

Snapshot: 2026-09-22. This is an unfinished research implementation. The
low-resolution results below do not qualify the higher-resolution campaign or
complete the paper's figures. This branch is intended for a draft PR.

## Implemented cases

- `01_mpi_kokkos_hoffart_euler.py`: full finite-volume Euler model on the mapped
  disk, MPI and Kokkos/OpenMP, AMR, checkpoint/restart, and actual-state output.
  The finite-magnetic-field Schur tensor is solved with composite FAC and a
  prepared GMRES coarse solve. Polar Poisson inversion is a preconditioner;
  it does not replace the full tensor equation.
- `02_mpi_kokkos_hoffart_hyqmom15.py`: exact HYQMOM15 closure with strict
  realizability/spectral checks. A general oblique-direction obstruction remains;
  no collisionless HYQMOM15 trajectory is claimed.
- `04_mpi_kokkos_hoffart_fan_li15.py`: a separately named full Fan-Li15 model,
  including its nonconservative path terms. It is not relabeled as HYQMOM15.
- `../hyqmom_bgk/`: separate homogeneous and Cartesian shock examples with
  Knudsen-dependent isotropic BGK collisions and the exact HYQMOM15 closure.
- `03_render_results.py`: density, schlieren, Fourier diagnostics, and GIFs from
  actual saved states. Unreached paper times and unavailable growth fits are
  explicitly omitted.

All five simulation tutorials are linear module-scope scripts without helper
functions or classes. The selected authors' benchmark remains unchanged:
`R=16`, ring `6<r<8`, density `0.9+0.1*sin(m*theta)`, background `1e-6`,
`alpha=3.94784176e13`, `Omega=-6.28318531e12`, `T=1e-24`, final target time `10`.

## Repairs and their evidence boundaries

| Area | Change | Retained validation boundary |
| --- | --- | --- |
| Mapped field coupling | Full tensor coarse GMRES, polar Poisson preconditioning, original physical infinity residual checks | Local operator/manufactured tests, the low ROMEO runs and the rebuilt local `97279de` L3 pilot through `t=0.01` passed; current-source Linux and longer trajectories remain pending |
| FAC hierarchy | Synchronize covered parents after corrections and after the periodic gauge; conservative fine-interface flux support | Original FAC tolerances pass; covered-parent regression checks pass at `1e-12`; not a general high-resolution convergence guarantee |
| MPI execution | Serialize public model artifact publication; synchronize AMR body failures before flux publication | Focused MPI/runtime witnesses; not all backend or multi-node coverage |
| AMR continuation | Preserve accepted-attempt authority, histories and auxiliary images; rebuild generated resources after accepted history remap | Genuine expanded cold/restart comparisons on `65b0658` and freshly rebuilt `8b05af9`; the latter starts from the first actual expanded interior checkpoint and reaches `t=0.03` |
| Shared history projection | Bind immutable shared samples to their physical archive, project parent flux densities into canonical child faces, and preserve complete source identities in the ledger | The rebuilt `52b3e26` reaches transformation and rollback on the real 16² MPI fixture, then exposes an invalidated history reference during continuation. Six focused native witnesses and the complete shared restart paths require qualification with the lifetime correction. The existing single-block scientific runs do not exercise shared flux projection |
| Native packaging | Authenticate relocated installed extension bytes | Official build/doctor and retained source/installed manifests |
| Uniform checkpoint shape | Distinguish native reversed array shape from legacy field-free logical shape, preserving byte guards | Revision `8ae9097`: 36 focused Python checks; actual one-rank fresh-process rectangular restart; final Knudsen checkpoint |
| Exact global gather | Replace arithmetic summation with byte assembly after exact ownership checks, preserving signed zero | Official Dim2/MPI rebuild at `4adfddf`; 36 Python checks, four genuine MPI1/MPI2 capture/restart worlds and six MPI1/2/4 gather CTests passed, including signed zero and refusal before mutation |
| Generic C++ core | Generate Fan-Li physics in Python over variable-count/dimension-generic path-flux and moment primitives; remove the three model-specific C++ headers | 49 affected tests, OpenMP1/2 witnesses and generated Cartesian/composite syntax checks passed; arbitrary public Python path-model admission is not claimed |
| Generated-model contracts | Extract `ConservationLaw`, `StateConversion` and `NoSource` from headers containing built-in physics, preserving their bodies and legacy APIs | 38 affected source tests and seven syntax checks passed; actual Euler, Fan-Li15 and source/auxiliary/RHS emissions contain none of the five targeted built-in Euler/force definitions |
| CI inventory and scheduling | Account for every native/Python test, prebuild the six loader fixtures, split the complete MPI build into two bounded phases | 199 native targets, 13 C++ shards, 38 Python shards and all 121 MPI launches retained; revision `7d701d2` passes 120 MPI launches, with the remaining collective-exception test expectation repaired and locally checked at MPI1/2; final GitHub CI remains unverified |
| MPI Python inventory execution | Read both plans on a separate descriptor, close child input, and require completed counts to match the manifest | The apparently successful `19b2bb5` MPI job ran all 121 CTests and M4 but only one of nine Python entrypoints. On `8b05af9`, all 121 native launches and the first four Python scripts pass; the fifth correctly fails the job. All nine entrypoints and the final required aggregate still need to pass on the final SHA |
| Roe test process budget | Give the coherent two-model compilation/trajectory witness an explicit 450-second limit instead of the default 300 seconds | The unchanged 24 checks pass locally; the actual `8b05af9` Linux CI witness passes in 282.32 seconds. The 380-second scheduling weight remains an estimate, and shard/workflow limits and scientific controls are unchanged |

## Actual results

These are retained run results on the stated revisions, not a claim that the final
PR revision has passed GitHub CI.

The exact `8b05af9` run `35727344990` finished with 90 successful jobs, the MPI
failure and its derived aggregate failure. The old 8² shared-interface fixture was
already fully refined, so its expected hierarchy change was impossible. At 16² it
retains a real coarse interior and exposes a separate native refusal to transfer
shared history across levels. The generic correction preserves parent-face flux
integrals, source associations, duplicate refusal and rollback; it does not discard
history or relax conservation checks. The actual rebuilt `52b3e26` MPI2 run reaches
the shared hierarchy transform, injected rollback and successful retry, then fails
on the first continued step. The generated Program retains one history reference
across a second history read; replacing the complete scratch map invalidated that
reference. Deferred reads now preserve existing object addresses and publish only
the prepared value after collective agreement. A dedicated regression covers two
histories, rank-local preparation refusal, retry and repeated reads; it is included
in the existing MPI test filter. New source/native qualification is recorded
separately from these retained CI results.

- Euler `65b0658`, ROMEO job `714111`: modes 3, 4 and 5 each reached `t=0.1`
  on base `16x64`, two AMR levels, MPI2/OpenMP2, with 110 accepted steps,
  zero rejections and 12 saved states. Maximum relative mass drift was at most
  `7.55e-13`. Expanded interior restart from step 10 to step 33 also passed.
- The same allocation failed its subsequent mode-5 `32x128`, three-level,
  MPI4/OpenMP2 pilot before any accepted positive-time state. The coarse solve
  exhausted 512 iterations: true/original-stencil infinity residual
  `2.5020132470362846e-12`, required tolerance `2.2716170886597616e-12`.
  This is a valid refusal. A new local MPI4/OpenMP2 capture after rebuilding
  `f57f5dd` reproduces the failure at FAC outer ordinal 2, with a true residual
  of `2.545564254221145e-12` and 512 one-column restart cycles. Corrections of
  approximately half an ULP disappear when added to the same coefficient.
  Moving that coefficient one representable value downward yields a true
  residual of `2.1814437913689897e-12`, below the unchanged tolerance. This is
  an exact-input diagnostic witness. The generic single-column GMRES recovery
  now reaches this accepted residual in three iterations on the MPI4 capture.
  Eight relevant tests pass at MPI1/2/4, including a coupled counterexample
  that correctly refuses convergence, unchanged iteration caps, extreme
  preconditioner scaling, and workspace reuse. The actual rebuilt `7d701d2`
  pilot passes that input but fails later at FAC outer ordinal 5, before step 1:
  true residual `3.1455104848800301e-13`, required `2.6428759556514764e-13`.
  Its one-column corrections alternate between adjacent candidates. A generic
  follow-up halves the next complete single-column correction after a rejected
  non-decreasing true infinity residual. Both exact MPI4 captures then pass in
  three and six iterations; nine relevant tests pass at MPI1/2/4. The new
  model-independent two-equation regression fails with the previous headers.
  The change preserves the true-residual acceptance check, iteration caps,
  workspace size and L2/multicolumn paths. The rebuilt `6dc0bcb` pilot accepts
  its first authored normalization step at `t=1e-10`, then fails at a later
  FAC outer ordinal 3: true residual `1.274676468585329e-12`, required
  `1.0529430645742341e-12`, after 512 iterations. This first small interval is
  intentional and does not establish a CFL collapse.
  A subsequent generic correction detects an actual adjacent-value promotion
  followed by rejected true-residual non-descent. It then applies complete
  one-column corrections only at the current global residual maxima, including
  all exact ties. Persistence avoids a demonstrated coupled three-equation
  rounding cycle; normal multicolumn updates leave this recovery mode.
  All three captured systems pass in 3/4/10 iterations, and 11 affected tests
  pass at MPI1/2/4, including asymmetric ownership, cap/reset checks, extreme
  scaling and terminal provider failure after a coordinated update.
  Each Linf one-column update now requires a local Kokkos reduction per
  fab/component; possible activation also requires one MPI maximum on the
  prepared lane. No prepared field, extra operator application, tolerance or
  iteration allowance is added. Residual maxima do not generally identify the
  responsible unknown, so this policy does not guarantee convergence for every
  coupled operator. The original true-residual guard remains authoritative.
  The actual rebuilt `19b2bb5` pilot accepts five steps, zero rejections, through
  `t=0.00400000006`, then refuses another coarse solve at the same 512 cap:
  true residual `5.656354536045722e-13`, required `5.0993206885413197e-13`.
  Its promotions occur in descending cycles; an ordinary update increases the
  residual on the following cycle without a new promotion. Requiring both events
  in the same cycle therefore misses the trigger. A local boolean now remembers
  actual promotions across that rejected single-column episode; the existing MPI
  reduction combines histories at the next non-descent. Multiple columns and new
  invocations clear the history. No field, matvec or reduction site is added.
  All four captured systems pass in 3/4/10/10 iterations; twelve targeted tests
  pass at MPI1/2/4, including a causal dyadic regression that fails on `19b2bb5`
  and a distributed companion whose residual maximum belongs to another rank.
  An independent SPD counterexample retains the stated limitation of selecting
  unknowns from equation maxima. This repair does not guarantee convergence for
  every operator. After official reconstruction, the actual `97279de` local
  MPI4/OpenMP2 pilot reaches `t=0.01` on `32x128`, three AMR levels, in eleven
  accepted steps with zero rejections. It takes 578.90 seconds including binding;
  the final density and accepted potential history match the checkpoint bit for
  bit on all three levels. Density is at `t=0.01`, while the potential belongs to
  the midpoint `t=0.009500000005`. All stored state components are finite; final
  mass is `79.16885115358725`, compared with analytic mass `79.16885115358781`.
  The checkpoint manifest and payload digests are authenticated, and the actual
  process tree is closed. This qualifies the local L3 endpoint only; runtime
  restart, current-source Linux and long-campaign qualification remain separate.
- Fan-Li15 `65b0658`, local MPI2/OpenMP2, `16x64`, two AMR levels: cold and
  genuine expanded restart reached `t=0.03`, 33 accepted steps, zero rejections.
  The fine level grew from 2048 to 2560 cells; full stored state/history endpoint
  parity passed. Linux Fan-Li validation and the long trajectory remain pending.
- Four fresh local continuation gates on `8b05af9` are accepted: 38 native cases
  in 74 MPI1/2 worlds, separate short Euler and full Fan-Li15 cold/interior restart
  chains, and an expanded Fan-Li15 continuation. The expanded cold trajectory
  reaches `t=0.03` in 33 accepted steps with no rejection. The actual first
  expanded interior checkpoint is C10 at `t=0.00900000001`, with fine coverage
  increased from 2048 to 2560 cells. Its genuine restart takes 23 accepted steps
  to C33 with no rejection. Immediate restoration matches all 65 carriers; C11,
  C22 and C33 each retain 63 byte-identical carriers with only two separately
  authenticated execution-lineage fields differing. All three output snapshots
  and the continued calendar agree. The cold/resumed archives pass respectively
  41,472 and 28,672 stored H1 checks. The original relocated-auditor import failure
  is retained; a reviewed dependency-only repair reruns the audit, not the physics.
  These physical cases use one `plasma` block and no shared-interface history.
  Their actual source remains `8b05af9`; no later binary or metadata equivalence
  is implied by reusing their results after an affected-path review.
- Homogeneous HYQMOM15/BGK `65b0658`: actual Knudsen values `0.01`, `0.1`, `1`;
  the `Kn=0.1` temporal study measured orders `2.028` and `2.014` against the
  relaxation solution. Only initial/final states were saved for those runs.
- Spatial HYQMOM15/BGK `8ae9097`: `64x4`, `Kn=0.1`, MPI1/OpenMP2, HLL,
  first-order forward Euler, outflow in x and periodic y. It reached `t=0.01`
  in 200 accepted steps with zero rejections, saved 201 full 15-moment states,
  and wrote an authenticated 16,428-byte final checkpoint. Independent checks
  covered 51,456 exact H1 cell instances and an open-boundary invariant-balance
  residual at most `4.04e-16`. This is not full spatial restart or general
  hyperbolicity qualification.

The Maxwellian audit agrees that all source-free isotropic Maxwellian
characteristics are real. The obstruction concerns other realizable states in
oblique directions: a correlated Gaussian gives imaginary parts of magnitude
`0.06488662033` for a unit normal, stable under high-precision evaluation.
See [the exact audit and limitation](HYQMOM15_LIMITATION.md). No imaginary-part
projection or modified closure has been used to manufacture a trajectory.

## Required continuation

1. Rebuild the final source with the repository's official incremental workflow.
   The gather repair is qualified on `4adfddf`; the generic core and diagnostic
   changes were rebuilt at `f57f5dd`. Each later production change requires a
   new authenticated native artifact before scientific validation.
2. Repeat the now-passing local larger-grid case on the rebuilt Linux source.
   The `97279de` local pilot reaches `t=0.01`; the 512/64 coarse controls and the
   original-stencil residual remain unchanged. Optional capture/trace instrumentation is
   integrated and has passed on/off parity, exact replay and collective refusal
   checks at MPI1/2/4. Longer restart cycles and persistent compensated updates
   were tested on the exact captured input and did not solve the defect.
3. Run the relevant local tests and current-revision ROMEO low/restart/larger-grid checks. Do not carry
   acceptance across an unverified source or native-artifact change.
   For the shared-history correction, rebuild both the native runtime and generated
   Program libraries, then check projection, archive association, ledger identities,
   strict restart and distributed rollback. Preserve the four actual `8b05af9`
   gate records and review unaffected paths explicitly instead of relabelling them.
4. Complete the separate Linux Fan-Li check, then approximately two hours of
   measured validation and time/storage estimation. **Zero designated two-hour
   validation allocations and zero of the two high-resolution allocations have
   been used.**
5. Launch exactly two total roughly 23-hour higher-resolution allocations shared
   across Euler modes 3, 4 and 5, with separate genuine checkpoint chains.
   Refresh jobs, quotas, resource limits and all source/runtime identities first.
   Never duplicate an uncertain submission or mutate inputs of an active job.
6. Retrieve and inspect actual states through the required times, then complete
   figures 5.1--5.4, the schlieren sequence and GIFs. Current Euler images cover
   only `t<=0.1`; no growth-fit window has been reached.

The local task's `output/STATUS.md`, `output/FIGURES.md` and
`output/HANDOFF_ASTRA.md` contain the full job ledger, exact evidence hashes,
artifact paths and deployment instructions. Raw outputs, private deployment
metadata and rejected diagnostic prototypes are not shipped as source changes
in this draft PR.
