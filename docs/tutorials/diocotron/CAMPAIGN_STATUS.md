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
| Mapped field coupling | Full tensor coarse GMRES, polar Poisson preconditioning, original physical infinity residual checks | Local operator/manufactured tests and the low ROMEO runs passed; the newly repaired larger-grid solve still needs a rebuilt full trajectory |
| FAC hierarchy | Synchronize covered parents after corrections and after the periodic gauge; conservative fine-interface flux support | Original FAC tolerances pass; covered-parent regression checks pass at `1e-12`; not a general high-resolution convergence guarantee |
| MPI execution | Serialize public model artifact publication; synchronize AMR body failures before flux publication | Focused MPI/runtime witnesses; not all backend or multi-node coverage |
| AMR continuation | Preserve accepted-attempt authority, histories and auxiliary images; rebuild generated resources after accepted history remap | Genuine expanded cold/restart comparisons on revision `65b0658` |
| Native packaging | Authenticate relocated installed extension bytes | Official build/doctor and retained source/installed manifests |
| Uniform checkpoint shape | Distinguish native reversed array shape from legacy field-free logical shape, preserving byte guards | Revision `8ae9097`: 36 focused Python checks; actual one-rank fresh-process rectangular restart; final Knudsen checkpoint |
| Exact global gather | Replace arithmetic summation with byte assembly after exact ownership checks, preserving signed zero | Official Dim2/MPI rebuild at `4adfddf`; 36 Python checks, four genuine MPI1/MPI2 capture/restart worlds and six MPI1/2/4 gather CTests passed, including signed zero and refusal before mutation |
| Generic C++ core | Generate Fan-Li physics in Python over variable-count/dimension-generic path-flux and moment primitives; remove the three model-specific C++ headers | 49 affected tests, OpenMP1/2 witnesses and generated Cartesian/composite syntax checks passed; arbitrary public Python path-model admission is not claimed |
| CI inventory and scheduling | Account for every native/Python test, prebuild the six loader fixtures, split the complete MPI build into two bounded phases | 199 native targets, 13 C++ shards, 38 Python shards and all 121 MPI launches retained; local planner/fixture checks passed, final GitHub CI remains unverified |

## Actual results

These are retained run results on the stated revisions, not a claim that the final
PR revision has passed GitHub CI.

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
  preconditioner scaling, and workspace reuse. A rebuilt full trajectory is
  still pending.
- Fan-Li15 `65b0658`, local MPI2/OpenMP2, `16x64`, two AMR levels: cold and
  genuine expanded restart reached `t=0.03`, 33 accepted steps, zero rejections.
  The fine level grew from 2048 to 2560 cells; full stored state/history endpoint
  parity passed. Linux Fan-Li validation and the long trajectory remain pending.
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
2. Run the real larger-grid case with the qualified generic recovery for
   captured representability stagnation. The 512/64 coarse controls and the
   original-stencil residual remain unchanged. Optional capture/trace instrumentation is
   integrated and has passed on/off parity, exact replay and collective refusal
   checks at MPI1/2/4. Longer restart cycles and persistent compensated updates
   were tested on the exact captured input and did not solve the defect.
3. Run the relevant local tests and current-revision ROMEO low/restart/larger-grid checks. Do not carry
   acceptance across an unverified source or native-artifact change.
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
