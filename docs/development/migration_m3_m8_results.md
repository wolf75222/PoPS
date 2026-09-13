# PoPS migration results — representative qualification

**2026-09-13, v34 review.** M0–M2 are preserved. The current frozen source is
175b0883f9f0752f5710ef3db2c7d7d6802e400a, with 2,495 tracked files and freeze SHA256
0b23660d2f1cfedad3c85bb3ff985362b5bab2ab4b38f2439b2e99771575ff8e. At this source the
representative M3–M8 contract is complete. The [evidence index](migration_evidence/representative_20260913/index.json)
binds issues to results, source/package identity, numerical properties, and reuse decisions.
The [M0–M2 record](migration_m0_m2_results.md) remains unchanged.

The representative-validation policy replaces repeated full matrices. Unaffected successful
receipts may be reused only with a source-impact decision; changed or failed routes receive a
targeted witness. Native execution, numerical qualification, and performance remain distinct
claims. Failed original receipts remain failed when a corrected successor passes.

## Implementation and evidence

| Phase | Result and representative evidence |
| --- | --- |
| M3 — fields | Independent storage, qualified stage observations, joint solve publication, and common normalization are retained. The public three-field AMR test recorded at v29 passes its N16/32/64 solution and gradient oracles. A separate joint-tuple rollback and same-instance retry test passes. The v29 bridge set also retains split-failure rollback and native/Python law comparison; historical native evidence covers signed/noncommon-kernel modes and normalization. |
| M4 — interactions and native calls | Typed authenticated calls and heterogeneous outputs preserve inventories, derivative contracts, and failure disposition. Six native solve/failure cases pass on each of two MPI ranks, covering exact, approximate, and finite-difference derivatives. Imported/native and Python laws match on two profiles. Stable Darwin linker identities preserve complete binary reproducibility for native components and generated models; the corrected model defect affected 49 metadata bytes outside code/data sections. |
| M5 — transport–diffusion | Public rotated tensors retain the N16/32/64 sequence with observed orders 1.959 and 1.990 and inventory defect below 1e-17. The two-component implicit tensor AMR solve passes actual-interface conservation, residual, and refinement checks without a second reflux. Accumulation, solved-potential drift, nonidentity bridge, and halo-factory witnesses pass. |
| M6 — solves, transactions, and time | Prepared resources and nested rollback preserve accepted state, fields, histories, and exchanges; failed native iterates cannot publish recipients. The v33 oversized-step rollback witness is reused because its guard is unchanged. IMEX ARS222 retains its 8/16/32/64-step order sequence. Inline and imported scalar AMR methods each complete 128 accepted steps, 129 regrids, and a bit-identical restart continuation. |
| R1 — public workflows | All eight requested workflows execute: scalar AMR advection, field-coupled transport, Euler–Poisson source, heterogeneous interaction, explicit diffusion, implicit diffusion, variable-coefficient scalar field, and imported native primitive. Imported scalar AMR supplies an additional method witness. Euler–Poisson uses its declared isothermal source partition. One-step examples rely on separately indexed numerical oracles and do not establish convergence alone. |
| M7 — coupled supports and AMR | Generic physical reductions, pullbacks, and transfers retain signed weights and component/axis layouts. The prior MPI4/OpenMP2 transfer witness passes 4 ranks × 7 tests with two threads per rank. Two v31 field/map restart variants pass on each of two ranks, alongside retained rollback evidence. Fifteen actual old/new uniform-map arrays are byte-identical. |
| M8 — equivalence and retention | Gaussian uniform/AMR comparisons preserve exact integrals, rebinding, regridding, and restart; both signed far tails pass a strictly positive erfc oracle. A three-axis AMR mixture passes its weights and restart path after a checkpoint-name fixture correction; the original analytic3 failure remains historical. The reviewed semantic deletion set is empty because active consumers remain. |

The Python FV ledger retains exact accepted weights and ancestry limited to the conserved previous state. Native group capture performs one evaluation per block and supports either a physical or periodic closure, with atomic face/residual publication; Uniform and AMR group-capture r2 each pass one test per rank, and build-r3 passes. Both explicit and implicit composition cases pass on each MPI2 rank: four observations, zero skips, 69.254648 s. Each maximum update defect is 8.881784197001252e-16 against an independent Fourier oracle; 4,096 distinct signed records reconcile separate FV and diffusion physical amounts, accepted state, multiplicity one, occurrence IDs, contexts, measure, and temporal weights. The implicit residual is at most 1e-12 max(1, reference).

The v34 Dim2 wheel, installation, byte proof, and doctor checks pass for source
175b0883f9f0752f5710ef3db2c7d7d6802e400a. The source review and independent Astra review accept
the corrected group API. Explicit tensor AMR remains refused because no composite stability bound
is proved. Shared-interface, embedded-boundary, and unsupported-legacy group-capture requests
retain explicit refusals; no-capture consumers remain available.
Unsupported temporal affine/history reconstruction, asynchronous field policies, and arbitrary
coupled-system fitted fluxes retain their indexed diagnostics.

This record combines source-specific results, not one new full-suite run. The retained native
gate at `13dc8bba797b42f359267422a19501af4bbb5e73` has 1,212 passes and 21 conditional skips.
Public workflow/field bridges use `77c1eb300b9a35ed018d0a31c7f7efa0a793c014`; the earlier
Dim1/2/3 packages use `3eb2722e4618d25f1586632b592ee20b4dc505fa`. Their source-impact reuse
and later fixture-only changes are recorded separately in the index.

The representative contract is complete for this source and scope. Historical PMPI evidence is
one instrumented two-rank lifecycle: 705 calls per rank, 78,960 collective operand/contribution
bytes, 448 send-argument bytes, 448 posted receive-capacity bytes, and process peak RSS of
24,477,696 and 22,773,760 bytes. These are logical API quantities and process-lifetime memory,
not network traffic or solver-only costs. No current speedup is claimed. CPU/OpenMP/MPI evidence
does not qualify GPU execution or cluster scaling, and local checks do not establish GitHub CI;
PR CI remains unverified.
