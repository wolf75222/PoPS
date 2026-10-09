# SDK8 nonconstant N8 reception - 2026-10-02

The [sealed receipt](sdk3d8481_spatial_n8_scientific_reception.json) receives one
nonconstant N8, width2 periodic full-y AMR strip on each of Serial and MPI2.
The actual installed PoPS package has NativeABI8, ProgramIR16, CP12 and the
default accepted-state8 contract. Source/build `3d8481c9`, native DSO
`7fd8c7fe`, all 1,136 shipped sources and before/after identities are pinned.
Serial uses the MPI-enabled CPU Kokkos/OpenMP SDK; it is not a separate
non-MPI build. MPI2 has one case on each rank, not two distinct cases.

The independent reader recomputes composite Q, original field residuals,
coarse/fine active masks and volumes, nonlinear restriction, periodic fluxes,
physical clocks, persisted field histories, grown carriers and exact
restart/replay from saved images. Accepted Original-F weighted L2 is
`4.211236792117065e-11` on Serial and `4.211237330332404e-11` on MPI2.
Both satisfy the unchanged independent `1e-10` threshold. Physical Q keeps
its `3e-8` guard.

The raw native relative residual remains approximately `1.1598995e-10`.
The actual selected legacy rule tests `norm <= 1e-10 * max(1, reference)`;
the saved raw quotient is preserved separately. The reference norm is
`0.3630691185175441`. The norm comparison uses authenticated IR operations
and the `4*dimension/h_min²` sensitivity bound. No equation, initial data,
Newton control, physical guard or empirical rounding margin was changed.

The independent read-only audit (`/Users/romaindespoulain/dev/tmp/sol61-n8-independent-20261002/README.md`)
replays both sealed receptions exactly, rehashes 50/51 pins and every shipped
source, and refuses ten copied, synthetic SourceOnly adversaries. These are
reader adversaries, not native failure injections. The earlier SDKbb N8 run
remains FAILED. Fresh native runs and fresh ROOT seals were used here.

Reproduction uses the command arrays in the pinned native run receipts. The
scientific reception command is the existing reader, with all four sealed
arguments from the pinned owner and approval files:

```text
env -u PYTHONPATH <python> tests/review/sol61_evolved_stage_amr_spatial_reception_v3.py receive \
  --pins <root-owner3.json> --pins-sha256 <owner digest> \
  --approval <root-approval3.json> --approval-sha256 <approval digest>
```

The receipt pins the ROOT preparation script. Its first preparation had a
malformed JUnit row; a separate attempt2 corrected the inventory without
rewriting the native captures or the first owner files.

This receives the declared constant signed-D, Cartesian strip. Arbitrary 2D
AMR, candidate-dependent D, EB, Field/GhostBC, opt-in accepted-state9, other
dimensions, ROMEO/GPU and the full mission remain open. Opaque program and
auxiliary bytes are compared for integrity/replay; no reconstructed
C++-to-DSO cryptographic build graph is claimed. Published N8 profile @3
remains intact; N16 uses the separately versioned @4 profile.
