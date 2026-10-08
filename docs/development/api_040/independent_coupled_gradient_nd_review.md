# Independent review - Cartesian CoupledGradient in Dim2/Dim3

Reviewed author commits `08ebcc95`, `6176eb14`, `221dd9af73974201b7d8493b9d8939db656b58c4`, relative to `9ea0342`. Review edits contain tests, an independent NumPy oracle and this receipt only. Production files belong to the author. Verdict: no blocking source defect found in the bounded periodic constant-matrix extension; installed numerical reception is still required.

## Mathematical and native mechanism

The physical matrices act on components: K=D+R, D symmetric positive semidefinite, R skew. The spatial tensor remains the identity. Anisotropy in this review means distinct Cartesian spacings, not an invented direction-dependent constitutive law. The existing `PreparedDiffusion<Dim,N,true,true>` loops cover each Cartesian direction. Its centered-gradient plus second-difference face reconstruction cancels algebraically to K(U_right−U_left)/h_axis for this constant law. The divergence sums directions, and the accepted ledger computes the product of tangential spacings and outward signs −1/+1. No change to the equations, component packing or physical matrices is introduced.

The independent oracle uses exact Fourier cell averages (product of sinc factors) and the symbol
`lambda_h = -sum_axis 4*sin(pi*m_axis/N_axis)^2 / h_axis^2`.
Its rank-one three-component D and noncommuting R cover component permutation, an oblique wave, a direction with zero wave number, the constant mode, ker(D), and the checkerboard mode. Direct finite differences are compared to the Fourier expression. The energy identity is checked independently by face jumps:
`<U,L_h U>_V = -sum_faces V/h_axis^2 * jump^T D jump`.
R contributes zero to this quadratic form while it can still evolve a mode in ker(D). A second oracle sums all oriented face contributions with their actual areas and both SSPRK2 weights; unit areas or a single reused spacing are discriminating wrong alternatives. These NumPy checks validate the oracle mathematics, not native numerical execution.

### Frequency correction also changes Dim1

The production factor changes from 2.5 to 4 for **all Coupled dimensions**, including the already supported Dim1. This is a correction of the reported magnitude bound, not merely removal of a dimension admission check. A concrete counterexample is D=0, R=[[0,−.3],[.3,0]], and a checkerboard on a four-cell periodic mesh in each active direction, with the second component initially zero. Direct application of the discrete operator gives
`||L_h U||_infinity / ||U||_infinity = 4*.3*sum_axis h_axis^-2`.
The previous `2.5*.3*sum h^-2` is smaller by a factor 1.6, already in Dim1. Independent tests verify this in dimensions 1, 2 and 3 with distinct spacings. The new quantity bounds the absolute operator norm; it supplies no monotonicity certificate. In particular, SSPRK2 on i*z has squared amplification `1+z^4/4 > 1` for nonzero real z. No general Hall stability claim follows from the bound or the finite trajectories.

### Identity and stale providers

The method retains schema 1 and `periodic_two_point_component_matrix_v1`: the representation and constant face stencil remain unchanged. This does **not** authorize reusing an old provider. `prepared_diffusion.hpp` is covered by `include/pops_headers.manifest`. The independent source test copies the author's include tree and restores only that header from `9ea0342`, then runs the actual `pops_header_signature` function:

- Previous signature: `62398f3c13c193eb48db07518735fe755e6d108821adffc37acccd8fcea290eb`.
- Reviewed signature: `1ca25b8fe94ee8ccbb4980dfe9bba1e004675d6c65f156bd98b973cfff334ebf`.

With the same physical law and schema metadata, the real `artifact_spec_identity` function yields distinct specifications when only this ABI signature changes. Source inspection confirms `_compile_drivers.py` passes the current header signature in `abi_key` to `program_artifact_spec`, and `abi.check_compiled_matches_module` compares the cached artifact signature to the signature baked into the native module before dlopen (when available). This is a source/identity-seam proof; no stale DSO was loaded and no new installed SDK is qualified here. The installed runtime and generated providers must be rebuilt together.

## Executed checks and fixture review

- **13/13 NumPy oracle tests** pass, including the three old-frequency counterexamples.
- **1/1 SDK/artifact identity test** passes.
- **2/2 complete Program translation units** pass `/usr/bin/clang++ -std=c++20 -fsyntax-only -fno-fast-math` with real Kokkos/OpenMP/MPI headers, `POPS_NATIVE_DIM=2` or 3, and the author's own PoPS headers. They contain the public SSPRK2 Program, three components, repeated weights .5/1.5, scratch/status/provider wiring and accepted ledgers. The Dim3 case permutes components. They are not extracted kernel fragments.
- The public worker authenticates its source import and also checks rejection of a conormal boundary on the last axis and rejection by the actual AMR emitter. On baseline `9ea0342`, the Dim2/Dim3 positive authoring probes fail at the old `one periodic axis` admission, before compilation.

The author's Dim2 native fixture converges construction, allocations, bind, run, state gathering and root assertions. Its ledger checks use the explicit SSPRK2 StagePoint identity rather than insertion order. The review requested genuine Fourier cell averages, one operation identity, replay metadata in NPZ and finite JSON serialization; all are present in `221dd9af`. The fixture retains oriented measures, occurrence weights, fluxes and cell increments. Its saved states and ledgers have not been produced by this reviewer. The chosen single-box MPI route can have an empty peer; distributed multi-patch numerical coverage is a separate obligation.

Reproduce the review after selecting the author's source checkout:

```bash
env -u PYTHONPATH POPS_ND_REVIEW_SOURCE_ROOT=/path/to/reviewed/PoPS \
  python -m pytest -q -o pythonpath= \
  tests/python/unit/numerics/test_coupled_gradient_nd_independent.py \
  tests/python/unit/codegen/test_coupled_gradient_nd_review.py
```

The exact syntax commands, XML and source/header hashes are recorded in `evidence/coupled-gradient-nd-review/receipt.json` and neighboring small receipts. Nonperiodic traces, AMR composite transfer, variable matrices, implicit spatial stages and transport composition remain explicit implementation gaps. Dim3 here has source/complete-TU syntax and mathematical oracle evidence only. No package installation, JIT, linked native run, GPU or HPC test was performed.
