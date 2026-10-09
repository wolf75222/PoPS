# Independent AMR mathematical/source receipt — SOURCE_ONLY

Source freeze: `cd0ebd61` (including the generic tag-selection correction). Detached private review checkout. No production/physical code changed; no PoPS import, native build, JIT, environment mutation, numerical guard adjustment, or native qualification. This receipt does not review the independent tag-selection patch or its tests. Historical reader @1 remains archived; reader @2 constraints were reviewed with externally authenticated files and CP12. ROOT subsequently identified its legacy accepted-contract schema-7 dependency; current production schema 8 requires the forthcoming strict reader @3 before real reception. This note does not certify @2 against schema 8.

## Authority and scope

Principles 1.1–1.8 were actually read from the primary handoff. Equations and physical bodies remain authority (1.1); the witness is a linear physical script composed through existing public mechanisms (1.2, 1.4–1.6). The reasoning uses mathematical effects and storage/coverage contracts rather than field-name recipes (1.3, 1.8). No cost claim is made without measurement (1.7).

Files inspected: `tests/python/support/evolved_stage_amr.py`, `evolved_stage_mms.py`, `captured_diffusion_mms.py`; `python/pops/time/evolved_field_stage.py`; `python/pops/codegen/program_emit_evolved_field.py`; `include/pops/runtime/program/prepared_amr_field_residual.hpp`; `include/pops/numerics/elliptic/nd/prepared_composite_general_field.hpp`; active-coverage preparation and field-mask accessors in `src/runtime/amr/amr_system.cpp`; the history checkpoint context mask accessors; reader @2 at private freeze `bcf304c` (independently cross-checked with actual CPP hash/codec in review freeze `2b00b43`).

## Exact expectations for this homogeneous witness

Let a=T0 and b=T1. One component: Q=a+a². Two components: Q0=a+a²+0.1b², Q1=b+b²+0.2ab. The authored stage residual is Q(Tnext)-Qprevious-dt*(div(D grad Tnext)+forcing), not a residual divided by dt. dt=0.01. D=[[0.012,0.002],[-0.001,0.014]] is signed and nonsymmetric. For homogeneous T, its diffusion action is zero.

One component: initial a=0.15 and Q=0.1725; first accepted a=0.16, Q=0.18560000000000001, forcing=1.3100000000000027. Step two Q=0.19870000000000004 and the positive root is a=0.16985072964056705. Stable scalar inversion is 2Q/(1+sqrt(1+4Q)).

Two components: initial (a,b)=(0.15,0.25), Q=(0.17875,0.32), total=0.49875. First accepted (a,b)=(0.16,0.24135127657485897), Q=(0.1914250438704314,0.3073249561295685); forcing=(1.2675043870431417,-1.2675043870431417). Step two Q=(0.20410008774086283,0.29464991225913717), positive solution (a,b)=(0.16984205688727266,0.23263074959881333). Algebraic state z=0.25a+0.5b at each step. Load is fixed from the first target, so Tstep2 must not be extrapolated by a constant T increment.

The accumulation Jacobian is [[1+2a,0.2b],[0.2b,1+2b+0.2a]], with step-two determinant 2.0063298114667125. It is positive definite in this local positive witness region; this is not a global invertibility claim and does not make D symmetric. A bounded positive-root oracle uses conservation to solve b from a and then scalar bisection for Q0.

## Composite quotient and storage distinctions

For N² complete coarse cells and k coarse cells covered by aligned fine patches, active coarse count=N²-k, fine count=4k, composite count=N²+3k. Exact measure is (N²-k)/N²+4k/(2N)²=1. Scalar unknown width is 1; the coupled solve width is 3 including z, so active basis size is width*(N²+3k). Owners must produce each distributed active degree of freedom exactly once; replicated levels contribute only on rank zero to composite reductions. Keep the fixed 256 MiB basis budget; topology-derived size and the existing full allocation formula identify a budget failure, without changing the budget or the grid.

Native original-field dot products use active coverage and physical cell measure, excluding duplicated replicated contributions. Reader @2 independently derives coarse=base minus coarsened fine coverage and fine=present, demands genuine partial refinement, aligned/nonoverlapping boxes, complete base, owner authority, and exact Fraction volume one. A zero-filled absent fine cell is not a scientific cell.

The projection emitter's pointwise_active_mask is the embedded-boundary activity mask. Finest-level coverage is a separate mask/accessor. With no EB, projection writes all valid coarse cells, including covered cells, from the synchronized candidate. Do not infer that covered Q remains previous. Full ghost carriers and checkpoint restoration are separate from active scientific amounts. In a nonconstant state, restricting nonlinear Q and evaluating Q on restricted T need not commute; this homogeneous witness cannot expose that broader issue.

## Falsifiable risks and minimum real failure diagnostic

1. Wrong coverage/ownership or row-to-owner capture pairing: report actual boxes/owners, independently derived masks, active counts/volume, each active Q amount, and the largest original residual/projection/constraint error with level, component and coordinate. Width-two rows are deliberately reversed while unknown and owner ordering is unchanged. Check existing IR/capture authority rather than introducing name-dependent remedies.
2. Candidate/ghost synchronization or coarse/fine operator mismatch: report per-level T minima/maxima, cross-level discrepancy and interface adjacency of error maxima. Archive full native carrier hashes separately from valid-cell scientific arrays. Constant fields must remain in the zero-diffusion kernel, but a PASS does not qualify nonzero interface flux or D versus its transpose. The uniform periodic np.roll MMS oracle must not be transplanted to partial AMR.
3. Solver diagnostic contract: the final original-F recheck accepts norm <= tolerance*max(1,reference_residual_norm), while the published rel_residual is norm/reference (reference positive). Reader @2 requires rel_residual <= tolerance. Reference below one can therefore distinguish these two criteria. Preserve actual norm, reference and relative norm; an apparent diagnostic failure must not be repaired by relaxing the reader or inventing a residual scale. This is a source risk, not an observed native failure.
4. Nonlinear branch: require the positive witness branch plus independently computed Q and constraint. Do not accept cancellation of global amount as evidence of pointwise correctness. The actual numerical acceptance remains 3e-8 and native tolerance remains 1e-10.

Smallest next diagnostic: the first failing real representative case, one rank and its unchanged actual grid, exact native identity, original traceback/phase, saved initial/accepted/continuous files plus CP12/full carrier blob and external seals, the counts/errors above and native norm/reference/rel trio. If first-step science passes, step two uses the computed nonlinear target above; restart/replay requires exact carrier equality independently of these tolerance-based scientific checks. No synthetic file may stand in for ROOT scientific reception.

## Principle → decision → file → oracle → command → status

| Principle | Decision / file | Oracle / command | Status |
|---|---|---|---|
| 1.1, 1.3, 1.8 | Keep authored Q, load, constraint; support/evolved_stage_amr.py | `python3 tests/review/sol61_amr_math_oracle.py` | Pure independent arithmetic PASS; no runtime claim |
| 1.2, 1.4, 1.5 | Inspect public stage and generated projection, no wrapper or method branch | Source ranges named above | SOURCE_ONLY |
| 1.6 | Separate active quotient, full carriers and externally sealed reader @2 | Actual CPP hash cross-reader tests in freeze 2b00b43 | Prior pure source/host PASS; current native receipt belongs to ROOT |
| 1.7 | Report actual topology/basis size; retain budget | Actual counts plus existing required_bytes formula | Native cost not measured here |
| 1.8 | Homogeneous zero-kernel cannot qualify nonzero AMR flux | Minimum real diagnostic above | Limit explicit; no expanded physics slice |
