# Independent reception of c100 AMR Newton repair — 30 September 2026

Candidate `c1009a821dca080bf1bbf8aa52641e9aa6ea247d`, parent
`47a7b2aa`: received in a new private checkout, then combined with the earlier
independent diagnosis `018b0332c6701f0d7ce68474e0b50900afefac45`. Diff was read
before the author's justifications. This follow-up changes only independent
tests, probes and receipts; no production file is repeated or modified.

## Corrected seam

The production header is byte-identical to its parent after removing exactly
the seven added lines: a comment plus `local_phase_` that scales every level
of the workspace defect by -1 after evaluating the complete physical F.
The local phase fences and converges errors before the workspace reductions.
No component-zero or level-zero specialization is introduced.

The entire generic AMR Newton workspace, Uniform original residual adapter,
composite general field provider and scalar AMR operator are byte-identical to
the parent. Thus the repair preserves the RHS convention, additive update,
Krylov algebra, ownership, masks, measures, halo/restriction/reflux mechanisms
and established Uniform route. In the edited header, the full physical evaluate,
positive central JVP, independent physical recheck, attempt authority, captured
storage and publication logic remain unchanged outside the seven-line addition.
`sol61_amr_newton_fix_byte_reception_20260930.json` records these exact checks.

The independent mathematical witness from 018 continues to distinguish the
old uphill convention from the corrected direction, with the original reaction,
cubic equation, tolerance and iteration budgets. The source probe now reports
`sign_convention_compatible` on c100. It does not assert that every later source
still contains the historical defect. The old receipt
`sol61_amr_original_newton_source_math_20260930.json` remains historical and
unchanged; the new result is in
`sol61_amr_original_newton_fix_source_math_20260930.json`.

Six explicit negative source-string copies were refused: remove defect
negation, negate physical evaluation, reverse the JVP, negate the JVP after
evaluation, negate the original physical recheck, and negate only the first
level. These copies never modify a header on disk. A wrong sign yields the
same independent uphill counterexample; mutations of physical/JVP/recheck
separation are rejected before a source-compatible report can be emitted.

## Serial/MPI inventory

The original diagnosis's unit inventory probe now treats six included tests as
a historical minimum; adding a native regression cannot falsify that old
observation. A separate exact c100 check requires **all ten historical names
plus the new identity-residual name**, exactly eleven cases, and verifies the
fragment is actually included in the compiled `.cpp`.

The actual CMake target declares the fragment as DISCOVERY_SOURCES and passes
it to `gtest_add_tests` without adding it to compiler sources. The independent
CMake scanner runs with `LANGUAGES NONE` and an imported executable that is
never launched; it discovers exactly those eleven named cases. Four negative
copies remove an old case, omit the target discovery argument, omit the scanned
extra sources, or drop the compiled include; each is refused. The author's
independent inventory test also passes. This receives configure-time inventory
only, not eleven executed native tests.

The new native witness retains a real two-level partial composite mesh and
tests ordered widths1/3/5 with distinct constant targets. Identity diffusion,
homogeneous Neumann boundaries and zero seed give the original global constant
mode F(q)=q-target. It observes the actual candidate, original residual and
unchanged live solution before publication. The previous nonconstant coupled
N16/N32 and permutation cases remain unchanged and are still required for
native reception; the new constant witness does not replace them.

## Historical red evidence and limits

The old wave6 root log and XML hashes are listed in the 018 diagnosis. The log
SHA256 remains `e5108926f24636d83d17ce669976ae7bf39e2a14d19e59d3eec8df5911975681`;
rank0 XML `8b1c075eaf12a356eaf6d351ee01eff2774d78e0c4d3dc312dcb3d86bce8a9af`;
rank1 XML `fa9de948ee543196055539d6102eb4e488858d8be23cb83af42d36a2c9900b0c`.
They record two failures among ten cases on each rank of the old wave, and no
Serial execution of those new cases. They are not used as outcomes of c100.
No root receipt or shared file was modified.

```sh
rtk proxy env -u PYTHONPATH /Users/romaindespoulain/miniforge3/envs/pops-api040/bin/python -m pytest -q tests/review/test_sol61_amr_original_newton_sign_probe.py tests/review/test_sol61_amr_newton_rhs.py tests/review/test_sol61_amr_newton_fix_independent.py
rtk proxy env -u PYTHONPATH /Users/romaindespoulain/miniforge3/envs/pops-api040/bin/python tests/review/sol61_amr_original_newton_sign_probe.py --checkout . --output docs/development/api_040/sol61_amr_original_newton_fix_source_math_20260930.json
```

**38 source/math/CMake checks passed**: the previous17, the author's8 and13
new independent tests. Ruff and diff-check pass. No PoPS import, compiler,
native build, installed runtime, Kokkos or MPI execution was used in this
counter-reception. Root owns the rebuilt Serial eleven-case and MPI eleven-case
runs, including physical-original-residual checks and transactional outcomes.
Source compatibility does not close those runtime obligations or qualify a
complete nonlinear AMR PDE, convergence study or GPU execution.
