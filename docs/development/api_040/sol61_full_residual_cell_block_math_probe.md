# Full original-residual cell-block probe, 2026-10-01

This is a negative offline source/math reception. No PoPS import, native execution, production change, tolerance change or full-Newton success is claimed. The native donor is Root's read-only `bfae73f34174079bc6f2d8f4d50aa11d08eca5b5` source at `/Users/romaindespoulain/dev/tmp/pops-api040-native-reception-20261001`. Its retry1 receipt reports unchanged source/native, and its logs fail N32/012 at Newton index1 in serial and MPI2. The first-Jacobian 233-column result does not cover subsequent Newton equations.

The probe constructs the 1D partial-refinement scalar operator independently by restricting child values to covered parent cells, interpolating the two child ghosts with ratio-two quadratic weights, computing signed gradients, replacing both coarse interface fluxes by fine fluxes, and taking their divergences. It checks constant annihilation and physical weighted conservation. Witness diffusion/reaction matrices, cubic coefficient, target means/waves and all seven controls are read from the actual C++ fixture, with SHA256 pins emitted. This isolated finite matrix is diagnostic algebra, not an HPC provider or a new physical model.

The original equation is `F(q) = (S ⊗ D + I ⊗ R)q + 0.2 q³ − f`, with forcing constructed from the original target. A full cell block is the diagonal spatial block plus the complete reaction block and cubic derivative. Analytic profiles use the exact block. Central-FD profiles obtain every block column from two full evaluations of the original F, using the witness's normalized `1e-5` step. Those blocks are inverted with finite partial pivoting, rebuilt at each Newton iteration and held fixed throughout its GMRES. The pivot routine also receives signed nonsymmetric blocks of widths1/3/5 and rejects a singular block. This makes no SPD assumption.

GMRES retains restart80, budget240, two-pass MGS, the original weighted norm and the original relative stopping criterion. Each cycle checks the actual complete JVP on the accumulated correction. The nonlinear line search and final original-equation residual retain tolerance `2e-9`, twelve Newton iterations, linear tolerance `1e-5`, Armijo `1e-4` and minimum step `1e-6`. Central FD and analytic JVPs are distinct profiles; neither is asserted equal to Kokkos/MPI roundoff.

| Cells/order | JVP and blocks | Result | Last original norm | Last linear actual / stop |
|---|---|---|---|---|
|16/012|analytic|Solved|`2.537081e-11`|`2.535170e-11 / 4.612109e-11`|
|16/012|central FD|Solved|`1.962342e-12`|`6.660848e-14 / 4.612043e-11`|
|16/201|analytic|Solved|`5.988978e-12`|`5.706449e-12 / 4.611662e-11`|
|16/201|central FD|Solved|`3.159024e-11`|`3.157581e-11 / 4.611724e-11`|
|32/012|analytic|Rejected at Newton3 /240 columns|`4.633223e-6`|`1.426940e-9 / 4.633223e-11`|
|32/012|central FD|Rejected at Newton3 /240 columns|`4.599723e-6`|`1.471307e-9 / 4.599723e-11`|
|32/201|analytic|Rejected at Newton3 /240 columns|`4.640984e-6`|`1.713987e-9 / 4.640984e-11`|
|32/201|central FD|Rejected at Newton2 /240 columns|`7.157069e-3`|`1.082611e-7 / 7.157069e-8`|

All N16 original residuals pass threshold `1.745834e-8`; all N32 profiles fail threshold `1.750457e-8`. The analytic spatial-diagonal control also fails at Newton1: actual linear residual `2.148245e-5` exceeds `3.021569e-6` after240 columns. Its first solve takes233 columns. Local full blocks improve some linear solves, but do not close the whole nonlinear witness. Thus this reception does **not** recommend a sufficient `FullResidualCellBlockJacobi@1` production contract or silently requalify C7.

A future generic implementation would require complete original-F/JVP authority, finite pivoted blocks, all-rank ordered assembly and failure votes, fixed blocks within GMRES, rebuilding per Newton, and exact capture/phase/attempt/topology/lane authentication. Direct basis evaluation costs two original F evaluations per global stored DOF per Newton, plus baseline if needed; inverse storage scales as `O(m² Nstored)`. This active-DOF algebra exercises none of those native distributed storage, halo, EB or collective authorities. No size/rank/model cap or scaling promise follows from it. D(candidate) adds further complete coefficient differentiation obligations and remains a separate review.

Replay from the private checkout:

```sh
rtk proxy env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 \
 /Users/romaindespoulain/miniforge3/envs/pops-api040/bin/python \
 tests/review/sol61_full_residual_cell_block_probe.py \
 --source /Users/romaindespoulain/dev/tmp/pops-api040-native-reception-20261001 \
 > /tmp/sol61-full-cell-block-probe.json
```

Exit0 means the **expected negative classification** was received (four N16 successes, four N32 failures and the failed spatial-diagonal control), not native success. The JSON includes every cycle/stop, controls, source pins and full original final residual. Receipt SHA256 for this run: `50e2b80bc446de7976c387b0a65fe941dd88d359434749dad87876aee6da6fda`. Root owns any subsequent native qualification.

## Separate full-basis LU experiment

Root authorized one further bounded mathematical probe after the cell-block failure. With `--full-basis-lu`, the script constructs **every column of the complete original Jacobian** from two original-F central-FD evaluations at the current Newton iterate. Even the profile with analytic JVP builds its preconditioner this way. Partial-pivot LU factors that entire matrix; triangular solves supply the stationary right-preconditioner to the same GMRES. They do not replace GMRES or its complete-correction JVP check. Factors are rebuilt at each Newton, never reused from the seed. All seven controls and the original equation remain unchanged.

|N32/order|GMRES JVP|Newton steps / columns each|Final original norm|Max target error|
|---|---|---|---|---|
|012|analytic|4 /1|`3.184859e-12`|`9.254819e-13`|
|012|central FD|4 /1|`2.442288e-12`|`7.525092e-13`|
|201|analytic|4 /1|`2.633823e-12`|`6.683543e-13`|
|201|central FD|4 /1|`2.933323e-12`|`8.203438e-13`|

All four pass the unchanged original threshold `1.750457e-8`. The last full-correction JVP residuals are `1.24e-14`–`8.78e-14`, below their actual linear thresholds around `4.60e-11`. For this witness there are144 active DOFs, so assembly requires288 full F evaluations per Newton. The final original F is re-evaluated independently. A generic signed nonsymmetric matrix requiring pivoting is also checked against its original equation, and an exactly singular matrix is refused. This finite mathematical result is independent of the author's first-Jacobian helper, but is not a native reproduction or evidence of distributed LU implementation.

The positive math result permits proposing an **explicit, nondefault** `FullResidualBasisLU@1` realization. Its resource contract must enumerate the exact active-owned Krylov DOFs (level, layout, owner, cell, component), not include projected covered/EB-zero rows that would make the factorization spuriously singular. Every full-tower basis response must evaluate the original residual including restriction, halos, fluxes, reactions and candidate-dependent coefficients when supported. Assembly costs `2*Nactive` complete F evaluations per Newton, matrix/factor storage `O(Nactive²)`, and pivoted factorization `O(Nactive³)`. Full stored towers still cost scratch memory and every F evaluation's native work. These costs must be explicit and uncapped; this is an optional expensive reference realization, with no scalable-HPC or universal-convergence promise.

Production would need to authenticate captures and coefficient/state generations, exact point/phase, attempt lease, prepared lane, topology, layout/owner mapping and component order before basis evaluation. Matrix row distribution, collective assembly, allocation/finite/pivot failures and rollback must be explicit all-rank protocols. Factors must be immutable within GMRES and revoked/rebuilt on the next Newton iterate; no seed-frozen or secretly cached derivative qualifies. Signed/nonsymmetric finite matrices are permitted, while singular/nonfinite factors fail closed. The original full JVP and original F guards remain mandatory before outcome acceptance/publication. D(candidate) @3 and these authorities remain unreceived.

Replay uses the preceding command with `--full-basis-lu` and output `/tmp/sol61-full-basis-lu-probe.json`. Exit0 for that selection means four strict **math** successes, not native acceptance. Receipt SHA256: `8690b6047dda247c5bfb1d11d3ca1872c2abcd2a8e7865340ef31a12ac23e6c7`. The default cell-block selection remains negative and byte-identical to its previous receipt.
