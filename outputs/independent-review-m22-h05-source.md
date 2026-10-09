# Independent review: closed M22/H05 reservoir slice

Reviewed `4ccfaf8` in `PoPS-resource-lifetime` against the baseline registry, historical H05 calculation, public authoring, pure oracle, documentation, and prepared native tests. This is a source review only; the author reports 13 source/oracle checks passing, but I did not run or compile this tranche.

## Scope and mathematics

The handoff's H05 fixes only the algebraic normalized exchange, with `(E0,T0)=(2,0.5)`, `k=0.8`, and backward Euler `dt=0.4`, plus the scalar M1 endpoint factor. The authored two-model `LocalResidual` is exactly

`E1-E0 + dt*k*(E1-T1) = 0`, `T1-T0 - dt*k*(E1-T1) = 0`.

It captures `old_E`, `old_T`, and a bound `k` separately from the two Newton seeds, consumes the joint outcome before `commit_many`, and guards the old and accepted energies and the nonnegative rate. The model source operator only materializes `k` for the Program; no extra physical source or transport equation is attached. Two blocks have homonymous `U.value` states, and both canonical and reversed block insertion are exercised. The NumPy oracle solves the independent 2×2 matrix and gives `(70/41, 65/82)` after the specified step; its continuous mean/difference formula is independently compared with a matrix exponential. The scalar `chi` oracle is rationalized from the registry formula and tested separately. The script and document explicitly exclude full M22, a material `T^4` law, opacity, Marshak boundaries, a spatial pressure tensor, and diffusion limit. That scope is honest.

`FixedDt(0.4/count)` with counts `(1,2,4,8,16)` requests the same final time and supports a first-order comparison to the continuous solution. Criteria are declared in the source and document before native execution. The script gathers both native state fields, saves/reopens each NPZ before comparing with the discrete 2×2 oracle, checks cellwise inventory and nonnegative values, computes time-order and reversed-block agreement, and records hashes/ABI. The installed test uses one artifact per block order and rebinds `k=0.8, 0, -0.8`; the negative case checks unchanged state and clock. Source identity uses a block-qualified parameter; the public bind schema authenticates its canonical alias by qualified identity, without a name-only fallback. Actual installed binding remains unverified.

## Independent counter-tests for native reception

1. **Cell-locality discriminator:** bind nonuniform positive 4×4 arrays, for example `E[i,j]=2+0.1*i+0.03*j`, `T[i,j]=0.5+0.02*i+0.07*j`, with `k=0.8`. One accepted step must match the independent 2×2 backward-Euler solve at every cell, preserve each cell's `E+T`, and retain spatial differences. The current all-constant 4×4 witness cannot distinguish a correct local solve from accidental global broadcast/copy. This adds coverage without altering H05 physics or thresholds.
2. **Capture/rebind discriminator:** use the *same artifact and initial arrays* with `k=0` then `k=0.8`; zero rate must preserve every cell, whereas positive rate must exchange energy in the direction set by that cell's `E-T`. The prepared integration test already covers the parameter values on uniform data; nonuniform data would also test per-cell capture and block identity.
3. **MPI failure harness:** the prepared test catches only `RuntimeError` from the negative-rate run before an all-rank gather. An unexpected rank-local `ValueError` or other exception would bypass that agreement and could leave peers waiting. Capture and converge the exception class/text before assertions, then require a diagnostic from the nonnegative-rate guard rather than accepting an arbitrary failure. This is a test-harness robustness gap, not a demonstrated native solver failure.

The nonuniform, same-artifact rebind and collective-error checks are prepared in
`tests/python/integration/runtime/test_api040_m22_h05_adversarial_runtime.py` as
an independent installed test. It requires the H05 example to be integrated
first; it has not been run or JIT-compiled. The strict negative assertion
requires `RuntimeError` carrying `nonnegative_rate` after all ranks have
reported their exception class and text. These are proposed reception tests,
not measured defects. The current source slice is reviewable and mathematically
faithful within H05; no native H05 qualification is claimed here.
