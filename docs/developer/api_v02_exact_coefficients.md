# Exact coefficients and numerical admission

`Program(name, max_exact_coefficient_bytes=65536)` declares the maximum combined
base-2^32 limb storage of one exact flux coefficient. The value is an integer in
`[8, INT64_MAX]`, contributes to Program identity, and is exported by
`pops_program_flux_exact_coefficient_bytes`. The installed AMR budget contract
authenticates it. Older artifacts without this symbol use the documented 64 KiB
default. The existing bounds on live bases and polynomial terms remain active.

`ExactWeight` represents canonical signed rational numbers. Finite IEEE binary64
inputs are decoded exactly, including subnormals. A runtime rate multiplier `a`
attached to a duration `h` records `ExactWeight(a)/ExactWeight(h)`, rather than
rounding `a/h` and requiring its product with `h` to recover `a`. Applying the
measure rounds the exact product with the source duration once. For example,
`(.3/.07)*.07` differs from `.3` in binary64; the rational measure recovers the
actual `.3` coefficient executed by the field equation.

Static rational metadata uses the compact int64 representation when possible and
canonical hexadecimal integer strings otherwise. The numerical equation still
uses the normal `pops::Real` scalar lowering. Metadata never replaces an executed
field equation, and it is not an arbitrary-precision physical field backend.

A generated `HistoryTransfer` callback evaluates its scalar equations in the
same bounded rational algebra. IEEE input times and intervals enter as exact
dyadics; rational literals and `+`, `-`, `*`, `/`, integer powers, comparisons and
lazy selection preserve their mathematical values. Transcendentals and explicit
rounded nodes form documented binary64 boundaries. The target time, interval and
each weight are rounded once at the row output. Native field combinations consume
those rounded weights, while retained flux expressions consume their exact
values. A numerical weight that underflows to zero cannot erase a nonzero exact
source dependency. The row validates exact/rounded agreement and seals EW1 values
into collective and reconstructed-history identities before publication.

This closes a concrete counterexample to simply encoding computed doubles:
Lagrange weights at target `.10` from IEEE times `(.08, .16, .12)` computed with
separate binary64 operations sum exactly to `1 + 2^-54` when reinterpreted as
rationals. Evaluating the same rational equations exactly gives sum one without
normalization or a tolerance. Older native transfer callbacks with no exact-weight
payload retain their dyadic interpretation and are subject to the same closure
checks. The extension does not infer a conservation law for arbitrary authored
weights.

Every stored result and arithmetic temporary is checked before limb allocation.
An operation may exhaust its resource budget even when algebraic cancellation
would make the final result small. This is a declared workspace limitation,
not a fixed exponent or denominator limit. Operations build a candidate before
publication; collective metadata preparation preserves the existing rollback
boundary. Arithmetic is iterative host metadata work; no physical-model dispatch
or replacement runtime is involved. Up to a fixed number of temporary limb
vectors is live in an operation; the budget does not describe process RSS.

The EW1 wire representation is canonical: ASCII `EW1`, one sign byte, uint64
little-endian numerator/denominator limb counts, then their uint32 little-endian
limbs. Numerator and denominator are coprime, the denominator is positive, leading
zero limbs are forbidden, and zero is sign 0, zero numerator limbs, denominator
`[1]`. Budgets are receiving resource policy and are not embedded in mathematical
identity. Counts and declared byte limits are checked before allocation, followed
by canonicality validation. POPSANDE stores EW1 strings in face and interface
measures; POPSFLX5/6 store them in retained polynomial coefficients without/with
physical snapshots. Readers retain the previous int64 formats. Checkpoint
capacity estimates include variable weights and their stage-identity strings.

Clock phases still use the distinct int64 `amr::Rational` representation. This
change widens equation coefficients, not the clock protocol. Numerical fields
remain native floating point; nonfinite values and invalid physical intervals
are rejected separately from coefficient-storage exhaustion.

A signed composition can carry an explicit authored admission through
`guard(..., role=GuardRole.NUMERICAL_STABILITY, action=RejectAttempt())` (or
`FailRun()`). Its compiled Boolean must depend on that exact candidate and dominate
every admitted consumption. Dead guards, template-only references, guards of
another state, and unguarded paths retain the default numerical checks. This is
an executed acceptance predicate, not a theorem certified by the compiler. The
default sufficient convex stability checks remain active otherwise.

Source/compiler tests and native arithmetic, ledger, checkpoint, and scientific
receivers are separate evidence. The tests introduced with this change must be
received against the rebuilt PoPS native artifact before claiming runtime
qualification. No GPU execution claim follows from host exact-metadata tests.
