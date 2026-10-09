# Small-block numerical scaling contract, version 1

This decision applies to `pops::detail::block_inverse<N>` and
`block_apply_inverse<N>`. It closes the determinant overflow/underflow defect
exposed by generic Program condensation. It defines numerical admissibility,
not a model-specific physical condition or a global solver policy.

## One admission decision

For each row, compute `s_i = max_j |A_ij|`, requiring finite entries and a
nonzero scale. Form `R_ij = A_ij / s_i`. Partial-pivot Gauss–Jordan on `R`
uses the relative pivot threshold `max(tol, N*epsilon(Real))`.
`tol` must be finite and nonnegative. Its historical default `1e-300` remains
source-compatible, but its meaning is now relative to the equilibrated block.
The machine floor is explicit; it is not a tolerance on a physical quantity.

Both ordinary and extreme exponent paths use this same decision. Multiplying
a row by a finite positive change of units therefore does not switch between
an absolute determinant criterion and a relative pivot criterion. Input
rounding, underflow already present in the input, and the representability of
the requested output still bound this invariance.

## Output and arithmetic

The inverse is `A^-1 = R^-1 D^-1`, so column `j` is divided by `s_j`.
Candidates are fully computed and checked before writing the destination.
Failure leaves every destination entry unchanged, including for generic N>3.

For N=2/3, the existing direct-division inverse and factored-vector operation
trees remain in the ordinary exponent range. The range is bounded by the
fourth roots of the smallest positive normal value and largest finite value,
with a factor `N+1` for sums. The relevant closed expressions contain products
of degree at most four. These bounds select a representation; they do not
decide invertibility. Nonfinite closed-form results also use the balanced
representation.

Applying the inverse never requires materializing `A^-1` outside this ordinary
range. Each term `R^-1_ij v_j / s_j` is represented by a binary mantissa and
exponent. Terms are added in decreasing exponent order, renormalizing after
each addition, and the final value is converted to `Real`. Thus neither an
overflowing determinant nor an overflowing intermediate adjugate product
forces failure when the resulting solution is representable. Each output row
has its own scaling: small independent solution components are preserved.

Examples covered include `diag(1e308,1e-308)`, where both inverse entries are
representable, and `diag(1e-320)` with a right-hand side `1e-320`, where solving
gives one even though materializing the inverse must fail. Nonrepresentable
solutions, NaN/Inf inputs, and numerically singular blocks fail without
publication. Aliasing `out` with the right-hand side is supported by temporary
publication.

## Boundaries of this decision

This is floating-point elimination and accumulation, not exact arithmetic,
iterative refinement, an SVD, or a claim about arbitrary ill-conditioned
systems. The relative pivot floor can reject a mathematically invertible
block. Generic condensation must still check its original coupled residual
when that solve interface supplies such a criterion.

The balanced admission adds work to the ordinary small-block path. This
change prioritizes one coherent domain of validity; no performance improvement
is claimed. Existing exact operation-tree parity tests remain required.

Host tests exercise the actual header and emitted condensation fragments.
MPI and GPU execution require separate central acceptance; source portability
annotations and host sanitizers are not GPU evidence.
