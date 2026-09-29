# Independent review of T3 local-product reception `acf2f03`

Reviewed the five test/documentation files in the isolated
`PoPS-resource-lifetime` checkout, based on integrated source `cfef849`.
No production code or native package was changed by this review.

The independent NumPy oracle does not call PoPS residuals. The public body
evaluates `P.source(2*z)` and `P.apply(z)` on three separate Models with
widths 1, 2 and 4. The resulting original equation for each component is
`4*g*x²+(c+1)*x+0.1*sum(all x)-old=0`. The oracle manufactures all seven
nonconstant component values, constructs their captured old values, and checks
every original residual after native execution. Two distinct, nondefault gain
triples use exact block-qualified parameter handles with the same compiled
artifact. Reversed block insertion and unknown-key order are separate test
variants. The helper retains its original three-value return unless the new
`return_parameters=True` option is requested.

For the refusal case, one cell has all seven captured old values set to -100.
Summing the equations and completing squares gives
`sum(R) >= 700 - sum((c+1+0.7)^2/(16*g)) = 695.506665... > 0` for the selected
positive gains. Thus no real simultaneous zero exists; the refusal is not
inferred from a solver iteration count. The source test checks this bound at
the minimizing vector.

In the native witness, each `pops.run` exception is collected on all ranks
before checking its type and `coupled_implicit failed:` diagnostic. State
reads are collective on every rank, while the root compares all three blocks
bitwise against their pre-attempt values. The test also checks time, macro
step and temporal envelope after two refusals of the **same** runtime. A new
bind of the same artifact with consistent inputs then succeeds; this is not
called a successful retry of the incompatible runtime. I found no source
counterexample in these bounded guarantees.

I independently ran the five new pure/source oracle tests: **5 passed in
3.04 s**. The author reports 17 targeted source checks and two collected
native variants. No native, MPI, JIT or GPU run was performed in this review.
The central rebuilt package must still qualify both variants and their exact
rollback/accepted-state evidence. The test does not claim arbitrary component
permutations, independent asynchronous block clocks, uniqueness outside the
selected positive branch or a general Schur implementation.
