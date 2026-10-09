# explicit-consumer-roles@1

This compiler contract selects where an explicit spatial stability budget is consumed.
It does not change the mathematical Program, its SSA identity, storage type, arithmetic,
operation order, solver controls, CFL course, coefficient bounds, or acceptance guards.

`Program.value` materializes State/RHS affine vector expressions in State-shaped storage.
Consequently a `linear_combine` with `vtype="state"` is not, by that label alone, an
independent explicit State advance. For example `K = R(U)` or `K = a R(U) + b S(U)`
is an observation/contribution when the accepted value is `U + dt K`.
The complete consuming update owns the State weight and every exact rate coefficient.

The reachability pass starts at committed State results. It also visits State inputs
sampled by contributing rates (including each coupled sampled State), explicit loop
initial/body results and explicit branch results. Successful acceptance guards retain
the role of their original value. Existing implicit-solver residual exclusions are
unchanged. A value reached first as a contribution and later as an actual State
consumer must be visited in both roles; its second use cannot lose its budget.

A nested contribution is only recognized as a pure rate observation when its leaves
are the existing authenticated rate operations, possibly under affine combinations
and successful same-value acceptance guards. This recognition does not certify its
coefficients, classify arbitrary maps as derivatives, or infer an algorithm from a
model, article, function name, Field name, or formula. All other nested State-shaped
combinations retain their existing strict classification.

Committed rate-only values and rate-only values sampled as a State remain true State
consumers and still fail for a missing State base. The existing affine budget solver
continues to require nonnegative State weights summing to one, first-power dt rate
weights, the exact sampled State/point/block and scoped frequency evidence. Negative
weights, higher dt powers, missing stages, foreign blocks, scheduled frequencies,
non-affine source effects, and absent regional frequency carriers keep their refusals.
No numerical runtime guard or tolerance changes under this contract.

The ordinary rate evaluations and their finite/status checks remain emitted. An
unconsumed diagnostic rate observation does not spend an accepted-step CFL budget;
any subsequent State consumption restores that obligation. The contract is a local
sufficient convex stability-budget construction, not an order, positivity, entropy,
or general stability theorem for arbitrary physical equations.

The original failure is reproducible with the unchanged public atomic-cubature
Python equations and public ForwardEuler composition: the named RHS scratch and
its accepted update were both classified as State consumers. Source witnesses use
the real installed package with Native loading blocked, real issued Program values,
no hand-written vtype overrides, and preserve exact Program serialization. Source
emission success is not a rebuilt Native, MPI, GPU, scientific or GitHub CI result.
