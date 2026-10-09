# Scientific conditional and floating evaluation

`pops.math.where(test, yes, no)` calls both nullary expression builders once
during authoring. The resulting common Expr node evaluates only the selected
branch in native C++. Branch-local operations and their CSE temporaries remain
inside that branch, including in Program component expressions and local Newton
residuals. Named physical primitive recipes are inlined into this same DAG
when an evaluation boundary requires their scope. An invalid inactive branch contributes no failure; a non-finite
selected intermediate is rejected by the checked Program evaluation before any
candidate is published. This API does not invoke Python from a cell loop.

`pops.math.rounded(expression)` retains an explicit binary64 evaluation boundary.
Its native realization uses a volatile double store and load. Its identity
differs from the unrounded expression; CSE cannot remove the boundary. Its
formal real derivative is the derivative of its argument. This does not claim
a bitwise derivative of IEEE rounding or cross-architecture reproducibility.

The existing analytic initialization API keeps its own accepted input types;
`pops.math.where` is the common physical/Program expression API. A predicate must
be a typed symbolic Boolean expression and both branch arguments must be
callable expression builders. Unsupported argument families fail explicitly.

Production DSL artifacts now always use `-fno-fast-math` and
`-ffp-contract=off`, including when `POPS_DSL_OPTFLAGS` selects a custom
optimization level. Conflicting options are rejected, rather than silently
overridden: `-Ofast`, `-ffast-math`, `-fassociative-math`,
`-ffinite-math-only`, `-fno-signed-zeros`, `-freciprocal-math`, and enabled
floating contraction. This deliberately removes previously accepted unsafe
optimization choices. The effective flags enter the artifact cache identity.
The change is part of the pending public API 2 / semantic IR 2 migration,
alongside the new expression identities.

The strict profile does not establish that H applied to finite-volume stored
coordinates equals the finite-volume representation of H. That representation
obligation remains separate from floating evaluation and conditional laziness.

The current realization also routes diffusion constitutive evaluations and
linear coefficient consumers (local apply/solve, affine moments, condensed
matrices) through the same checked expression emitter. Diffusion expands named
primitive recipes before sampling. Linear coefficient matrices retain their
existing requirement to be independent of conservative and primitive state
variables; conditional coefficients over auxiliary fields and parameters remain
valid. The publication and failure protocols belong to each existing native
consumer, with installed runtime checks reported separately from emitter tests.
