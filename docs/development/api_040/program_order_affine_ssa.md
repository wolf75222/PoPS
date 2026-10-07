# Formal order from affine SSA expressions

Seven existing factory-order assertions failed on the unchanged baseline: a named
rate value is a real `linear_combine` SSA node, but the certificate decoder expected
the endpoint to refer directly to the underlying rate call. The library correction
recursively traces exact polynomial coefficients through state/rate combinations.
It reconstructs the initial-state weight, stage matrix and endpoint weights before
applying the existing Runge-Kutta order conditions. Debug labels, preset names,
physical model names and formulas do not select an order.

This is a missing library analysis over existing public Program constructs. It adds
no language primitive, schema field or Native ABI and does not rewrite the floating
arithmetic emitted for execution. The public certificate contract is unchanged.
Unknown primitives, representations, evaluation effects, clocks, cadence, forward
SSA rate dependencies and invalid coefficient powers remain refused. The entire
executed graph is checked: an opaque effect is not ignored because it is disconnected
from the committed value or multiplied by zero.

The initial-to-endpoint interval remains0 to1. Formal stage abscissae can lie outside
that interval. The first candidate incorrectly imposed `0 <= c <= 1`; independent
public counterexamples revoked its generic verdict before Main integration:

| Stage row | Endpoint weights | Abscissae | Formal order |
|---|---|---|---|
| A21=2 | (3/4,1/4) | (0,2) | 2 |
| A21=-1 | (3/2,-1/2) | (0,-1) | 2 |

Both graphs use already available initial State and preceding rates. Their stage
time does not introduce a forward SSA dependency. The R2 candidate removes the
unjustified bound while preserving clock/window identities and effect checks.
Native evaluation-point admission remains a separate contract; these formal
certificates do not claim native execution, physical PDE convergence, spatial
stability or asymptotic preservation.

The author receives61 Source tests, with the original seven order assertions intact.
The independent lane receives52 Source tests and15 formal probes, including
autonomous `y'=y*y` and nonautonomous `y'=(1+t)*y` coefficient recurrences, nested
aliases, stades2 and-1, and renamed methods. A method named RK4 with a genuine
second-order coefficient graph is classified as order2. The primary replays61
tests on integrated Main:61 PASS, zero failure/error/skip, no Native module loaded.

R2 patch SHA256 is
`4090f1041d1c54a0036e1d6245c0690a3a8d2c00a2ee27001b34fe9784ad0e46`;
independent revised review SHA256 is
`a4cb58c1c3cec4f3118f3c12ddad46f0dc909b2f47b775dcd44fa938fa624eed`.
The first candidate, its revocation and both counterexamples are retained as
separate evidence. The correction must be packaged through the official incremental
build before installed-package/native qualification of the updated revision.
