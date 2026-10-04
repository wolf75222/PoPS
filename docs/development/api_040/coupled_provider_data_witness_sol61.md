# Coupled-rate provider data: prospective independent witness

This test-only composition is distinct from BGK. It uses ordinary two-dimensional
spatial support, a non-square 7×3 or 3×7 layout, two two-component states and a
one-component read-only catalyst. Its public producers are
`AnalyticAux(seed=1+x+2*y)` and `DerivedAux(gain=2+seed²)`. The exchange equations
are `q=gain*c*(b-a)`, `da/dt=q`, `db/dt=-q`; both response components have rate
`gain`. The catalyst has no output rate. These declared equations are evaluated
without any model-name branch in the production emitter.

The Program evaluates at exact fractions 1/4 and 3/4: the first rate builds a
half-step predictor, the second advances the accepted state with dt=1/64. Input
order is either `(alpha,beta,catalyst)` or `(catalyst,beta,alpha)` and block
declaration order is independently reversed. Thus the provider's first input may
be the scalar catalyst while the output-loop driver is a two-component species.
Source checks inspect both exact evaluation times, prerequisite preparation before
local loops, immutable emission, and the declared provider's device view. NumPy
independently computes the two evaluations, full states, exchange conservation and
exact unchanged catalyst.

**The provider data here are fixed in time.** Two different evaluation fractions
do not establish a time-dependent provider. The current `AnalyticAux` contract
explicitly rejects analytic time expressions. A real public FieldProblem variant
was attempted on source 7586c23: its separate forcing carrier has
`ds/dt=cos(2*pi*x)`, its candidate Poisson field depends on the stage value of that
carrier, and `gain=2+phi²` would therefore change at the two fractions. Publication
to the three Model instances currently refuses with
“consumed field publication cannot share a model-definition provider key across
block instances”. The exact Python counterexample (SHA
`d2cbd64af6ce7a3e4583b670162384fdc384e7550df8d048be835dfdb513339b`)
and actual traceback are preserved in
`sol61-coupled-provider-data-preparation-20261005/field-publication-negative-7586.*`.
No publication guard, analytic-time contract, BGK production change or ABI was
altered to bypass that refusal.

`test_coupled_provider_data_runtime.py` is prospective installed execution. It
requires a subsequent Root-authenticated CPU package containing the admitted
production change, verifies installed Python and Native origins and the published
typed ABI, retains complete actual model/Program C25 before bind, and saves all
three states as NPY plus public composite CP9, exact clocks, consumer cursors and
local ownership at initialization and two accepted steps. Full states and
pointwise exchange balances use absolute 2e-12; the catalyst uses byte equality.
These checks have not been executed on Native. Local Source/Host checks do not
qualify callback publication, provider time dependence, MPI/GPU, or BGK science.
