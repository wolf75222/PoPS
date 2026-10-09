# M19 isolated self-BGK: declared inputs and prospective receipt

The original corpus (`CORPUS_ORIGINAL.md`, M19) asks for Vlasov–Poisson and then
BGK, including isolated relaxation. `mathematical_original.tex` explains that a
BGK collision is local in position and nonlocal in velocity through its moments.
Its cited [Gkeyll collision-model documentation](https://gkeyll.readthedocs.io/en/latest/dev/collision-models.html)
defines self-species BGK as

\[
\partial_t f=\nu(M[f]-f),\quad
n=\int f\,dv,\quad u={\int vf\,dv\over n},\quad
\theta={\int v^2f\,dv\over n}-u^2,\quad
M={n\over\sqrt{2\pi\theta}}\exp[-(v-u)^2/(2\theta)].
\]

The product support, explicit weighted reductions, constant extension, arbitrary
arity coupled rates and read-only catalyst states already existed. The missing
mechanism was an explicit coupled rate's read of a declared auxiliary provider:
the public Maxwellian needs a velocity coordinate, but the old emitter rejected
every auxiliary leaf as “prim/aux vars are deferred”. The separate moment-hierarchy
`bgk_source` helper does not constitute this kinetic product-space witness.

Explicit coupled rates now bind providers from the exact operator owner's
`ProgramModelGraph` and immutable `ProviderPack`, using expression first-use order.
Their existing collective `prepare_provider_values` boundary precedes any local
Fab loop. The device lambda reads the resulting provider FieldView by value and
the exact conservative input views read-only. Evaluation time uses the authored
explicit StagePoint. Missing/foreign provider authority and ambiguous conservative
names remain errors; primitive and implicit auxiliary recipes remain deferred.
Provider-free coupled kernels retain their previous emitted bytes. No ABI or
physical model dispatch was added.

`tests/python/support/m19_bgk_case.py` is a public Python composition: three scalar
moments are reduced from `(velocity, position)` to position, then lifted into
three read-only phase states before *each* SSPRK2 collision evaluation. Velocity
is a spatial fibre rather than a component axis. The accepted distribution uses
the authored SSPRK2 tableau; moment states store the last predictor moments,
explicitly rather than claiming endpoint caches. An unrelated two-component state
remains unchanged. Reversing declarations and changing the collision scope does
not change this routing.

The original corpus gives Nx=32,64 and Nv=32,64, a truncated velocity domain and
defined boundary fluxes; it does not prescribe a BGK collision frequency or initial
mixture. This bounded witness explicitly declares v in [-8,8], periodic x in [0,1],
nu=3/2 and dt=1/128, and a positive equal mixture of Gaussians at velocities ±0.7
with standard deviation 0.75 and density `1+0.1*cos(2*pi*x)`. There is no transport
operator in isolated relaxation, hence no velocity boundary transport flux. Values
are the authored uniform cell-state samples. The continuous Maxwellian is evaluated
at velocity cell centres without a discrete moment fit or renormalization. The
independent NumPy reference reports the finite quadrature defect, and advances the
same SSPRK2 instead of replacing it by an exact exponential step.

The prospective installed witness is:

```sh
env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 "$NEW_CPU_ENV/bin/python" -m pytest -q \
  'tests/python/integration/runtime/test_m19_bgk_runtime.py::test_installed_isolated_bgk_two_stages[32-32-False]'
```

It requires a real subsequent CPU build containing the generic emitter change,
exact installed Python/Native origins and the published typed ABI. It retains
actual complete C25 model sources, module IR, Programs, DSOs and compile sidecars
before bind, and joins all six compiled transfers to their verified source
packages and binary bytes. Each initial and accepted receipt records all eight
NPY states, public composite CP9, exact clock, consumer cursors and local ownership.
It compares full distribution and predictor moments to independent NumPy at
2e-11 absolute, checks all three moments against initial values at 1e-12, positivity,
and exact untouched-state bytes. These are explicit prospective witness guards;
no previous scientific criterion was relaxed or previous result reused.

Local Source authoring/lowering, independent reference checks and actual-header
Host syntax do not establish installed BGK evolution, MPI, GPU, collision rollback,
Landau/two-stream physics, temporal/spatial convergence or all of M19. Those claims
require subsequent Root-owned executions and closed scientific receipts.
