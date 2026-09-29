# Independent public SSP/path stability review

Tests live in `tests/python/unit/test_api040_path_ssp_stability.py`. They import the
current MAIN Python package source read-only, resolve an ordinary public Case,
and emit the real Uniform or AMR Program source. No native mesh, MPI, GPU or
installed-package execution is claimed.

## Rational certificate

For the public Butcher SSPRK2, the final expression is
`U0 + dt R0/2 + dt R1/2`, where `U1 = U0 + dt R0`.
Its exact identity is `U0/2 + U1/2 + dt R1/2`. The predictor and endpoint
therefore have `(alpha,beta)` equal to `(1,1)` and `(1/2,1/2)` respectively.

For the public Butcher SSPRK3, `U2 = U0 + dt(R0+R1)/4` and the endpoint is
`U0 + dt(R0+R1)/6 + 2 dt R2/3`. The corresponding exact identities are
`U2 = 3 U0/4 + U1/4 + dt R1/4` and
`U_next = U0/3 + 2 U2/3 + 2 dt R2/3`.
The three budgets are `(1,1)`, `(1/4,1/4)`, `(2/3,2/3)`, with one current
RHS in each budget. Earlier RHS evaluations are not spent a second time.

The implementation substitutes only positive exact rational identities and
leaves the emitted field arithmetic untouched. This proves a sufficient
convex decomposition for these authored stages; it is not a universal claim
for all RK tableaux or all floating-point rounding behavior.

## Results

Initial independent run: **7 passed in 11.74 s**, receipt
`outputs/path-ssp-source.xml`:

- public SSPRK2 and SSPRK3 through both source emitters;
- M17's acceptance topology (current, predictor, endpoint guards), on an
  independent two-component SymbolicPath rather than the large Fan–Li model;
- guard names retained in source and two effective stability budgets;
- explicit midpoint refused, despite its positive second-order tableau.

An additional adversarial public Program exposed a fail-open boundary:
`commit(q.next, dt * R(q.n))`. The actual endpoint has `vtype='state'` and
`op='linear_combine'`, but the partition proof returned no groups when no
state weight existed. Source emission succeeded without a stability budget.
`test_accepted_rate_without_state_budget_cannot_skip_stability_proof` was
**red in 1.80 s** (`outputs/path-ssp-no-state-red.xml`). Root was notified to
distinguish an unconsumed diagnostic rate expression from an accepted state
consumer without a convex state budget. No production source was edited here.

After root added the explicit-state refusal, the complete independent SSP and
consumer suites passed: **15 passed in 24.05 s**, receipt
`outputs/path-ssp-consumer-source-green.xml`. This includes the two small host
C++ compilations of the actual emitted scalar consumer guard from the earlier
consumer suite. The common source proof suite also passed separately:
**17 passed in 0.49 s**. MAIN was at `e621b3d` plus root's uncommitted two-line
explicit-state refusal when these checks ran. Combining tests from the two
checkouts in one pytest invocation was not valid because their same-named
conftest modules collide; the suites were then run in separate processes.

Reproduction from this checkout:

```sh
env -u PYTHONPATH /Users/romaindespoulain/miniforge3/envs/pops-api040/bin/python \
  -m pytest -q --tb=short -o pythonpath=../PoPS/python \
  tests/python/unit/test_api040_path_ssp_stability.py
```

The prior root `outputs/path-consumer-source.log/xml` and the prior independent
red/green consumer receipts were preserved. Native acceptance remains a
separate central reception.
