# M18 independent source and mathematical review - 30 September 2026

Reviewed candidate: `279e0cd9a696c6fccabafa949d3fa592ee9caa42`.
Private branch starts at `99d651c182d24aba5aaaf431c5e667f810ad5bb5`.
These are different historical branches; an exact targeted Git diff confirmed
that the example, native fixture, snapshot helper, offline oracle, twelve-case
countermodel harness and existing contract tests are byte-identical to 279e0cd9.
No prior result is used to qualify this candidate.

## Concrete defect and bounded correction

The original offline `compiled_contract` checked the freshly hashed IR and
three residual roots but did not inspect their scalar expression DAG or the
actual seed/capture/publication routes. Sixteen independently altered and
redigested public-source IRs passed that contract: zero residual roots,
reversed moment equations, altered quadrature weight, altered exponent basis,
swapped target component, captured data in an exponent, aliased unknown
component, replaced exponential operator, reversed solve inputs, future target
capture, foreign capture clock, boolean capture step, publication bypass,
failure consumer bypass, duplicate node id and a hidden operation.

`sol61_m18_original_equation_guard.py` now checks the serialized equations with
exact rational arithmetic over binary64 literal values. It expands arithmetic
into exponential atoms and requires, for each k = 0, 1, 2,

```
F_k(lambda, u) = sum_j w_j v_j^k exp(lambda_0 + v_j lambda_1 + v_j^2 lambda_2) - u_k
v = (-1, -1/2, 0, 1/2, 1), w = (0.1, 0.2, 0.4, 0.2, 0.1).
```

This is equation equality, not a finite set of sampled residual evaluations or
a copied compiler hash. The guard also validates the actual three-component
seed and readonly target state routes, their points and clock ownership, the
FailRun consumer and the dual publication path. All sixteen resealed source
IR countermodels are refused. The existing saved-state oracle invokes this
guard. No production Python, C++ header, fixture method, Newton budget or
tolerance changes.

The original/corrected observations are preserved in
`sol61_m18_independent_source_review_20260930.json`, SHA256
`19fa521242ce0b3731cc277dddcc3e66525db240ab782ff74f6f46dc3e081e63`.
These are source IR probes, not native state or external owner receipts.

## Independent mathematical obligations

The new mathematical reference uses stdlib loops and `math.fsum`; it imports
neither PoPS nor the author's moment/cone/entropy helpers. The retained polygon
is checked through homogeneous rational facets: mass >= 0, |first| <= mass,
second <= mass, and each of the four lower chords through adjacent (v, v²).
For example (1, 0.25, 0.1) lies outside the discrete cone even though it exceeds
the continuous parabola bound 0.25². Fourteen facet cases and all five extreme
rays are exercised; positive powers-of-two scaling preserves classification.
The exact classifier has no residual tolerance. The production review oracle's
2e-11 boundary band remains explicitly numerical; this review does not qualify
near-boundary native behavior.

For twenty mathematical multiplier vectors, each computed population is
positive. The Hessian is the Gram matrix sum_j p_j b_j b_j^T. All three Sylvester
minors are strictly positive under exact rational evaluation of the computed
binary64 populations. Thus the local dual objective is strictly convex on
this witness, without assuming an inverse or a special numerical outcome.

Two independent integer null directions, (1,-3,3,-1,0) and (0,1,-3,3,-1),
preserve all three moments. Their small positive perturbations satisfy the
global entropy identity H(q)-H(p) = grad H(p)·(q-p) + D(q||p), a strictly
positive relative-entropy gap and its curvature lower bound. Because
grad H(p) is lambda·b and the directions annihilate every moment, the
stationarity term vanishes. This is a global strict-convexity certificate,
beyond observing one increased entropy value. These vectors are mathematical
probes only; no fabricated positive NPZ campaign is generated.

## Fixture and future native evidence

The fixture has one actual 5-by-4 uniform periodic mesh: 60 dual unknowns and
60 readonly captured target values, with five derived populations per cell.
It publishes only the dual block. InitialConditions are BindArray with
ConservativeCellAverage and exact state subjects. Compile-once MPI routing and
the shared directory broadcast are inherited from the existing public helpers;
checks and state collection are collective. The saved helper writes actual
gathered buffers, geometry, rank boxes, temporal state and public checkpoints.

All ten phases are present: initial, accepted, continuous, restored, replayed,
outside_before, two outside rejections, safe_rebind_initial and
safe_rebind_accepted. An independent AST probe confirms both rejection attempts
use the same `failed` instance and no bind occurs in that loop. Recovery is a
fresh `bound(targets)` on the same artifact/context, not an accepted retry of
the impossible target. The changed cell (1,0,1.1) violates second <= mass, so
reducing dt cannot cure the immutable target. Initial checkpoints have explicit
bound_initial authority; a failed attempted run may have genuine run authority
while its accepted clock remains zero.

The twelve existing file-copy countermodels were inspected individually. Four
exercise solved multiplier/target/shape/overflow checks, one replaces the
impossible target by a feasible boundary, one corrupts rollback, two alter
clocks, one deletes ownership, two alter source capture/tolerance and one
changes a weight in provenance. They rehash copied checkpoints and file pins
and explicitly label their seal as negative harness authority. Their rejection
logic is consistent with the revised oracle. **They have not been executed on
authentic native files in this review**, because root's future Serial/MPI2
campaign and external owner pins are pending.

Positive reception requires the external root seal plus all ten phase file
pins, source/helper hashes, native DSO and ABI identities, artifact identity,
one successful named JUnit witness per rank, physical geometry/ownership,
checkpoint envelopes and actual solution equations. The owner must separately
authenticate the installed source manifest against the execution commit before
sealing; the oracle checks a pinned source_commit spelling but does not itself
reconstruct the complete installed source manifest from Git. A SHA format or
the pure mathematical checks alone provide no native authenticity.

## Commands and results

```sh
rtk proxy env PYTHONPATH=python /Users/romaindespoulain/miniforge3/envs/pops-api040/bin/python -m pytest -q tests/review/test_sol61_m18_independent_review.py tests/review/test_sol61_m18_entropy_offline_contract.py
rtk proxy /Users/romaindespoulain/miniforge3/envs/pops-api040/bin/python -m ruff check tests/review/sol61_m18_original_equation_guard.py tests/review/test_sol61_m18_independent_review.py tests/review/sol61_m18_entropy_offline_oracle.py
```

**73 source/protocol/mathematical tests passed**, including the subprocess that
forbids every PoPS import. Ruff passed. The exact original279 guard replay
observed 16 semantic acceptances versus 16 corrected refusals with recomputed
IR hashes. No install, shared environment mutation, JIT, build or native/MPI
execution occurred. This receives the moderate five-node M18 review mechanism;
it does not receive saved native states, near-boundary closure, AMR/GPU,
M19, Vlasov/BGK, spatial convergence or a complete migration PDE.
