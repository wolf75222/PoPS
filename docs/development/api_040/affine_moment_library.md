# Affine raw moments as a public mathematical body

`pops.moments.affine_push_forward(moments, indices=..., matrix=..., offset=...)`
builds ordinary scalar expressions for the push-forward of a raw moment measure
under `v' = A v + b`. This is library composition, not a new Program opcode or a
physical closure. The core neither imports this library nor recognizes component
names, a fifteen-moment basis, or a particular velocity dimension.

For each explicitly supplied multi-index α, the polynomial
`prod_i (b_i + sum_j A_ij v_j)**α_i` is expanded and its coefficients contracted
with the supplied raw moments. The returned tuple follows the input index order.
Dimensions, permutations and orders are author-controlled. The complete set of
moments through the largest requested total degree is required for a general
dense affine map; absent moments are rejected even if some matrix entries are
literal zero. No closure is inferred from a particular value of A or b. Duplicate,
negative, boolean, mixed-rank indices and mismatched matrix/offset dimensions are
rejected. There is no explicit maximum order or velocity dimension; construction
and evaluation costs still grow with the polynomial basis.

The zero moment is the original object. No Jacobian determinant factor multiplies
density: these are moments of the **push-forward measure**, including a singular A,
not pointwise samples of a density function with a presumed inverse map. No
positivity, covariance or realizability repair is performed. Signed measures are
mathematically admissible to this helper. Physical domains belong to the authored
consumer, and its ordinary expression checks reject non-finite active results.

## Floating-point and evaluation contract

Coordinate factors are expanded in coordinate order. Each coefficient accumulates
contributions in the fixed polynomial traversal order; the final moment is a
left-associated sum of terms in lexicographic multi-index order. Shared coefficient
polynomials retain expression identity. No symbolic zero comparison, runtime
coefficient pruning, expression reassociation or new evaluator is introduced.
Existing lazy `where` and `rounded(binary64)` nodes retain their ordinary meanings.
Unused coefficients of a degree-zero-only request are not evaluated. Numeric
Python inputs use Python arithmetic; symbolic inputs use the common checked native
DAG. They are not a promise of identical exception classes between these modes.

This direct raw-moment expansion is not a compensated sum. Large translations or
strongly cancelling high moments can lose accuracy or produce non-finite
intermediates; finite arithmetic is checked by the common native path. The retained
`Program.affine_moment_update` has its own robust Cayley/exponential angle calculation,
positive-density and unchanged-density admission, and exact endpoint first-moment
copies. This tranche does not silently replace those contracts or claim bitwise
equivalence. It does not migrate the existing large-frequency AMR source-prefix
consumer or qualify arbitrary AMR stage trace realization of a direct pointwise map.

## Public consumers and evidence

The existing conditional consumer in
`tests/python/integration/runtime/test_conditional_consumers_runtime.py` now has an
`affine_library` variant. It authors a fixed-step Cayley matrix (dt=1e-4), translates
around the initial mean and explicitly preserves its positive-density domain. The
source stores the mathematical increment `mapped-old`; the Program publishes
`old + source(old)`. This is an explicit increment composition, with its associated
rounding, rather than an implicit Increment-to-State cast. It uses local StateStorage
without adding a transport flux. The old `affine` consumer remains available.

Two installed checks are prepared there:

* `test_native_consumer_accepts_inactive_and_rejects_active_invalid_branch[affine_library]`
  keeps the inactive invalid coefficient lazy and requires rollback on the active one.
* `test_native_affine_library_consumer_matches_particles_and_retained_recipe`
  compares the migrated and historical consumers against independently moved
  particles, including nonzero means and anisotropic moments. The fixed tolerance
  is rtol=2e-13, atol=2e-14; density is compared exactly.

`test_affine_push_forward_runtime.py::test_native_affine_body_on_stage_and_unknown_matches_discrete_measure`
prepares four further installed checks: 1V order six and 2V order five, each in two
index orders, on real Uniform Dim2 storage. It materializes the affine body at a
stage, solves the same body against that captured target from a distinct 1.1× seed,
and publishes the solved body's image. The original residual tolerance is 1e-12;
the particle-oracle acceptance tolerance is 3e-12 absolute, fixed before execution.
These velocity dimensions are independent of the explicitly chosen spatial Dim2.

Source/host reception before native execution:

* `test_affine_push_forward.py`: independently transformed weighted particles in
  1V/2V order six and 3V order three, permutations, density identity/signed zero,
  invalid bases and dimensions, declaration/stage/LocalResidual authoring,
  complete migrated storage-loader and Program emission.
* `test_affine_push_forward_host.py`: ordinary common C++ expression emitter under
  `-O2 -fno-fast-math -ffp-contract=off`, compared against moved particles and the
  actual retained `affine_velocity_push_forward<4>` header. Two finite angle regimes
  are checked, including a large angle, and active/inactive invalid translation
  branches with an explicit rounding barrier. This is a small host test, not an
  installed backend reception or an all-range numerical equivalence theorem.

Command (source checks, no JIT):

```sh
env -u PYTHONPATH python -m pytest -q -o pythonpath=python \
  tests/python/unit/moments/test_affine_push_forward.py \
  tests/python/unit/moments/test_affine_push_forward_host.py \
  tests/python/unit/codegen/test_conditional_consumers.py
```

For installed reception, use the central authenticated installed-only runner with
the two integration file paths above, the rebuilt Python package and its matching
native/SDK receipt. No installed, MPI, accelerator or performance qualification is
claimed by the source/host results.
