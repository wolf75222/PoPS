# M11/W10: finite constrained friction through the public native solve

Scope is the closed **W10 finite witness**, not the complete Poisson–Maxwell–Stefan
model M11. Sources read: `corpus.json` M11/W10, handoff `CORPUS_ORIGINAL.md` M11,
mathematical notes “Collective constraints and joint constitutive fluxes”, and
`reference/PoPS_API_v0.4.0/legacy/v0.2.0/tests/run_revision.py`, lines 163–173.
The last file supplies the exact reference data:

```
friction = [[0,2,1],[2,0,3],[1,3,0]]
L        = [[3,-2,-1],[-2,5,-3],[-1,-3,4]]
force    = [.3,-.4,.1]
[ L  1 ] [j     ] = [force]
[ 1ᵀ 0 ] [lambda]   [0    ]
```

The legacy `xfrac=(.2,.3,.5)` is unused in those equations. This example does not
silently turn it into a physical closure. Molar masses, charges/electrochemical
potentials, composition-dependent friction, the Poisson equation/coupling, physical
initial/boundary conditions and the transposed finite-volume method remain
unspecified here. No three independent diffusion equations replace them. Full M11
and spatial collective constitutive-flux integration remain gaps.

## Public composition and original authority

`examples/migration/scientific/api040_m11_w10.py` declares two Blocks: three flux
components and one multiplier. The public `LocalResidual` product solves the full
four-component augmented equation using the existing native provider and consumes
the outcome before exposing either output. There is no inversion of L, Schur
assumption, custom opcode, Python callback per cell or post-solve renormalization.
L has rank two and kernel span{(1,1,1)}; the multiplier's 1×1 diagonal block is also
zero, while the complete 4×4 matrix is invertible.

Each initial flux storage value is captured as the dimensionless forcing for this
finite algebraic experiment. It is replaced by the solved flux after one Program
publication; the fixed .01 step has no claimed physical PDE time meaning. An exact
zero seed is distinct from the captured forcing. Two native guards require
`max_abs(L*j-force) < 1e-12` and `max_abs(sum(j)) < 1e-12` before **each** commit.
The solver tolerance is fixed at `1e-13` before native execution to leave room for
the requested original-equation criterion; no measured result was used to tune it.

For any force the augmented system has a solution, with
`lambda = sum(force)/3`. Therefore an incompatible force is **not** an incompatible
augmented system. Adding .3 to species zero at one cell gives lambda=.1 there:
the augmented residual and sum(j) can be tiny while the original `L*j-force` is
nonzero. The negative native test must fail at the generated `original_friction`
guard; accepting that multiplier and reporting only the augmented residual would
silently project the force and violate this witness. This distinguishes numerical
solve success from authorized publication.

## Independent reference and adversaries

`api040_m11_w10_oracle.py` has no PoPS import. It constructs L independently from
the off-diagonal friction matrix and solves the full 4×4 system with NumPy. It
evaluates original, augmented and constraint residuals separately. SVD/rank tests
confirm the singular leading block and nonsingular augmented system. The naive
independent formula `force/diag(L)` violates the constraint; subtracting its mean
repairs the sum but still violates the original equation. Neither is used as a
runtime algorithm.

The native test uses the exact constant legacy force and two nonuniform compatible
captures on a 4×4 mesh, reusing one artifact. Species permutation `(2,0,1)` permutes
both axes of L and the force; Block insertion and product-key order also reverse.
All returned fields are restored to scientific species order only for comparison,
without changing values. All seven source/pure tests passed. Both public variants
pass validate → resolve → ProgramModelGraph → emission; the two commits explicitly
depend on their original-residual and constraint guards.

Two native test variants are collected, **not executed by this worker**. They use
compile-once MPI preparation and an authenticated ExecutionContext, gather errors
before assertions and perform collective state reads. Accepted flux and lambda
are saved as NPZ and reopened for independent residual/reference evaluation;
metrics including actual lambda are recorded in JUnit. For the incompatible force,
the native diagnostic is recorded (the rejected provisional lambda is not exposed),
both blocks must remain bitwise unchanged, and time/macro-step/accepted temporal
envelope must roll back. A compatible fresh bind of the same artifact then runs;
this is not a hot mutation or successful retry of the incompatible runtime.

## Central reception commands and limits

Using the exactly installed candidate and the standard native-test environment:

```
env -u PYTHONPATH python -m pytest -q \
  tests/python/integration/runtime/test_m11_w10_constrained_runtime.py
```

Repeat with the repository MPI2 runner and its normal timeout. The standalone
example command is `python examples/migration/scientific/api040_m11_w10.py`; it
uses an MPI-enabled ExecutionContext (one rank is valid) and prints actual native
metrics and artifact identities. The integration tests additionally preserve
arrays and exercise collective refusal/rollback. No JIT, native build, installed
runtime execution, MPI or GPU test was run in this worker tranche. Source results
do not close W10's native receipt or the broader M11 model.
