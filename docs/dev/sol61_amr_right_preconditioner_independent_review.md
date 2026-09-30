# Independent C++ right-preconditioner review — 1 October 2026

Reviewed original candidate `c7cdd2c38cc7145abb5981b90f38e9b6035b6e5e`
(parent `619fec88665983843b00e33f52c136658e99befb`), then author corrections
`ec2571305db245e6040d40189226215d812dee60` and
`3a03a2ef49d37c6fe779db85672fd2b1a89cf86f`. Private checkout:
`PoPS-sol61-amr-right-preconditioner-review`. MAIN, installed environment and SDK
were not edited. Only small standalone host C++ probes were compiled, one process
at a time; no native Kokkos/AMR/MPI, JIT, installation or large build was run.

## Two authority findings and received corrections

On c7, the authentic `require_authority` accepted a distinct lane object while
owner, attempt, points, captures, topology and coefficient generation matched.
Preparation had checked the lane once, but invocation votes/Newton used the new
argument while provider apply/dot used its original lane. This missing invocation
check existed in the legacy carrier too; the new @2 realization made the retained
lane claim explicit. The first correction ec25713 adds the missing comparison.

That comparison initially ran inside `local_phase_(lane)` on the untrusted
argument. A rank-local foreign-lane injection could therefore select a different
MPI context for its validation vote from its peers. This collective-context
finding follows from actual source lane selection; no real MPI hang is claimed.
Correction 3a03a2e obtains the provider's prepared `authority_lane`, votes and
agrees on that lane, and checks the requested lane inside that authenticated
phase before provider/JVP work.

The independent scaffold compiles the **actual** invocation method and records
which lane its local phase chooses. It substitutes the collective transport and
metadata containers; it does not authenticate a native communicator. Recorded
profiles in `tests/review/sol61_amr_right_preconditioner_receipt.json` show:

| Source profile | Foreign lane refused | Validation context |
|---|---:|---|
| c7 | false | foreign lane 2 |
| ec25713 | true | foreign lane 2 |
| 3a03a2e | true | prepared lane 1 |

Each profile also passes thirty-six separate host assertions. Source/header and
generated-scaffold hashes are recorded; these are ordinary host receipts, not
external seals of scientific/native results. Source phases for historical
profiles are read from their exact Git objects. The receipt uses full revision
identities and the actual supplied source bytes.

## Right GMRES and original-equation checks

The workspace uses `J(q) M⁻¹ v` for every Arnoldi direction. Its correction is
accumulated from the same stationary right-preconditioned basis, using existing
image storage. The image is then safely reused for `J(q) correction`. Acceptance
of a projected Arnoldi stop still requires the norm of the actual complete linear
defect `rhs - J(q) correction`, including restart and approximate JVP cases.
Inactive/covered storage is projected out before JVP and correction accumulation.
The old `solve` delegates through an identity callback; no new default numerical
preconditioner or tolerance is selected there.

The extracted authentic Arnoldi/rotation/backsolve/update/recheck phase runs over
an independent weighted host vector scaffold. Three component permutations of
a signed nonsymmetric matrix converge with the original matrix residual below
the declared stop. A fourth inactive stored component is never evolved despite
deliberately nonzero scratch values. A direction-dependent approximate JVP must
still satisfy the independently recomputed complete-correction residual whenever
the source reports convergence. Iteration budgets are unchanged. These checks
receive algebraic statements, not spatial ghost filling or AMR stencils.

Six in-memory source mutants bypass right apply in Arnoldi or correction, replace
the true JVP, omit the affine zero response, force positive diagonals, or ignore
inactive coverage. All fail the independent equations/admission checks. Two lane
mutants separately restore c7's admission or ec25713's foreign validation context;
both are detected. No production source is edited by these mutations.

The physical provider/body still evaluates original +F, the Newton RHS alone is
-F, and central FD remains applied to original +F. Candidate availability still
requires original residual recheck after synchronization, followed by authority
validation; Outcome consumption/publication is unchanged. This review does not
replace the original F by a preconditioned residual or a condensed local inverse.

## Spatial-basis realization and limits

`SpatialBasisJacobi` computes the actual response diagonal
`d_i = [A(e_i)-A(0)]_i`. The host probe executes the authentic Kokkos-lambda body
with one-cell view/reduction substitutes. It receives affine-offset subtraction,
signed negative and very large finite diagonals, inactive zero inverse, and
refusal of zero/nonfinite diagonals or nonfinite reciprocals. Negative diagonals
are legal for right GMRES; there is no SPD assertion.

Source inspection receives complete globally ordered **stored** cell/component
traversal over all levels. Nonowners skip local insertion/extraction, never the
operator collective. Covered storage still counts toward preparation cost, and
its inverse stays zero according to the prepared active mask. Local initialization
and extraction vote/fence before another apply. The actual prepared native
operator supplies the responses; no model name, reaction formula, component cap,
grid size or guessed stencil diagonal appears in preparation.

One zero response plus one full operator application per stored DOF is an
explicit O(stored DOFs) application count. It has overflow admission and no
artificial count ceiling, and retains one owned inverse tower. This is expensive
reference realization, not an efficient/default AMR smoother. O(stored DOFs)
**applications** is not O(stored DOFs) total arithmetic cost; with a linear-cost
operator apply the preparation can have quadratic total work. Its stationarity
depends on the already frozen linear spatial operator; it does not assert that
an arbitrary nonlinear provider's unit response equals its Jacobian. The current
native provider's actual boundaries/ghost interpolation/restriction/reflux remain
to be received by root, rather than inferred from these host substitutes.
The authentic application-count loop is also extracted and executed: two-level
stored counts give 128 applications; a representable very large count passes
without a fixed ceiling; a product exceeding size_t refuses before operator work.
Twenty authority mutations cover owner, attempt/level leases, exact point fields
(including one-ULP physical-time change), captures/components, topology and
materialization/coefficient generations. A substituted negative agreement reply
is refused by the actual contract method.

## Contract, controls and evidence

Identity remains the default prepare argument, uses residual authority @1, and
adds no Jacobi contract bytes. Explicit Jacobi uses residual authority @2 and
`pops.amr.original-spatial-jacobi.basis-response@1`. The existing exact builder
sequence for identity is preserved by the author's replayed historical-source
comparison. Unknown realization and mixed-rank realization native tests remain
present. Python ports, IR promotion and checkpoint changes are outside this
C++-only reception.

The positive old test name is retained but now explicitly selects Jacobi; that
realization change is documented and recorded in JUnit. It does **not** qualify
the historical identity realization. A distinct N32/012 identity case expects
the actual iteration-budget rejection, consumes RejectAttempt and checks no
visible candidate/live publication. Both profiles use the same original helper,
N16/N32, permutations, physical body/captures, FD step 1e-5, nonlinear tolerance
2e-9, linear tolerance 1e-5, restart 80 and budget 240. No tolerance/physics change
is used to erase the old red history. Other existing test names remain present.

Run from the corrected private checkout with
`PY=/Users/romaindespoulain/miniforge3/envs/pops-api040/bin/python`:

```sh
rtk proxy env -u PYTHONPATH "$PY" -m pytest -q tests/review/test_sol61_amr_right_preconditioner_independent.py tests/review/test_sol61_amr_spatial_jacobi_realization.py tests/review/test_sol61_amr_gmres_full_residual.py
rtk proxy "$PY" -m ruff check tests/review/sol61_amr_right_preconditioner_probe.py tests/review/test_sol61_amr_right_preconditioner_independent.py
rtk git diff --check
```

Observed: **11 new independent pytest cases PASS**, including actual host probe's
**36 assertions**, and **7 author source/host cases PASS**; total **18 PASS**.
Ruff/diff PASS. Standalone probe:
`rtk proxy "$PY" tests/review/sol61_amr_right_preconditioner_probe.py . outputs/right-host`.

Source integration requires c7 + ec25713 + 3a03a2e together. No additional current
P1/P2 is demonstrated within this bounded C++ reception. Root still owns actual
serial/MPI, full-tower physical/operator, all-rank authority failures, empty-rank,
rollback/publication and cost reception. The retained positive test names or
passing scalar host scaffolds do not substitute for those native results.
