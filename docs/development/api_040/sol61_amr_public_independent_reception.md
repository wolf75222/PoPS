# Independent source reception of the public original AMR field connector

Candidate: `a8d6f04b34596d804c72f3a80abbeea4c4f5e319`, parent
`85a79b6f4c346f1a523470d38b14094fdf6226ef`. No production file changed by
this review. The capability's earlier independent reception remains `c8907cd`.
No additional blocker was demonstrated in the public connector.

## Public physical source and independent calculations

`tests/review/test_sol61_amr_public_original.py` imports no author fixture or
oracle. It declares the physical source before Program wiring, and constructs
original FieldProblem equations with 2, 3 and 5 co-localized unknowns, distinct
forcing/material State captures, signed nonsymmetric full diffusion matrices,
nonlinear cross reactions, permutations, and explicit central finite differences.
The exact physical body is

`R_i = alpha*q_i + (i+2)/10*q_i^3 + .03*q_i*q_next + .02*q_next^2 - f_i - sum_j D_ij*Delta(q_j)`.

An independent AST interpreter compares the encoded original reactions with this
formula on nonconstant sample arrays. An independent periodic anisotropic
five-by-four stencil checks matrix signs, row/column permutation and rejects a
transpose. An independently differentiated polynomial checks the explicitly
selected central full-residual derivative. These are algebraic/source checks;
they do not run an AMR solve or derive a continuum convergence certificate.

The public resolved/emitted witnesses include periodic and homogeneous Neumann
boundaries, two levels/ratio two, refinement tagging, explicit optional seed,
accepted history, FieldObservation publication into the physical auxiliary
carrier, and genuine SourceTerm consumption in the temporal balance. Field
sizes, names, order and solver controls differ from the author's witness.
No native initial data are bound and no native run is attempted here.

Negative probes cover capture count, seed index, boundary and FD policy drift,
a foreign equation binding without Program mutation, and subcycled execution.
A modified reaction is fully resealed against its local request identity and
passes the local request validator, then fails against the registered original
physical equations during public resolution/emission. Separate seeds preserve
the equation identity and change only initialization identity.

## Reviewed native path

The emitted synchronized driver gathers each prepared level, calls the root
composite solve once outside a level-zero checkout, observes every level,
publishes staged auxiliaries, and only then reconstructs/commits each level.
There is no independent per-level Newton solve. Gathered captures and seed are
read with `hierarchy_field_scratch(..., false)`, preserving the actual inputs.
The core deep-copies capture towers; the prepared provider and immutable
invocation strings/vectors are strongly retained. Real attempt and per-level
leases, runtime topology/materialization epochs, canonical stage fraction,
actual boundary-evaluation points and exact graph/equation/capture contracts
remain checked before consumption and pre-Accept.

The real original provider apply composes signed full D rows without inversion,
with arithmetic original faces, conservative FAC flux mismatch, owned coverage
and physical measures. Original scalar/diagonal arithmetic selection and ghost
synchronization changes were inspected against the legacy default paths.
The real full-tower Newton/GMRES and original residual recheck remain the core
consumer. The generated callback checks finitude inside the voted local phase;
mutable callback outputs are authenticated before Newton reductions.

`stage_original_field_candidate_collectively` validates admission/report
consensus, authority, candidate layout, synchronization and finite publication
images before reserving publication. Its actual SolveOutcome retains the
provider and authority callback; Accept checks authority again before the
existing atomic image publication. Rejection releases the reservation. This is
structural reception of the actual code, not an MPI fault-injection run or a
proof of checkpoint/restart behavior.

## Receipts and exact legacy comparison

- 16 new public source/math tests PASS in 110.40 seconds.
- All 8 author AMR source tests PASS on the pinned source. The same first run
  had 13 failed independent fixture constructions, retained in its full log;
  those failures were corrected in the fixture, not production.
- 37 independent host tests from `c8907cd` replayed using the candidate's
  actual headers: PASS in 5.00 seconds. They cover canonical points,
  owner/attempt/capture leases and revocation, callback-output/collective phase
  ordering, active coverage/measures, conservative flux mismatch, and signed
  matrix composition. Those host tests explicitly substitute serial lanes,
  kernel traces or a modal scalar backend where stated in their sources.
  This is not native MPI execution.
- A fresh Uniform parent/candidate comparison uses the exact same physical
  fixture file and callsite. Full emitted C++ SHA, IR SHA and all three Module
  hashes match exactly. Complete receipts, including source/package provenance,
  are in `sol61_amr_public_legacy_parity.json`; no normalization of these hashes.
- Ruff and Git diff checks PASS. No new native compilation, JIT, installation,
  SDK/environment mutation or MAIN edit was performed. Host probe translation
  units are the bounded replay of the previously reviewed test mechanism.

Commands (from the exclusive checkout):

```sh
env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 "$PYTHON" -c \
 'import sys;sys.path.insert(0,"python");import pytest,pops;print(pops.__file__);raise SystemExit(pytest.main(["tests/review/test_sol61_amr_public_original.py","-q","-p","no:cacheprovider"]))'
env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 "$PYTHON" -m pytest \
 tests/review/test_sol61_amr_original_{collective,flux,matrix,authority,point}.py -q -p no:cacheprovider
env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 "$PYTHON" tests/review/sol61_amr_public_legacy_parity.py "$PARENT_SOURCE"
env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 "$PYTHON" tests/review/sol61_amr_public_legacy_parity.py "$CANDIDATE_SOURCE"
```

`PYTHON` was `/Users/romaindespoulain/miniforge3/envs/pops-api040/bin/python`.
Source was explicitly prepended before importing PoPS. Parent source was the
private `PoPS-sol61-amr-nonlinear-review` checkout at `c8907cd` (production code
exactly the `85a79b6f` parent). Complete local logs are preserved in the untracked
`outputs/sol61-amr-public-review/` directory. The host files are already delivered
by `c8907cd`; this patch delivers only the new public tests, runner and report.

## Remaining qualification

The author's public native fixture has constant solved fields with spatially
varying captures. It can receive coupling and transactions once genuinely run,
but diffusion vanishes on its target and therefore it cannot by itself receive
nonconstant coarse/fine flux action. The separate native nonconstant composite
operator fixture remains pending central execution. No native AMR/MPI/GPU solve,
nonlinear convergence campaign, complete restart/rollback/retry, partial-topology
public saved-state scientific reception, M14, M27, or continuum convergence is
claimed by this review. Boundary realization is periodic or homogeneous Neumann
on synchronous Cartesian AMR, excluding EB/shared-face/subcycle extensions.
