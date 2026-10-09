# Original FieldProblem candidate diffusion: source reception boundary

Production source is frozen in `48292403c0f42192e17c61a3be9f97b3dd2d3c19`
(parent `7d91f5ab06eca3db1d32f2c0164359a5426c2c9c`), followed by
`2e9a07329d008cf910f442998e54f5a47ed6e6d4` (private construction drain).
`5394fda54aaf9623f784a74703da69635f4d2dae` adds the independently
authenticated TrueCorrectionResidual criterion requested during review.
There are sixteen production files: nine Python files and seven existing SDK
headers. This note and its two test files are a separate commit. No native run,
SDK installation, JIT, MAIN mutation, or environment mutation was performed by
this author. Root owns compilation, installed reception and integration.

## Physical body, method, realization and acceptance

The physical model still declares the original ordered equations before Program:

    F(q) = -div(D(q, exact State captures) grad(q)) + R(q, captures) - f

Program binds the original FieldProblem, exact State captures and seed. It does
not insert another equation, materialize a uniform global quantity, recognize a
model name, freeze D at the seed, or replace the Program with a host solve.
Select the constitutive evaluation and face discretization explicitly:

```python
CellCenteredNonlinearCoupled(
    finite_difference_step=original_step,
    face_policy="Arithmetic@1",
    coefficient_evaluation="PerCandidate@1",
)
```

The method has contract `pops.spatial-field-residual@3`; the evaluation policy is
`pops.field.coefficients.per-candidate@1`. Its exact physical diffusion AST,
ordered unknown product, qualified capture bindings, point, registered physical
problem and registered method are authenticated again during resolution and
emission. A coherent re-digest of a foreign solver-owned AST cannot replace the
registered physical body. A deferred coefficient descriptor cannot escape into
a material field consumer. The descriptor carries the real AST and authority;
it has no D=0 physical interpretation and emits no coefficient kernel before F.
Initialized native storage is allocation storage until the actual body is run.

Arithmetic@1 remains an explicit signed arithmetic face discretization. Finite
negative, singular and nonsymmetric coefficient matrices are admitted; this is
not an SPD certificate or a convergence guarantee. All original Newton/GMRES
controls, physical boundaries, central FD step and terminal original-F criterion
are retained. The request schema is conditionally 2 and Program IR conditionally
11. Literal @1, captured @2 and SpatialBasisJacobi legacy profiles retain their
previous request/IR images.

`SpatialBasisJacobi@1` is a frozen linear spatial-operator realization. The
specific PerCandidate/Jacobi combination is refused before a kernel; A(e)-A(0)
is not silently relabelled as a nonlinear Jacobian. Identity realization remains
available. No mathematical restriction on other future nonlinear
preconditioners is introduced.

## Native execution and AMR order

Uniform owns a separate coefficient field, trial field, frozen capture fields
and a real PreparedSpatialResidual/FieldNewtonKrylovWorkspace. The actual
Kokkos constitutive body writes all m² entries at every F evaluation, prepares
coefficient ghosts, applies the existing arithmetic stencil, then adds the
unchanged original local body once. Central JVP invokes that same F at q+h v
and q-h v. The emitted terminal recheck invokes the complete F again before
SolveOutcome consumption/publication.

True AMR uses the existing composite field provider, coverage, measure, owned
cells, coarse/fine schedules and lane. Its optional capability is
`pops.hierarchy.original-field-candidate-operator@1`. One private apply-only
resource is built per invocation and retained by the original residual. It
contains independent FAC entries/coefficient/phi/image and transfer storage;
it does not create another GMRES basis. Each F is the following composition:

1. Deep-copy the entire trial q tower and synchronize it, restricting fine q
   into covered coarse cells and preparing the existing q halos.
2. Evaluate the actual D(q, frozen exact captures) body on every stored valid
   cell, with no physical coefficient evaluated from a prototype.
3. For each coefficient entry, descend from the finest level to level 1 and
   invoke the real FAC `Connection::restrict_into` coefficient transfer into
   covered parent cells. Then extrapolate valid D, exchange same-level/periodic
   ghosts and apply the existing coefficient physical boundary closure.
4. Apply the existing arithmetic composite flux and reflux to that same q.
5. Add the unchanged original local body on the synchronized q tower.

This order is part of @3. In particular it computes restrict(D(q_fine)) on a
covered parent, not D(restrict(q_fine)) as a substitute. The existing @1/@2
preparation keeps `restrict_covered=false`; only the private @3 resource passes
true. Central JVP and the terminal original recheck differentiate/re-evaluate
this entire composition. Native covered rows and active measured norms remain
the provider's existing authority; this extension introduces no new EB or
non-Cartesian geometry capability.

## Authority, exceptions and publication

AMR authenticates original provider ownership, retained attempt leases,
synchronized level attempts, exact ordered points, capture identities and
widths, topology epoch, materialization generation and the prepared execution
lane. The original provider preparation generation is frozen. The private
coefficient preparation generation advances separately and must match on each
subsequent evaluation/consumption. Layout/component/pointer replacement,
foreign lane, changed epoch or revoked attempt refuses before candidate access.
The lane argument is compared and refusal voted on the provider's authenticated
lane, including a rank-local foreign argument.

Uniform captures its real prepared lane, attempt and evaluation point. The @3
allocation/body/reduction phases catch local exceptions, fence even after a
failure, and vote before the next collective. Its opt-in PreparedSpatialResidual
and FieldNewton guards default to false for historical callers. Guarded spatial
solve refuses a foreign lane on its captured lane. Provider callbacks which
already own collectives are not enclosed in another local-only wrapper. Private
FAC/composite construction also drains earlier launches before its allocation
failure vote.

The existing all-level SolveOutcome stage/validate/accept protocol remains the
publication authority. Coefficient resources are private attempt resources;
solving alone does not publish an accepted field. A publication validation
refusal leaves the staged candidate retryable. An attempt rejection revokes it,
and a retry obtains a new prepared resource. No report is changed to manufacture
acceptance.

Uniform @3 explicitly declares `pops.field.linear.true-correction-residual@1`
(TrueCorrectionResidual@1). This is independent of `guard_local`: its separate
`verify_true_correction=false` native default preserves historical callers.
After every GMRES correction, including a projected-converged/happy-breakdown
cycle, @3 computes the actual full JVP(correction) and the norm of rhs-Jδ. A
projected small residual alone cannot return a converged linear result. If the
true norm exceeds the unchanged linear threshold, restart continues within the
unchanged column budget or the solve returns its structured refusal. The
criterion is declared in method metadata, source contract, solve request,
registered resolution and scratch cost; the @3 generated Uniform constructor
passes the independent flag explicitly. The existing AMR full-correction path
already has this criterion and its @3 invocation authenticates the same identity.
The final original-F check is still mandatory after numerical convergence.
The historical projected-only branch is preserved only when its flag is false.

## Cost and source proofs

Each F adds constitutive work proportional to physical AST work over m² stored
entries, followed by the real existing coefficient transfers/halos and composite
flux. A central JVP costs two complete F evaluations. Every GMRES correction now
requires one complete JVP and true residual norm under the named @3 criterion,
including projected convergence. The terminal physical
recheck costs one more complete F. Additional coefficient and private scalar FAC
storage scale as O(m² * stored cells), plus transfer schedules/buffers and an
owned synchronized q tower O(m * stored cells). The original provider and the
original residual retain their existing allocations. The private @3 evaluation
adds no second Krylov basis; it is not a second solve. Scratch metadata records a
lower bound and explicitly excludes private FAC transfer/storage constants from
that bound. No hidden DOF cap, dense global solve or mandatory basis sweep is
introduced by this option.

The exact Fraction counterexample is the previously frozen
`sol61_candidate_diffusion_counterexample.py`: two periodic cells, h=1/2,
q=(1,2), D=1+q² give F=(-28,28), DF·(1,0)=(20,-20), whereas frozen-current-D
JVP gives (28,-28). Central complete-F FD gives 20+4 ε². For fine q=(1,3),
D(mean q)=5 while mean(D(q))=6. These are exact algebra discriminants, not
runtime or continuum qualification.

The Python source suite covers Uniform widths 1/3 and true AMR widths 3/5,
permuted ordered products, capture/point/unknown/policy/descriptor mutations,
coherent foreign re-digests, explicit Arithmetic and incompatible Jacobi refusal.
A pure host test compiles the exact existing local-phase helper with only
Kokkos/lane/vote seams: normal, launch-error, fence-error and legacy schedules.
It checks local/fence/vote order; it is not an MPI runtime simulation.

Three genuine native C++ fixtures are added to the existing discovered fragment:

- `CandidateDiffusionReevaluatesFullFJvpAndRecheck`: nonconstant signed/singular
  three-component D(q), N=8/16 and two permutations, real composite operator,
  JVPs, original recheck, stage/refused point/retry/accept. Its load is manufactured
  with the real discrete operator, so it does not qualify an independent
  continuum/FV manufactured-solution campaign.
- `CandidateCoefficientLocalFailureVotesAndCannotPublish`: rank-local wrong
  lane/epoch/storage, coefficient allocation/launch/nonfinite faults, invisible
  candidate and unchanged provider solution, revoked lease and new attempt.
- `CandidateCoefficientsRestrictFineValuesBeforeHaloFlux`: the real native
  coefficient transfer port must change covered D from 5 to 6, with genuine
  coverage and generation checks.

Root must reconfigure discovery and execute these names serially and through the
whole CompositeGeneralField MPI wrapper. This author only syntax-compiled the
fragment and real Dim2 candidate templates. Installed public Uniform/AMR runs,
MPI convergence/rollback, GPU, checkpoint/replay and physical campaigns remain
unreceived here. M26/M27, H05, Marshak and moving-wall campaigns are not qualified
by this mechanism.

## Reproduction and receipts

Use source Python with its source root inserted explicitly; do not install or
JIT for these source checks:

```sh
env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 /path/to/source-python -c \
 'import sys;sys.path.insert(0,"python");import pytest;raise SystemExit(pytest.main(["tests/review/test_sol61_candidate_diffusion.py","tests/review/test_sol61_original_captured_diffusion.py","-q"]))'
```

Fresh parity uses the new test module's `legacy_images()` with a fixed shared
fixture/test file path and two separate exported Python source roots (parent
7d91 and final 482). Compare the full JSON images with no provenance stripping:
IR/body/hash, emitted CPP hash, all three Module hashes, full manifests and full
SolveRequests. Four profiles @1, captured @2, Jacobi and captured-AMR-Jacobi are
byte-identical; their IR versions are 8/10/9/10. The original receipt comparison
at identical callsites has common SHA256
`fc88edf36c43fde8f56ccd857ac3654201c38c00c61c4f2fc93ef1e4133a01e2`.
The fixed-callsite script and images are private external receipts under /tmp;
`legacy_images()` provides the checked-in re-run mechanism. A different callsite
is allowed to change provenance and is not a reason to normalize it away.

The older c28 test `test_no_physics_or_numeric_controls_changed_in_emission_source`
still compares the entire emitter suffix to c7. It is preserved and demonstrably
red for the legitimate new conditional @3 branch. No scientific tolerance or
old test was relaxed. Full fresh legacy images above replace neither historical
red evidence nor native reception; they prove byte preservation of the selected
historical programs more directly than an unconditional source-text assertion.

Validation receipts: source/math initial 45 PASS, plus the two additional guard
and exact-host schedule checks PASS; final source replay at 482 passed 47 tests in 412.91 seconds. Five targeted
TrueCorrectionResidual/guard checks PASS after the follow-up; final whole-suite
51-test replay on 539 PASS in 414.43 seconds (external JUnit receipt
`/tmp/sol61-candidate-freeze-5394fda5-source.xml`). Ruff, nine Python ASTs and diff-check PASS. Actual Dim2 syntax
PASS covers native Uniform primitives/workspace and actual AMR templates/three
fixtures. A separate real Uniform @3 emitted CPP syntax check also PASS. The
source/headers checked by syntax matched all sixteen frozen Git blobs.
Native ABI and wire ordinals are unchanged; the existing public header manifest
schema remains, but actual SDK header hashes/signature change. All native
installed dimensions and generated DSOs must rebuild before a runtime claim.
