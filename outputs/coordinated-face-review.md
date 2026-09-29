# Coordinated face version 1: source and host delivery

Author: GPT-6 Astra/high. Independent contract/review and distinct three-state
oracle: GPT-6 Sol/high. Main implementation commit: `e0823a5`, based on M07
science/counterexample `68c6958` and production `3bd6ea9`.

The separate AMR patch depends on the standalone source-route fix `bf4f175`
(local cherry-pick `5cf2788`). The integrator already has this dependency;
do not cherry-pick it again. The new AMR patch modifies only the generated
AMR block header and its existing unit test file, besides this receipt. It
does not edit `amr_system.cpp` or replace the standalone acceptance evidence.

## Actual checks

Source Python uses `env -u PYTHONPATH`, the isolated `pops-api040-c11` interpreter
and `pytest -o pythonpath=python`; this is explicitly a checkout import, not
an installed-package receipt.

- New `test_coordinated_face.py` plus existing M07, SymbolicPath, path
  consistency and path parameter-routing files: **34 passed in 12.51 s**.
  This includes the new ten face checks at that revision. A subsequent
  M07-only run after adding separate boundary checks: **16 passed in 0.36 s**.
  These counts overlap and must not be added as a unique suite total.
- An earlier old-User/joint-stencil regression run plus the first six new face
  checks: **41 passed in 30.37 s**. Later changes were confined to parameter
  authentication, scientific receipt/test code and AMR route handling; this
  older count is not a claim of a final full suite.
- The positive RuntimeParam capture first failed on authored-vs-resolved handle
  comparison. The corrected lowering authenticates canonical exact declaration
  identities and the resolved owning block; positive and foreign-capture cases
  then passed. Raw failure retained locally in
  `outputs/coordinated-face/capture-red.txt`.
- Two durable host tests compile real generated `CompositeModel` bricks and
  execute their actual policy adapter under C++20/O2/strict floating flags:
  legacy version zero selects PathRusanovFlux; coordinated version one selects
  CoordinatedFaceFlux. The latter checks incident momentum outputs .405/.32
  and negative-depth DomainFailure. These passed as part of the ten face tests.
- A separate TU instantiated the complete generated System and AMR preparers
  for the real Dim=1 M07 composite, using the main Ninja SDK/MPI flags with the
  isolated include tree: **syntax-only exit 0**. It was not linked as a PDE
  executable. Local source/command/log: `outputs/coordinated-face/routes.cpp`,
  `routes-command.txt`, `routes-syntax.log`.
- Three exact gtest bodies extracted from `test_generated_amr_system_block.cpp`
  into a small host executable: **3/3 passed** at O2. New syntax admission is
  separated from exact body authentication; wrong digest, absent typed
  authority, malformed/unknown version, reconstruction mismatch, foreign
  policy and positivity floor are refused. Existing source_stencil/source_face
  syntax and typed checks remain green. Local command/source/log:
  `outputs/coordinated-face/amr-auth-command.txt`, `amr-auth.cpp`, `amr-auth.log`.
- **Six** M07 native tests collected, N=40/80/160 × two component orders.
  None was executed here. They require rebuilt installed genuine Dim=1.

## Boundaries and remaining receipt

M07's erf cell-average oracle, FE/HLL schedule, T=1 and threshold 1e-12 are
unchanged. For each native Dirichlet mirror boundary the fixed face value is
the average of the initial interior cell and the analytic exterior ghost
average. This reproduces that ghost at equilibrium up to floating rounding;
away from equilibrium it is a different closure. The receipt is therefore
explicitly the fully wet lake variant, not a perturbation/front proof.

No installed PoPS spatial, AMR, MPI, restart or performance result is claimed
by this delivery. The signed face kernel uses existing atomic path storage,
stage/halo and reflux contracts; complete tuple failure precedes publication.
Its numerical speed is an author obligation for the whole split method, not
a theorem based on the spectrum of DF+B. General high-order stencils and
wet/dry guarantees are not inferred from this first-order implementation.

Sol's separate nonintegrable three-state 30/70 witness, source permutation
test and future installed/rebind reception live in commits `5e48c6a`,
`7c526fa`, `ba1ded7`, `df85e2f` in his isolated branch. The integrator should
receive those independent tests separately and execute them on a matching
rebuilt Dim=2 package before interpreting them as native evidence.
