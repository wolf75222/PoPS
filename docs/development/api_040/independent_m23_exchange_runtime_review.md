# Independent review of the M23 matrix/ledger fixture

Reviewed `3b25472`, followed by the public-initial-values repair `a52e5809`.
This follow-up changes tests/example binding capture only; no production
operator, equation, time step, or threshold changes.

The mathematical check is coherent: D=vv^T for v=(1,2,-1) is PSD of rank
one, the declared R is skew, and the rate uses the two separate positive
occurrences with weights 0.5 and 1.5. The independent array oracle applies
the periodic centered Laplacian and the SSPRK2 polynomial. It checks the
native component permutation, each face flux, orientation, multiplicity,
measure and accepted temporal weight, then compares accumulated cell
amounts with h times the actual state change. This is a discrete matrix
test, not a claim that the array initialization represents a particular
continuous cell-average profile.

The finite 1e308 perturbation reaches the real constitutive evaluation and
overflows there; it is not a nonfinite bind input. The two refused attempts
check all-rank RuntimeError family/status, unchanged time/step/state and
unchanged exchange mailbox. The test does not infer GPU or general AMR
behavior from this periodic Uniform Dim1 case.

Resolved fixture weaknesses:

- `a52e5809` already repaired the initial-condition authority conflict by
  supplying the exact bound subject through `initial_values`. The earlier
  native failure was at bind, so it did not qualify any numerical step.
- Case construction/resolution and initial array preparation now converge
  local errors before peers can enter collective compilation or binding.
  The bad-state allocation is likewise completed in a collective check.
- The stage oracle formerly associated initial/predictor with the first
  and second contexts encountered. Native insertion order currently made
  this work, but it did not verify stage identity. The test now explicitly
  maps the retained `StagePoint(name='ssprk2_stage_0/1')` identities and
  refuses unknown, missing or duplicate stages, independently of ordering.
- Record component/cell/side bounds and exact occurrence-parent identity
  are checked. With unique keys and the existing exact record count, this
  closes missing/replaced-address ambiguities in the ledger oracle.
- MPI2 checks actual gathered batch sizes to establish an empty peer and
  a real owner, instead of assuming that the chosen small grid did so.
- The example's immediate bind lambda now captures `subject=subject`,
  fixing Ruff B023 without changing binding semantics.

Review evidence: **5/5 pure tests** exercise the exact fixture stage helper,
including reversed context order and refusals. Both native test variants
collect successfully, Ruff and changed-file diff checks pass. No native
simulation, JIT or compiler was launched by this review.

The parent separately executed `installed-m23-native-bind-repaired` at
source `a52e5809cc65df504a9c8c75e65e6471f993e927`: 3/3 pytest cases pass
(one Hall example case running five trajectories, two matrix variants).
Native Dim1 SHA is
`42436a0ccef7c5504273c9e459f4bd946652597aa6ee025bc484ac595ad7ca02`,
headers `62398f3c13c193eb48db07518735fe755e6d108821adffc37acccd8fcea290eb`.
I independently reopened its two real `accepted_matrix.npz` and
`accepted_ledger.json` files and replayed the flux/amount checks using the
new stage mapping with **reversed** context order:

| Variant | Records | Max face error | Max ledger/state-increment error |
|---|---:|---:|---:|
| canonical | 384 | 2.3092638912203256e-14 | 1.2734293329021151e-17 |
| permuted (2,0,1) | 384 | 1.4210854715202004e-14 | 1.2751233987966237e-17 |

These remain below the unchanged 3e-12 and 3e-13 thresholds. File hashes,
in canonical/permuted order, are:

- ledger: `37896e9c3e283acdb3c363fea5156f001b4e6b0cbe1f5ee4f7e3683b60c340c7`,
  `9ca8d17ce4b1983143c5633f52fab711c61ce5836f1195f77bd0710b37a7fc61`;
- NPZ: `ecd092de225e84100fb0686ca98d260960b2f48d98eb1063cd3dbe7693f9a678`,
  `083cb3fc26fbef2fed888d3d56643a6b51f3179b3e7ddd5d72075311f0fd01c7`.

The parent's run predates this fixture follow-up. The follow-up's actual
serial/MPI2 execution remains the next central reception, especially the
new empty-peer assertion. No production blocker was established in this
bounded review.
