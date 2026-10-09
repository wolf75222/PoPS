# Independent source/host review: replicated accepted currents

Target: `1f777538f852c27e5de369fca8961d27f123cbcf`, parent
`2c0e45f2db46188f53ee429553ccbb531e5dd7d7`.
Review branch: `codex/api040-sol61-replicated-exchange-review`. No MAIN,
installed package, SDK, native extension or JIT cache was changed. No native or
MPI execution was performed by this reviewer.

## Historical native failure, independently decoded

Input: `outputs/installed-integral-transport-amr1-mpi2-abi5-307-isolated-20260930/`
`rank0-tmp/test_native_accepted_face_amou0/accepted.npz` in the workspace.
The independent `checkpoint_exchange_images` reader fully consumed both
POPSEX02 images. Both are 323875 bytes, bit-identical, SHA256
`dad50f7c904dcc00da8134e65ab3c9a9c4cd85eb66370d70ec613736579da9be`.
Each contains 256 records and 8 exterior x+/component0 records, whose independently
summed orientation × measure × flux × temporal weight × multiplicity is `-0.01`.
Both persistent quantity pairs are `(0.7, 0.72)`. Thus the saved evidence matches
two copies of one physical current; a single contribution would yield `0.71`.
These are **pre-fix SDK307 receipts**, not reception of the corrected native code.

## Findings on the fix

No new production defect was demonstrated in this bounded review. The helper
returns true for every distributed lane rank and only lane ordinal zero for a
replicated carrier. It uses the actual field distribution, not a model name,
rank-global heuristic, face count or geometry mask. Its placement preserves both
mask authorities and every batch collective:

* Transport refreshes the active mask, reads already prepared finest-owner
  coverage, computes the owner predicate and validates both local layouts before
  the preparation-error and layout-error votes. Only the producer lambda returns.
* PreparedDiffusion does the same with its actual `variable_` and prepared lane;
  its prior `explicit_frequency()` preparation remains before the ownership gate.
  Generic CoupledGradient uses this prepared diffusion producer.
* Uniform and AMR contexts always call `prepare_exchange_batch` then the native
  `stage_program_exchanges`, even for an empty record vector. Preparation executes
  `collective_step_rejection_phase`; staging always converges its local exception
  and restores the old prefix on collective failure. Neither boundary has an
  empty-vector fast return. The helper adds no collective.
* The guard precedes all incidences/components inside the lambda, so a replica
  cannot retain duplicated interior records while suppressing only its exterior
  records. Coverage/EB, orientation, flux, measure, dt, multiplicity, occurrence
  and runtime-point qualification otherwise retain their prior formulas.

An existing host fixture needed one `using
pops::runtime::program::accepted_exchange_contributes;`: it extracts the method
into a global mock `Producer`, whereas production PreparedDiffusion lives inside
`pops::runtime::program`. Its first compile failure was a fixture namespace
regression, not a production lookup failure. This review commit repairs only that
test scope; no header is modified.

## Independent executable probes

`tests/review/test_replicated_exchange_1f777538_independent.py` reuses only the
existing thin host field/face adapter, then executes the actual transport emitter
and actual extracted diffusion method with its own assertions. C++20 compilation
uses the checkout headers and a temporary directory, without Kokkos/native MPI.

Both producers passed: seven replicas contribute one trace; both components
retain their expected signed amounts; EB and finest-owner masks intersect;
distributed lane rank6 contributes; empty rank0 still enters the batch; Uniform
without coverage is unchanged; covered x+ produces no x+ record. Replicated
nonowners reject foreign coverage, foreign EB and throwing active-mask lookup
before a batch write. A foreign mask on an empty distributed rank also refuses.
The local spies count two mask votes on valid paths, two on layout refusal, one
on preparation refusal, and completion of the batch tail on valid empty paths.
Already staged prefixes remain intact on refusal or suppressed production.

These spies prove local control-flow participation, **not MPI progress or votes**.
The source assertions separately inspect the real collective helpers and both
context forwarding methods. The existing three coverage/accessor host tests also
pass, including fine-only/mixed coverage, fine dt and no collective refresh in
the coverage accessor. Final coherent reception, including context-forwarding
assertions: **6 passed in 6.71 s**. Ruff and whitespace checks also pass.

Exact source-only command, from this review checkout:

```sh
rtk proxy env -u PYTHONPATH /Users/romaindespoulain/miniforge3/envs/pops-api040/bin/python -I - <<'PY'
import pathlib, sys
root = pathlib.Path.cwd()
sys.path[:0] = [str(root / 'python'), str(root)]
import pops, pytest
assert pathlib.Path(pops.__file__).resolve() == root / 'python/pops/__init__.py'
raise SystemExit(pytest.main(['-q',
    'tests/review/test_replicated_exchange_1f777538_independent.py',
    'tests/python/unit/codegen/test_accepted_exchange_coverage_independent.py']))
PY
```

Ruff checks those two files; `git diff --check` checks the delivered changes.

## Other producers and remaining reception

MovingInterval prepares geometry/source/amount records for every local restored
state, including replicas. Its POPSEX03 validator requires those exact local
occurrences and support tuples. These records have `exterior_trace=false` and
cannot be consumed by the IntegralState exterior selector. Applying this helper
there without changing the coupled codec contract would invalidate a replicated
rank's checkpoint. This fix correctly leaves that distinct contract intact.

JointInventory emits amounts from `ctx.sum_component` collective reductions and
does not declare exterior trace support. Its global inventory diagnostic ledger
is likewise outside this patch's local face-current contract. This review does
not establish an MPI normalization theorem for that separate realization; a
future exterior-current producer must explicitly choose ownership and trace
support rather than copying global inventory records into this API.

Root must still rebuild the exact updated SDK/extension, authenticate its native
identity and replay real MPI2 transport and diffusion, distributed rank1/empty
rank and rank-local mask failure, as well as checkpoint/restart with unequal local
record counts and equal persistent integral values. This review does not replace
those tests, native rollback, GPU, dimensions 1/3 or AMR evolution qualification.

Production bytes reviewed (SHA256):

| Source | SHA256 |
| --- | --- |
| accepted_exchange.hpp | `9a5f2acc1ee1f6c1ffb09979381a1d03808c500ea8a1d65ba0c37898ed28b1c9` |
| prepared_diffusion.hpp | `418743a832a2a8a3991fdb1f50b94896b32a9130adf0b88e4365c22bcbdc2b01` |
| program_emit_transport_exchanges.py | `6ef7493658686a35dbafe0b50342625980606ab51fa1663381f7b8320a82034b` |
