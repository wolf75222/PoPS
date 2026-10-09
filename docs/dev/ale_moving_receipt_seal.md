# Assemble owner-pinned ALE evidence without modifying donors

`tests/review/sol61_moving_receipt_seal.py` is a stdlib/NumPy evidence helper.
It imports the independent offline oracle's evidence primitives, never PoPS or a
native module. It does not run a simulation, a compiler, or the physical oracle.
The schema comes from `sol61_moving_interval_offline_oracle.py --describe-contract`.
The helper's `--template` prints an observed launch template with pending owner
authority; it does not authorize the observed hashes itself.

The external owner verifies the identity, source inventory and JUnit hashes in
that template and sets `authority` to `external_owner` in a separate owner file.
The helper requires those explicit pins. It accepts only the exact eight ALE
JUnit nodes with no failure, error or skip, and follows only their absolute
`moving_receipts` properties. It requires the complete 14-case campaign,
78 phase snapshots and 50 scientific NPZ publications. Paths and symlinks that
leave the launch directory are refused. Outputs must be new and outside the
donor launch; the helper never edits a donor or replaces an earlier output.

Each phase records the actual receipt/state/checkpoint path and observed SHA256;
checkpoint SHA256 must match the actual phase receipt. Artifact, point phase,
platform, geometry identity, frame, dimension and communicator size are checked
against the JUnit inventory. The installed source-files image is pinned and
checked against the installed identity's hash/count and clean-source digest.
Each cached compile sidecar is checked against its actual DSO bytes, with an
independent deterministic CBOR implementation of binary/artifact identities.
No DSO is loaded. Orphan binaries/sidecars are refused.

The serial e49 campaign was inspected read-only: its exact JUnit yielded
14 cases, 78 snapshots, 50 scientific outputs and 22 authenticated DSOs. This
was an inventory check, not a new scientific reception. Its launch did not keep
emitted C++/compile-command receipts or a public artifact bundle-to-component
mapping; the output states that gap explicitly. Installed source proof plus
sidecar/DSO identity does not prove that missing mapping. No new compiler or
physics qualification is inferred from cache filenames.

The MPI e49 launch with 6 passing and 2 failed nodes per rank is refused. It
must not be used as a positive campaign. `--rank` chooses and authenticates a
JUnit lane; phase files remain the genuine rank-zero gathered publications and
checkpoint rank images, not synthesized per-rank replicas.

Example owner workflow (run from the private checkout, use an unused output):

```sh
TASK_PY=/Users/romaindespoulain/miniforge3/envs/pops-api040/bin/python
ALE_LAUNCH=/absolute/path/to/installed-public-ale-canonical-relative-amount-dim1-sdke49c-20260930
rtk proxy env -u PYTHONPATH "$TASK_PY" tests/review/sol61_moving_receipt_seal.py \
  --launch-dir "$ALE_LAUNCH" --rank 0 --template > /fresh/external/owner-candidate.json
# Owner verifies the template and records external_owner in a separate owner file.
rtk proxy env -u PYTHONPATH "$TASK_PY" tests/review/sol61_moving_receipt_seal.py \
  --launch-dir "$ALE_LAUNCH" --rank 0 --owner-pins /external/owner-pins.json \
  --output-dir /fresh/external/ale-inventory
# Owner seals the produced oracle-pins.json externally before genuine reception.
rtk proxy env -u PYTHONPATH "$TASK_PY" tests/review/sol61_moving_interval_offline_oracle.py \
  --pins /fresh/external/ale-inventory/oracle-pins.json \
  --output /fresh/external/ale-independent-reception.json
```

`oracle-pins.json` keeps the existing oracle schema and all its original guards.
`inventory.json` remains `pending_owner_reception`. Only the separately invoked
oracle may produce a physical PASS. Preserve its pins/report hashes outside the
donor launch. For a future complete MPI launch, use `--rank 0` with its `rank0.xml`
(or another explicit rank lane); no failed lane becomes eligible through rank
selection.

Author checks: helper authority/path/JUnit negatives plus unchanged offline
oracle contract tests, 40 PASS; Ruff and diff checks PASS. Synthetic XML in the
unit negatives is explicitly software-only and never supplies physical states
or a positive scientific receipt. No SDK, native build or donor was changed.
