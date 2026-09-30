# Assemble actual M18 native reception without self-approval

This helper is prepared from `004a6757b76dceba0e60ff8bf4bf2ff9eb84798c`.
It does not run native code, import PoPS, fabricate NPZs, calculate a positive
physical state, or qualify a campaign. Native Serial/MPI2 reception remains
pending. The original scientific oracle and owner-pins schema are unchanged.

`tests/review/sol61_m18_owner_assemble.py` reuses the existing offline reader's
strict JSON, checkpoint/support protocol, original authored-equation guard,
and compiled IR/expression guard. It keeps scientific accepted-state reception
in the existing `sol61_m18_entropy_offline_oracle.py`, after external sealing.

## Inputs and closed inventory

Assemble each actual Serial and MPI2 campaign separately. Supply one Serial XML,
or exactly rank0 then rank1 XML for MPI2, each containing exactly one named M18
case and no failures/errors/skips, no foreign extra cases, consistent aggregate
counts, unique properties and exact native/artifact/rank identities. The recorded
`saved_receipts` must name the actual data directory. Both outside-cone failure
properties must exist. XML files must be distinct and under the archive root.

The data directory must contain exactly `provenance.json` and the ten original
phases' state/receipt/checkpoint triples: 31 files. Checkpoint basenames are read
from the real receipts, not guessed. Missing/extra files, directories, foreign
phases, reused checkpoints, digest mismatch, path traversal, aliases and symlinks
are refused. The owner-pins manifest has 35 distinct files in Serial, 36 in MPI2
(31 data +3 actual sources +1/2 actual JUnits), excluding the manifest itself.

All supplied paths and roots must be absolute and canonical. Sources must be
under an explicit source root, saved files under the archive root, installed
native/Python/SDK evidence under the installation root, and compiled package/C++
files under the runtime/cache root. Hashes are calculated from actual files;
large binary files are streamed. Original oracle file/decompression limits
continue to apply to saved-state protocol data.

The three executed sources (`fixture`, `example`, `snapshot`) are checked against
`provenance.source_sha256`. Authored three-unknown/original residual/read-only
capture/commit/Newton/duration guards and the independent compiled-expression
guard run before assembly. Actual native records must agree on each rank,
Dim2/capability ABI5, two named System package hashes and package ABI7. Compiled
platform, plan and component evidence must exist. No opaque package filename is
interpreted as a compiled physical-expression mapping.

## Execution-owner metadata and explicit gaps

The fixture records native extension path/hash, but **does not record** the
executed Python package origin or SDK file path. System packages carry named
block/hash/ABI records, but no binary paths. Supply an execution-owner sidecar
captured from the real campaign, stored outside the closed phase directory but
under the archive root. Its exact schema is:

```text
schema: sol61.m18-execution-owner-metadata@1
source_commit: exact executed 40-character source commit
python_package: {path, sha256} of the actual installed Python package entry file
sdk: {path, sha256} of the actual installed SDK identity/support manifest file
native: {path, sha256} of the extension recorded by the native provenance
system_packages: {dual: {path, sha256}, target: {path, sha256}}
generated_cpp: null OR a nonempty list of distinct actual {path, sha256} .cpp files
```

The helper verifies every sidecar file/hash and links extension/System hashes to
the actual rank provenance. Python/SDK origins remain **owner-attested execution
facts**: checking file bytes alone cannot prove which Python package/SDK executed.
External ROOT approval binds that fact, source commit, all pins and sidecar bytes.
No helper-created metadata is treated as independent execution authority.

If generated C++ was not retained, use `null`. The pending record explicitly says
`generated_cpp_mapping: not_stored`. If actual C++ files exist, their bytes are
pinned as owner-supplied actual files; no source-to-expression mapping is invented.
The original scientific expression guard authenticates recorded IR, not absent
C++ source. Missing native execution provenance cannot be recovered from an IR
or from mathematically correct arrays alone.

## CLI and ROOT approval

Choose an output outside the closed phase directory. No command overwrites an
existing file. Let `PYTHON` be the read-only Python with NumPy, and `HELPER` the
absolute path to `tests/review/sol61_m18_owner_assemble.py`.

```sh
env -u PYTHONPATH "$PYTHON" "$HELPER" assemble \
  --archive-root "$ACTUAL_ARCHIVE_ROOT" --data-directory "$ACTUAL_PHASE_DIRECTORY" \
  --source-root "$ACTUAL_SOURCE_ROOT" --installation-root "$ACTUAL_INSTALLATION_ROOT" \
  --runtime-root "$ACTUAL_PACKAGE_CACHE_ROOT" --metadata "$ACTUAL_OWNER_METADATA" \
  --fixture "$EXECUTED_FIXTURE" --example "$EXECUTED_EXAMPLE" --snapshot "$EXECUTED_RECEIPT_HELPER" \
  --junit "$ACTUAL_RANK0_XML" --junit "$ACTUAL_RANK1_XML" --output "$PENDING_TEMPLATE"
```

Omit the second `--junit` for Serial. Output is
`sol61.m18-owner-pending@1`, status `pending_external_ROOT_approval`, with explicit
`native_execution_in_assembler: false` and `scientific_reception: false`. It
contains the proposed **existing** `sol61.m18-owner-pins@1` manifest plus its exact
serialized SHA, origin evidence, roots and limitations. Successful assembly
proves file/protocol consistency only; it is not scientific reception.

ROOT reviews the actual native/JUnit/source/package receipts, then writes a
separate approval file with exactly these fields:

```text
schema: sol61.m18-root-approval@1
status: approved
approved_by: ROOT
pending_sha256: exact SHA of the reviewed pending file bytes
pins_sha256: the reviewed proposed_pins_sha256 from that pending file
```

ROOT must explicitly communicate its approval SHA externally. The helper never
creates approval JSON or approves itself. As with the existing owner-pin seal,
a SHA proves exact bytes, not human authorship; external communication from ROOT
is the authority, not merely a JSON string reading `approved_by: ROOT`.

```sh
env -u PYTHONPATH "$PYTHON" "$HELPER" approve \
  --pending "$PENDING_TEMPLATE" --approval "$ROOT_APPROVAL" \
  --approval-sha256 "$EXTERNALLY_COMMUNICATED_ROOT_SHA" --output "$OWNER_PINS"
env -u PYTHONPATH "$PYTHON" "$ORIGINAL_ORACLE" \
  --pins "$OWNER_PINS" --owner-sha256 "$ROOT_CONFIRMED_OWNER_PINS_SHA" --output "$SCIENTIFIC_RECEIPT"
```

Approval rechecks the closed data inventory, source/JUnit/provenance pins, and
all origin files before writing the exact existing owner-pins schema. Printed
SHA identifies the new file; scientific status remains false until the separate
oracle receives owner-confirmed pins. Keep owner files outside the data directory.
The twelve existing real-file scientific countermodels remain a separate step
after an authentic positive reception, not synthetic protocol unit tests.

## Tests and limits

43 new independent stdlib/NumPy protocol tests PASS; the affected pure contract/math selection totals 77 PASS (one public PoPS source-IR test deliberately deselected). They cover exact Serial/MPI2
JUnit inventory, status/rank/realm/properties; all ten phase names/triples;
foreign/missing/reused/extra files; source/package/SDK/native hashes; path escapes
and parent symlinks; absent generated C++; external approval missing/foreign/stale
byte seals and changed pin image; and a subprocess denying any PoPS import.
Synthetic XML and opaque `NOT-NATIVE` bytes are labeled protocol-only. They do
not reach a full assembled native receipt or scientific oracle success. No
accepted-state NPZ or fabricated positive native campaign was generated.

Command:

```sh
rtk proxy env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 /Users/romaindespoulain/miniforge3/envs/pops-api040/bin/python -m pytest tests/review/test_sol61_m18_owner_assemble.py -q -p no:cacheprovider
```

Ruff/diff checks PASS. Full assemble/approve/physical reception is unrun pending
actual future Serial/MPI2 output. No MAIN/SDK/environment/native/JIT/build mutation.
Both outside refusals must preserve the same impossible read-only target.
`safe_rebind` is a **new bind** of the original interior target using the same
artifact/context and zero seed. Reducing dt cannot cure the outside target;
there is no successful retry claim for that outside runtime.
