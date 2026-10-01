# Independent bind-content reception of 29ce923e

Candidate: `29ce923e008d8506b9e1e28ea5aa00ea74d1d668`. Review is source/host only;
no native loader, MPI process, JIT, compilation, installation into a shared
package, or MAIN mutation. Two files only: this report and
`tests/review/test_sol61_bind_content_independent.py`. No production edits.

The 19-line production change is bounded: `_evidence` selects only the exact
`InstalledComponent` class. Its new projection removes exactly the top-level
residence `path` and adds `schema_version=2`; `to_data()` retains the actual path
for provenance. The evidence still includes component/artifact/binary/manifest
identities, the complete runtime contract and native interface, platform,
entry symbols, loaded state and provenance. An ordinary lookalike or subclass
receives no new normalization. Other evidence and checkpoint identity protocols
remain unchanged.

## Independent host evidence

19 tests PASS (0.42s), Ruff PASS, diff-check PASS. No author test factory/helper
is imported. Tests construct real `ComponentManifest`, generated `Transfer`
interface, `ComponentRuntimeContract`, `CompiledComponentArtifact` and
`InstalledComponent` authorities. Actual `CompiledComponentArtifact.install`
creates three separate content-addressed temporary files and actual
`InstalledComponent.verify` checks their binary identities. Binary bytes are
explicitly SYNTHETIC test data; only symbol inspection has a host seam. These
files are never presented as native DSOs or scientific outputs.

The complete real `InstallPlan._payload` executes against a controlled receiver
record. Three rank-local residences produce exact equal canonical bytes and bind
identities. Eight independent changes (manifest/binary/artifact identity,
symbols, origin, runtime contract, platform, loaded state) each change that
identity. Actual file verification rejects changed bytes, missing files and a
foreign relocation. A fake `bind_identity_data` protocol cannot opt into the
new projection; neither can a subclass.

The true complete Uniform `checkpoint-capture-plan` projection is extracted
from `_SystemIO._prepare_checkpoint_capture`. All remaining runtime metadata are
explicit identical inert seams, and the bind comes from the true complete
InstallPlan payload. The real `collective_checkpoint_capture` preflight executes
with three ordered transport rows: relocated same-content binds enter the next
phase; one divergent origin blocks EVERY rank before its capture callback.
Callbacks deliberately stop before any native call or publication. No fake
successful checkpoint is emitted. The full parent `_evidence` function is loaded
from `29ce923e^`; component-free canonical evidence matches byte-for-byte without
normalization. This does not certify full bind constructor/native admission or
all possible legacy object graphs.

Content evidence is an identity projection, not a file verifier. File authority
is still enforced by actual install/load verification and the existing
`InstallPlan.__post_init__` checks (exact InstalledComponent, matching artifact,
loaded handle and `installed.verify()`). This review does not claim that hashing
alone authenticates fresh bytes, or that a pathname describes executed content.

Reproduction, explicitly selecting the private source checkout:

```sh
env -u PYTHONPATH /Users/romaindespoulain/miniforge3/envs/pops-api040/bin/python -c \
'import sys,pytest;sys.path.insert(0,"/Users/romaindespoulain/dev/tmp/PoPS-sol61-bind-content-review/python");raise SystemExit(pytest.main(["-q","tests/review/test_sol61_bind_content_independent.py"]))'
env -u PYTHONPATH /Users/romaindespoulain/miniforge3/envs/pops-api040/bin/python -m ruff check \
  tests/review/test_sol61_bind_content_independent.py
git diff --check
```

No new candidate blocker is demonstrated in this bounded scope. Genuine MPI
bind/checkpoint reception remains ROOT's responsibility.

## Frozen Uniform fixture and actual red reception

ROOT fixture `01cc51b6127d20ab37c551ca14f4e38611b95add` reads the real Uniform
`capture_auxiliary_checkpoint_accepted_state` getter. Its bytes replace an AMR-only
rank-local manifest accessor. `_system_io.py` uses the same native getter for the
checkpoint and requires exact bytes with POPSAUX2 prefix. Native
`System::capture_auxiliary_checkpoint_accepted_state` prepares its lane and sealed
registry/storage image. This is an actual auxiliary codec, not a reconstructed
carrier list.

Host tests extract the frozen fixture's real `capture`/`same_images` functions.
The observation seam has different q1/q2, all six q0/q1/q2 × slot0/1 getter calls,
and sealed-image getter bytes. Latest solution is read from slot1. Full history
bytes, publication identities, durations, fill and lifecycle participate in the
comparison. Test bytes are labelled synthetic and are not validated native
POP SAUX payloads. A diagnostic observation mismatch is preserved as a failure,
not silently accepted by a test.

Actual ROOT campaign (read only):
`/Users/romaindespoulain/dev/tmp/pops-api040-native-reception-evidence-20261001/captured-diffusion-history-v2-uniform-serial-dim2`.
Both cases compiled and solved, then failed `same_images(reloaded,accepted)`:
reloaded diagnostic tuple was empty while accepted contained five
`field_residual_4` diagnostics. Scalar had 189 full residual evaluations/92 JVPs;
coupled had 273/134. The log reports 2 failures, 169.29 s.

Actual log SHA256:
`55448ddddc5215b313b0201d8c331fe4a0a4ba6381679d2eec2afe601f5c9720`.
Actual JUnit SHA256:
`624be4ec68343fcff786f3bf71cf96c6cdc3c2dbe792883a7f338ef3e3e52600`.
These are red evidence, not approval.

Diagnostics are included in ProgramRuntimeState accepted snapshots/rollback,
but the current Uniform checkpoint serializer does not serialize the generic
`program_diagnostics()` map. Therefore this failure establishes an unsupported
fixture claim about post-restart observation equality; it does not establish
which scientific continuation bytes differed. No codec extension or production
patch is made here. ROOT must either implement actual persistence or bound its
fixture qualification and preserve separately the true diagnostics. The
independent @2 reader already reports absent in-memory diagnostic comparison
images as a gap.

No closed two-case inventory, pending positive pins or approval is produced from
this red dataset. Existing pending provenance remains unapproved. A new green
campaign is required before scientific reception; no CPP-to-DSO mapping is
inferred from filenames, and no arbitrary captured-D/AMR/full-family claim is made.
