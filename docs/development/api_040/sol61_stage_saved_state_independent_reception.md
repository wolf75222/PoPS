# Independent saved-state reception of uniform EvolvedStage

The reader `tests/review/sol61_stage_saved_reception.py` uses NumPy, Fraction and
stdlib only. It never imports PoPS, a native extension, the author MMS helper,
or the author `check_saved` implementation. It reuses only independent file,
NPZ and identity codecs from the preceding saved-state readers and the
independent `sol61_stage_checkpoint_lineage.py`. Optimized Python is refused
because that checkpoint codec uses assertions.

Qualification is `uniform-original-stage-saved-state-equations@1`: eight finite
uniform periodic witnesses, N8/N16, dt .01/.02 and either one Q/one unknown or
two distinct Q carriers/three unknowns. It is not a certificate for AMR,
Marshak radiation, continuum convergence or an entire nonlinear family.

The independent formulas are

- Q=T+T² in the scalar case;
- Q0=T0+T0²+.1T1² and Q1=T1+T1²+.2T0T1 in the coupled case;
- z=.25T0+.5T1, with zero diffusion in the auxiliary row/column;
- D=[[.012,.002],[-.001,.014]], signed and nonsymmetric, multiplied columnwise
  by 1+.4Tj² in the coupled case;
- F=Q−dt*(Div(D(T)Grad(T))+forcing)−Qn.

Each oriented periodic face is visited once, with an arithmetic mean of the
candidate coefficient and equal/opposite finite-volume contributions. x is
array axis 1, y array axis 0; volume is 1/N². Prescribed initial and first target
values are centre samples installed as cell-constant DOFs. They are not exact
cell averages of the analytic nonlinear accumulation. The forcing is derived
independently from that target and the same discrete face operator. The second
step retains the actual initial forcing; it has no fabricated second target.
The fixed original guard is 3e-8. Newton/GMRES controls and FD1e-6 remain exact.

All four saved phases are checked. Qn is the initial Q for the first step and
accepted Q for the second. The first cold history store initializes both slots
with the accepted field; the max-lag-one declaration has two physical slots.
The reader checks actual policy, typed slots, fill, exact outgoing dt, sample
ordinal and starting point, actual Q/forcing state, geometry and accepted
clock. Continuous/replay leaves remain byte exact, apart from the individually
authenticated run and restart identities. A replay run is a fresh continuation
of the accepted run, whose accepted checkpoint is the last restart. The reader
recomputes every checkpoint content receipt, run and restart digest. Opaque
auxiliary/diagnostic payloads are compared byte-for-byte, not given an invented
physical interpretation.

The real IR is parsed independently. Exact Fraction polynomial normalization
checks original local residuals, candidate D, Q projections and constraints;
three numerical samples are insufficient. Unknown and capture handles, native
issued tau/clock, capture ordering, distinct Q ports, commit targets, history
stores and observation ordering are checked. The Program hash is recomputed
from the actual IR with node provenance removed, matching the declared compiler
contract. No replacement IR or ProgramCPP is emitted by the reader.

Each case is a closed inventory of eleven files: receipt, initial NPZ, four
phase NPZ, three authentic checkpoint NPZ, one retained ProgramCPP and one
Program IR. Block/program DSOs and sidecars are external pinned leaves. Binary
and artifact sidecar digests are independently recomputed; the Program component
association is explicit. The fixture has no source/command for block DSOs and
no full compiler-owned artifact-spec payload or aggregate linking record.
Therefore the reader refuses a cryptographic CPP→DSO aggregate claim. It does
not reconstruct missing build records from a DSO path. The actual Program
compile command and component association are retained, not executed.

## Seals and pending assembly

Schemas are `sol61.stage-execution-owner@1`, `sol61.stage-owner-pins@1`,
`sol61.stage-root-approval@1` and `sol61.stage-scientific-reception@1`. Pending
assembly is idempotent and never approves evidence. Two SHA256 strings are
required externally: the ROOT owner-pins digest and a separate ROOT approval
digest. Approval must have exactly:

```json
{"schema":"sol61.stage-root-approval@1","approved_by":"ROOT","pins_sha256":"ROOT-provided digest","qualification":"uniform-original-stage-saved-state-equations@1"}
```

The owner contains `schema`, `source_commit`, `native_build_source_commit`,
`abi_key`, `python_package`, `sdk`, `native`, `sources`, `cpp_dso_links`.
Package/SDK/native and each source are exact `{path,sha256}` records. Sources
have exactly `fixture`, `physical_helper`, `controls_helper`; `cpp_dso_links`
is null. Run-source and native-build source commits are distinct. If the latter
is absent it is null, not inferred. Complete installed-package/build
attestation remains ROOT's responsibility; hashing an entry point alone does
not prove every installed source. Origins use their actual canonical paths;
this schema has no fallback to relocated archives or a rebuilt environment.

```bash
PY=/Users/romaindespoulain/miniforge3/envs/pops-api040/bin/python
rtk proxy env -u PYTHONPATH -u POPS_NATIVE_DIM "$PY" -B tests/review/sol61_stage_saved_reception.py assemble \
  --archive-root "$CASE_ROOT" --owner "$ROOT_OWNER_JSON" --output "$PENDING" \
  --file-root "$CASE_ROOT" --file-root "$SOURCE_ARCHIVE_ROOT" --file-root "$ACTUAL_ENV_ROOT" \
  --case "$CASE0" --case "$CASE1" --case "$CASE2" --case "$CASE3" \
  --case "$CASE4" --case "$CASE5" --case "$CASE6" --case "$CASE7" \
  --junit "$RANK0_XML"
# MPI2 adds --junit "$RANK1_XML"; both ranks must map the same eight archives.
rtk proxy env -u PYTHONPATH -u POPS_NATIVE_DIM "$PY" -B tests/review/sol61_stage_saved_reception.py receive \
  --pins "$PENDING" --pins-sha256 "$EXTERNAL_ROOT_PINS_SHA" \
  --approval "$ROOT_APPROVAL_JSON" --approval-sha256 "$EXTERNAL_ROOT_APPROVAL_SHA"
```

Strict JSON, closed dictionaries, aliases/parent aliases, duplicate JUnit
properties, foreign rank/case/artifact, failures/errors/skips, foreign phases,
changed digests and exact dtype/shape/clock violations refuse reception. MPI2
means eight scientific witnesses shared by two JUnit receipts, not sixteen.

## Observed genuine archives, still pending external seals

Read-only inspection on 2026-10-01 used
`/Users/romaindespoulain/dev/tmp/pops-api040-native-reception-evidence-20261001/`:

- `installed-sdk2e4-evolved-stage-continuation-serial-dim2/pytest-tmp/...`;
- `installed-sdk2e4-evolved-stage-continuation-mpi2-dim2/rank0-tmp/...`.

These are ROOT's actual closed successful native campaigns, not reader-produced
states. Sixteen unique archives (eight in each mode), 64 phase NPZ and 48
checkpoint NPZ passed the independent read-only checks. The maximum original
relative residual in both modes is 1.953003193578846e-11. Serial JUnit has eight
passing cases; MPI has eight passing cases per rank, with a single rank0 archive
per witness. No native call was made during this review.

Sources are ROOT's genuine Git135179aa9039101f47e47472979f97434ee2b0c9 exports.
Native build source is separately reported as
e01c12a7988b1fb0f5e55658d76ce9e0d2cb8c6a; the installed Python correction kept
the native DSO unchanged. ROOT supplied fixture SHA
4405fdb592835b4776d3b0657444aca0818b09096f807c0e9e5c74ba3f168e57,
physical-helper SHA 0a7048d1c6a06d9b57c633771002ea317901bd3646d93a016abd411f1d038aa7,
and controls-helper SHA 0226d9889de555b20c4a4a97d3ce29ff055fb719c1d6829cc4935240e8ca6f34.
Those source files are not imported.

Validation commands in this private worktree:

```bash
rtk proxy env -u PYTHONPATH -u POPS_NATIVE_DIM "$PY" -B -m pytest -q tests/review/test_sol61_stage_saved_reception.py
rtk proxy env -u PYTHONPATH -u POPS_NATIVE_DIM "$PY" -B tests/review/test_sol61_stage_saved_reception.py \
  --mode serial --campaign "$BASE/installed-sdk2e4-evolved-stage-continuation-serial-dim2" \
  --output /tmp/sol61-stage-serial-observed-final.json
rtk proxy env -u PYTHONPATH -u POPS_NATIVE_DIM "$PY" -B tests/review/test_sol61_stage_saved_reception.py \
  --mode mpi2 --campaign "$BASE/installed-sdk2e4-evolved-stage-continuation-mpi2-dim2" \
  --output /tmp/sol61-stage-mpi2-observed-final.json
```

Eleven ordinary tests cover independent rational face algebra, nonlinear Q,
exact polynomial discrimination and protocol refusals; their synthetic inputs
are never called native evidence. The explicit actual-archive harness also
refuses eighteen copied/redigested countermodels per mode: altered Q, readonly
forcing, bootstrap history, original IR body, D, Q commit, checkpoint sample,
slot and replay lineage. Three additional forged-rank JUnit copies are refused.
Countermodels are freshly serialized/read back and checkpoint content/restart
receipts are fully resealed before the semantic guard is reached. Donor files
and authentic native results are untouched. These reports explicitly remain
`pending_external_ROOT_seals`; this agent creates no authoritative approval.

Pending assemblies (inputs for ROOT review only) were written outside the
repository to `/tmp/sol61-stage-owner.pending.json` and
`/tmp/sol61-stage-{serial,mpi2}-pins.pending.json`. Their pending pin digests are
Serial `92c593caf674cb98772c81d8bc06f68cfcd825c6a706e110f2c0082a88134e11`
and MPI2 `029b930014f945f2c547ef9543bb0293f1991c7e5d6169ce96b1c2de68f2444a`.
ROOT provided the actual SDK manifest pin: installed
`pops/include/pops_headers.manifest`, 21324 bytes, SHA
4942e284b6200142ad3bde5462c6a6c908f68267a278f9a31f7393c5248bf26d.
This file digest is distinct from the 2e4d69fd… header signature.
The reader's NPZ budget is 64 MiB and binary-origin budget is 1 GiB; these are
bounded evidence-reader limits, not caps on production PoPS dimensions.

## Independent reception of ROOT's actual external pairs

ROOT subsequently reviewed the pending pins, checked actual origins/raw batches,
and issued separate approvals. On 2026-10-01 this agent reopened each actual
pair once in read-only mode with the four externally supplied hashes below.
`receive()` passed for both modes. Its complete result exactly equals ROOT's
saved scientific reception, and every scientific value and receipt pin exactly
matches the preceding independent observed report. No approval, positive
rescellage, native execution, environment or donor mutation occurred here.

Under the same evidence BASE, actual reception directories are
`stage-serial-scientific-root-owner-v1` and
`stage-mpi2-scientific-root-owner-v1`. Files are `owner-pins.json`,
`root-approval.json`, `scientific-reception.json`, `root-reception.json`.

| Mode | External owner-pins SHA256 | External approval SHA256 |
|---|---|---|
| Serial | 92c593caf674cb98772c81d8bc06f68cfcd825c6a706e110f2c0082a88134e11 | 2576c3baaa0108c63585effd39a1a136dff7b6b8e47b1bb483656f470b09d89c |
| MPI2 | 029b930014f945f2c547ef9543bb0293f1991c7e5d6169ce96b1c2de68f2444a | 42c98b569c7073c2edd4d449a34550483c492925ecec02dc67baf9f148d2fd22 |

ROOT's scientific-reception SHA256 is
`aa294483c8e46d7c0a7289b45b5d276aa0648d3ca2c5484367319e9e6d54a8c5`
for Serial and
`20898f73255eace06ab904bd55bfd8f91dba6af2f3f150a174efd56830827521`
for MPI2. ROOT's root-reception SHA256 is
`dc9a7b3558cd421124c550d8f786c0d34396d2d2464fe5e53e294e4bfdf746f8`
and `99957114191fa4f8ab890fdd582505d2a883ffbbd13833f4f21e17b75a7383a3`,
respectively. All four saved result files were independently rehashed.

The independent comparison report was saved outside the repository at
`/tmp/sol61-stage-external-root-pairs-independent-reception.json`, SHA256
`68e0b9df0ee6e10e78345a7a8c8e92da66b8ed18db8ab26906ae00793072dd4b`.
Both modes still give maximum original relative residual
1.953003193578846e-11. The actual external pairs close reception only for
`uniform-original-stage-saved-state-equations@1`; all previously documented
CPP/DSO association and uniform-witness limits remain. Historical source-only
checks, earlier native compilation failures and unsealed observations are not
silently promoted into these authenticated campaigns.
