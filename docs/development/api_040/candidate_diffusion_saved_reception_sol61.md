# Candidate-D saved-state reception @1 (Sol6.1 independent)

## Frozen scope and limits

Reader: `tests/review/sol61_candidate_diffusion_saved_reception.py`.
Tests: `tests/review/test_sol61_candidate_diffusion_saved_reception.py`.
Exclusive review branch `codex/api040-sol61-candidate-d-offline`, source base
`39b2db311148257099e47ec0b305a7d6d2d59e2d`. No production files changed.
The original `sol61_captured_diffusion_saved_reception.py` frozen @2 is byte
unchanged and is neither imported for science nor promoted to candidate-D.

**62 source/math/protocol tests PASS, 2.29s** before final formatting. No
PoPS/Native import, DSO execution, JIT, build or shared environment mutation.
Synthetic positives in unit tests are explicitly protocol/mathematics tests;
they are never Native receipts or ROOT authorizations.

The real SDK2e4 historical archive inspected in read-only mode is
`/Users/romaindespoulain/dev/tmp/pops-api040-native-reception-evidence-20261001/installed-sdk2e4-diffusion-and-diagnostics-serial-dim2-corrected-selection`.
Its raw JUnit has eight cases, six passes and two Uniform diagnostic failures.
The two exact candidate cases pass. Each candidate directory has seven files:
five NPZ, receipt and compiler-owned `program-3.cpp`. The checkpoint receipt
paths alias the phase NPZ: the fixture overwrote the native checkpoint with
saved observations. No archived CP, diagnostics, accepted ledger or history
proof survives there; in-process restart assertions remain JUnit evidence.
The retained CPP contains `pops_program_hash()` and real lowering, but no
serialized Program IR. Both actual seven-file inventories and CPP route/control
contracts were parsed successfully as **read-only protocol checks only**. No
scientific reception of these originals was run without external ROOT seals.

@1 qualifies original saved-state science plus actual source/CPP/component
binary evidence and ROOT-attested execution association. Output explicitly has
`checkpoint_reception=false`, `documentary_ir_reception=false` and
`cryptographic_aggregate_binding=false`. No aggregate payload or block CPP is
retained. Recomputed component binary/artifact identities do not prove a
cryptographic CPP-to-DSO graph. No MPI execution is performed by this checker;
Serial/MPI2 means the externally approved run's topology, not a new execution.
No AMR, GPU, convergence, SPD, arbitrary-D or partition qualification follows.

## Independent original equations

The physical order is `(0)` for scalar and `(2,0,1)` for the signed coupled
case. Rows and columns are returned to physical order before the oracle.

```
D scalar = .012; R scalar = 1.1
D3 = [[.012,.002,0],[-.001,.014,0],[.001,0,0]]
R3 = [[1.1,.03,-.01],[-.02,1.3,.02],[.01,-.03,1.5]]
D_rc(q,alpha) = D_rc * (1+alpha) * (1+3*q_c**2)
F_r = -div(sum_c D_rc(q,alpha) grad(q_c)) + sum_c R_rc*q_c + .2*q_r**3 - f_r
```

The matrix is nonsymmetric and singular in diffusion. No SPD surrogate is
used. Each periodic x/y face is visited exactly once; the arithmetic mean of
its two endpoint coefficients multiplies the difference of q. Opposite cell
contributions are equal and opposite. The second implementation uses exact
`Fraction` arithmetic on the actual binary64 inputs. It checks zero total
spatial contribution and the floating evaluation against exact arithmetic.
The oracle never imports the author's physical helper.

The declared target is `.15+.025*cos((c+1)*2*pi*x)+.02*sin(2*pi*y)` and material
is `.25*sin(2*pi*x)+.15*cos(2*pi*y)` at cell centres. Domain is periodic unit
square, 16x16, dt=.01, two accepted steps. Initial target/material agree with
these independent expressions to 2e-14; initial forcing is the original
candidate-dependent F evaluated independently at the target (relative 2e-13).
The initial NPZ is the declared initial input, not a separate Native initial
response snapshot. Accepted and continuation states satisfy **solution error
and original residual <=3e-8**. Source/alpha bytes remain exact. Consumer
response/time equals q within the same limit. Accepted/reloaded and
continuous/replay state arrays are compared bit for bit, including clock/step.
All strict seven Newton controls are authenticated in source, receipt and CPP:
`tolerance=1e-10,max_iterations=20,linear_tolerance=1e-8,
linear_max_iterations=240,restart=60,armijo=1e-4,minimum_step=1/1024`, FD=1e-6.

The source/protocol-only CP predicates already exercise CPUniform8, exact
component/state/geometry, empty POPSEX01/02, POPSDIA1 opaque names/value bits,
rank/width/cardinality/offsets, selective-history issued intervals and replay.
They do not run during @1 scientific reception or turn overwritten phase NPZ
into checkpoints. Polynomial tests normalize the exact documentary AST with
Fraction coefficients and reject changed D columns, beta, reaction/load and
capture identities; their input is explicitly synthetic documentary IR.

## ROOT seals and exact inventory

New schema `sol61.candidate-d-owner-pins@1`; qualification
`saved-candidate-d-original-residual@1`. Pins contain exactly:
`schema,qualification,archive_root,file_roots,mode,ranks,owner,junit,cases`.
Mode is serial or mpi2, ranks exactly 1 or 2, all-rank raw JUnit pins required.
Cases are exactly `scalar1` and `coupled3-201`. Case records contain
`directory,artifact,receipt,initial,phases,checkpoints,cpp,ir`; ir is null.
Phases are accepted/continuous/reloaded/replay, checkpoint aliases must match
accepted/continuous/replay phase pins exactly for this historical @1 contract.
A closed directory has seven actual files, not ten fictional files. File pins
are exact `{path,sha256}`. Symlinks in files or parents, `..`, foreign roots,
changed bytes, duplicates and noncanonical identities are rejected. Reads use
one regular-file descriptor, bounded fstat length, pread and exact stat/hash
stability, with no unbounded EOF read.

`sol61.candidate-d-execution-owner@1` exact fields:
`schema,source_commit,native_build_source_commit,abi_key,python_package,sdk,
native,sources,execution_association`. Package/sdk/native are external leaf
pins. Sources have four keys fixture/physical_helper/emitter/request_contract.
Current actual request contract adds only Stage@2 additive capture checks to
the reviewed base; that path is absent in candidate-D. Its eight added lines
were reviewed from `git diff 39b2db31 a1eb496f` without modifying production.
The actual four required source fingerprints are:

| key | file | SHA256 |
|---|---|---|
| fixture | tests/python/integration/runtime/test_public_captured_diffusion.py | 374b799cbd68e33f9de4d93ee5e0e8eb247c99e47e5e285c4a6d2b2933e97623 |
| physical_helper | tests/python/support/captured_diffusion_mms.py | 0226d9889de555b20c4a4a97d3ce29ff055fb719c1d6829cc4935240e8ca6f34 |
| emitter | python/pops/codegen/program_emit_nonlinear_field.py | 729f161c996c9712341241e1f01cb02e3fc479cf4fe447c0062587bcc07decc3 |
| request_contract | python/pops/fields/_program_nonlinear_problem.py | cf60a119549e2a095ac4c01dfd29b0d2ee2ee5371e22a73096d6f916dfe84f9f |

`execution_association` has scalar1/coupled3-201. Each case has
`authority,aggregate_artifact,components`. Authority is exactly
`ROOT-attested actual execution/component association; aggregate payload not retained`.
Components match the receipt's actual four names: block-response, block-forcing,
block-material and its actual program-* name. Each has `{so,sidecar,cpp}`;
block cpp is null; Program cpp is the same retained source pin. Each actual
.so and .so.pops-artifact.json is externally pinned, decoded, and recomputed
using the binary/artifact identity primitives. The CPP exported
`pops_program_hash()` must equal the Program sidecar semantic digest. This is
an explicit association, not inferred from filenames or aggregate digest.

ROOT approval is a **separate externally sealed** file with exact fields:
`schema=sol61.candidate-d-root-approval@1,approved_by=ROOT,pins_sha256,
qualification=saved-candidate-d-original-residual@1`. Both external SHA256
arguments are mandatory before input access. Assembly emits a pending inventory
and never mints approval. ROOT must control and publish both SHA256 authorities.
A raw mixed JUnit is accepted without filtering or rewriting: the two exact
candidate names and properties must pass, be unique and associate exactly with
case receipts/artifacts/dimension/rank/size. Summary counts must match raw
children. Other cases cannot claim the candidate receipts. Output preserves
all raw batch counts, including outside failures/skips; it never calls the
whole batch green.

## Commands and prospective @2 obligation

From the exclusive review checkout:

```sh
rtk proxy env -u PYTHONPATH -u POPS_NATIVE_DIM PYTHONDONTWRITEBYTECODE=1 /Users/romaindespoulain/miniforge3/envs/pops-api040/bin/python -B -m pytest -q --tb=short tests/review/test_sol61_candidate_diffusion_saved_reception.py
rtk proxy env -u PYTHONPATH -u POPS_NATIVE_DIM /Users/romaindespoulain/miniforge3/envs/pops-api040/bin/python -I tests/review/sol61_candidate_diffusion_saved_reception.py assemble --archive-root "$candidate_archive" --owner "$candidate_owner" --case "$scalar_case" --case "$coupled_case" --junit "$rank0_junit" --file-root "$candidate_archive" --file-root "$source_root" --file-root "$installed_root" --file-root "$cache_root" --output "$pending_pins"
rtk proxy env -u PYTHONPATH -u POPS_NATIVE_DIM /Users/romaindespoulain/miniforge3/envs/pops-api040/bin/python -I tests/review/sol61_candidate_diffusion_saved_reception.py receive --pins "$root_pins" --pins-sha256 "$external_pins_sha" --approval "$root_approval" --approval-sha256 "$external_approval_sha"
```

MPI2 adds `--junit "$rank1_junit"` at assembly. The ROOT paths/SHAs are
mandatory external input, not examples of fabricated positive authority.

A distinct future Candidate fixture@2/reader@2 must require separate immutable
`accepted-checkpoint.npz`, `continuous-checkpoint.npz`, `replay-checkpoint.npz`
and real same-component `dump_ir()` output plus external approval @2. The
public `CompiledProblemDumpMixin.dump_ir(path)` returns the compiled handle's
attached detached Program `_serialize()`; it must be retained during the true
run, not recreated from a builder or inserted into old CPP. Current archived
states-only @1 can never silently acquire these claims. Fresh Native campaigns
and exact new fixture source fingerprints remain ROOT's responsibility.
