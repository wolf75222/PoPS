# M18 saved-state reception: prepared, native reception pending

This tests/docs change is based on `48871851`. It adds a receipt contract to
the existing public M18 witness; it does not change the example, numerical
implementation, quadrature, Newton parameters or runtime. Source checks do not
authenticate execution of a native package.

The example solves three multipliers against a separate read-only three-moment
target in twenty cells. The five nodes are `(-1, -.5, 0, .5, 1)`, with weights
`(.1, .2, .4, .2, .1)` and basis `(1, v, v*v)`. For each saved cell the independent
oracle computes `p_j = w_j exp(lambda_0 + lambda_1 v_j + lambda_2 v_j^2)` and all
three original moments with `math.fsum`. Their absolute residual must be at most
`2e-11`. It also checks finite strictly positive populations, the specified
moderate multiplier recipe, the primal entropy `sum p*(log(p/w)-1)`, and a
positive entropy gap along explicit moment-preserving dyadic null vectors. It
uses no Newton solver, PoPS import, runtime or prototype.

The normalized target's five-node cone is classified independently by its
polygon edges. The native witness contains twenty moderate interior cells and
one outside target `(1, 0, 1.1)`. Mathematical boundary classifier tests do not
constitute a native boundary campaign. Native failure messages remain generic
`NativeOutcome` evidence; the offline labels are deductions from saved moments.

## Actual files required from the owner

The fixture writes `provenance.json`, then three files per phase: `<phase>-state.npz`,
`<phase>-receipt.json` and the actual checkpoint path returned by PoPS. The
checkpoint's basename and SHA are in the receipt. The ten phases are:

1. `initial`: zero seed and the actual read-only interior target.
2. `accepted`: first accepted solve, time `.01`, step 1.
3. `continuous`: second accepted solve, time `.02`, step 2.
4. `restored`: a fresh bind restored from the first accepted checkpoint.
5. `replayed`: its second solve, bit-identical to `continuous`.
6. `outside_before`: the immutable outside target and zero seed.
7. `outside_rejected_0`: first refusal on this runtime.
8. `outside_rejected_1`: second refusal on the same runtime.
9. `safe_rebind_initial`: a new bind of the original interior target, same
   compiled artifact, context and zero seed.
10. `safe_rebind_accepted`: its accepted solve, identical to `accepted`.

`safe_rebind` is not an accepted retry of the impossible outside instance. The
two refused attempts must preserve physical buffer bytes, temporal controller,
accepted clock and empty native exchange mailbox. No smaller dt is used as a
cure, and no target commit is introduced. The restarted and rebound targets
must remain bit-identical to the first actual initial target.

Saved arrays have explicit `component,y,x` storage `(3,4,5)`. Receipts include
actual native geometry and its typed array evidence, volume/coverage/validity
arrays, per-rank local boxes for both blocks, raw checkpoint state shapes,
artifact/bind/run identities, canonical time and temporal state, and failure
diagnostics. The oracle checks the whole `5x4` support exactly once, physical
area `.2*.25`, empty coverage, and matching block ownership. It distinguishes
distributed ownership (including empty ranks) from replicated ownership rather
than claiming that one layout mode was received in advance.

The owner supplies a JSON manifest of schema `sol61.m18-owner-pins@1` and its
externally communicated SHA256. Its exact top-level fields are:

```text
schema, source_commit, native_sha256, abi_key, dimension, ranks,
artifact_identity, native_capability_abi, native_system_package_abi,
provenance, sources, junit_by_rank, phases
```

`provenance` and every file leaf have exactly `{path, sha256}`. `sources` has
`fixture`, `example`, `snapshot`: the actual executed fixture, original example
and receipt helper bytes. `phases` has the ten phase names above; each contains
exactly `state`, `receipt`, `checkpoint` file leaves. `junit_by_rank` contains
one authentic JUnit file per rank with the named witness and its recorded
dimension/rank/native/artifact properties. Thus there are **35 pinned files in
Serial, 36 in MPI2**, excluding the owner manifest itself. JUnit files must have
distinct paths; skipped, failed or missing witnesses are refused. The fixture
does not manufacture owner pins or accept its own output as external authority.

The provenance records actual native extension SHA/ABI key/capabilities on each
rank, both actual System package binary SHAs and exported ABI versions, platform,
compiled plan/component evidence, actual Program IR and its hash, and the three
executed source SHAs. Current source distinguishes native module capability
**ABI5**, System package **ABI7**, and this local M18 **ProgramIR5**. Historical
Exp/schema/GeneratedPackage terminology does not authenticate a current binary.
The offline reader authenticates checkpoint envelopes, typed arrays and restart
identities, bound-initial provenance without an invented run, run provenance,
physical state and the empty rank-local `POPSEX01/02` ledgers.

## Independent countermodels

`sol61_m18_entropy_countermodels.py` first requires a received authentic
positive manifest. It then copies those actual files to a new empty private
directory and prepares twelve negative controls: wrong multiplier, changed
read-only target, transposed axes, population overflow, feasible boundary in
place of the outside target, changed rejected buffer, one-ULP publication
clock, boolean clock, missing owned support, future target capture, relaxed
original residual tolerance, and a negative quadrature weight.

It reseals edited state/checkpoint arrays, restart identities, receipt hashes
and external file pins. Each local manifest is explicitly named `NOT-OWNER`;
its seal is a negative harness authority, never a replacement owner/native
receipt. All file pins are checked before requiring a scientific/protocol
refusal, and the authentic positive is rechecked afterward. These twelve real
file controls are **prepared but not yet executed**. No positive physical state
was generated for this review.

## Checks and reception commands

In the private checkout, the source/unit command below received **46 PASS**.
It includes a small existing host C++ Exp expression probe. It does not build
or load a generated PoPS package, install anything or run JIT/native M18.

```bash
rtk proxy env -u PYTHONPATH -u POPS_NATIVE_DIM /Users/romaindespoulain/miniforge3/envs/pops-api040/bin/python -I -c 'import sys; from pathlib import Path; sys.path.insert(0,str(Path.cwd())); sys.path.insert(0,str(Path.cwd()/"python")); import pytest; raise SystemExit(pytest.main(["-q","--tb=short","tests/review/test_sol61_m18_entropy_offline_contract.py","tests/python/unit/moments/test_m18_discrete_entropy.py","tests/python/unit/moments/test_exp_entropy_independent_review.py"]))'
```

Ruff passed on the fixture, helper, oracle, countermodel harness and contract
tests. Source collection received the one public witness; that is not execution.
Source tests validate actual public resolve/Program IR and reject altered Newton
budgets, six-unknown output, future target capture, scaled original residual,
extra target commit, malformed clock, wrong ownership and absent external pins.

Root receives the installed **Dim2** witness in Serial and MPI2 with coherent
native/SDK/package identities and one separate JUnit file per rank. Each native
lane must execute all ten phases, both outside refusals, exact restart/replay,
and safe rebind without a skip. Root records the authentic artifact, source,
native, package and support facts and provides the external manifest/seal.
Compilation inside this compiler-marked fixture is exclusively root's task.

The offline commands run against those future actual files without importing
PoPS. Paths and SHA below are supplied by the owner, not calculated as a
positive seal by this review:

```bash
rtk proxy env -u PYTHONPATH /Users/romaindespoulain/miniforge3/envs/pops-api040/bin/python -I tests/review/sol61_m18_entropy_offline_oracle.py --pins /OWNER/m18-owner-pins.json --owner-sha256 OWNER_EXTERNAL_SHA256 --output /PRIVATE/m18-reception.json
rtk proxy env -u PYTHONPATH /Users/romaindespoulain/miniforge3/envs/pops-api040/bin/python -I tests/review/sol61_m18_entropy_countermodels.py --pins /OWNER/m18-owner-pins.json --owner-sha256 OWNER_EXTERNAL_SHA256 --countermodels-dir /PRIVATE/EMPTY-m18-controls --output /PRIVATE/m18-controls.json
```

Without `--pins`, the oracle emits only the pending file contract. Present
evidence is source/unit evidence. There is no new native, MPI, AMR, near-boundary,
Vlasov/BGK or M19 qualification, and no production defect is demonstrated by
this lot. Earlier M18 execution receipts remain historical evidence for their
own exact source/native versions.
