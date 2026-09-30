# Independent public ALE fixture audit and pending offline reception

Source base: `009ed9d2e27d9f41d697ef3c9362878705cf22c0`. Exclusive checkout:
`work/PoPS-sol61-ale-offline-reception`, branch
`codex/api040-sol61-ale-offline-reception`. This change contains fixtures, an
independent reader/oracle, protocol checks and this report only. No production
header, Python implementation, MAIN checkout, installed environment, SDK or JIT
was modified. **Scientific receipt status: pending actual native ALE data.**

## What the original witness could establish

The declared public chain is Case -> validate -> resolve -> compile -> bind
with a declared BindArray/ConservativeCellAverage initial subject -> run. Its
original equation uses the authored physical flux `F=aU`, original source
`S=.05U` when enabled, periodic traces and a moving Reynolds projection. The
mesh law is `x(X,t)=X+.08*sin(2*pi*X)*t`; the selected prepared realization is
Uniform1D, first order, centered relative Rusanov. This is the currently
available provider's scope, not a permanent production dimensional limit.

Initial constant states distinguish component names and permutations: `a=2`,
`b=3`, `c=4`. No-source constant-state preservation is a GCL witness, not an
independent transport accuracy witness. With source and old-volume quadrature,
the density becomes nonuniform after the first moved interval, so subsequent
real face receipts can receive the sign of relative characteristic speeds.
The three-component `(a,b,c)` case now uses `a=-.3`; scalar and `(c,a,b)` use
`a=.7`. Original source/projection/equations and all existing tolerances remain.

The original fixture checked real NPZ endpoints/measures/inventory and exact
restart field/wire equality in memory. It did not preserve all restart phases
as files or exercise a refused attempt. Its scientific files were also read
in lexical filename order. Actual filenames put the run-family identity
before the step, so that order is not a physical chronology across runs. The
fixture now orders by the authenticated accepted macro-step from NPZ.reopen,
requires exactly steps 1/2/3 and retains the original strict GCL/inventory guards.

## Fixture changes and MPI boundaries

For each actual N16/N32 scalar/component/source case the saver captures six
phases: initial, step1, accepted (step2), continuous (step3), restored (step2),
replayed (step3). Each phase preserves a real global physical state, global
physical nodes and measures, geometry generation, accepted time/step, exact
checkpoint and JSON receipt with all rank-image SHA256 values. The initial
image must carry bound_initial provenance, zero accepted clock, generation0,
no interval receipt and no accepted exchanges. It does not invent an initial run.
Restored and replayed arrays, geometry generation, clock and rank-local POPSEX03
images must equal their corresponding accepted phases byte for byte.

Two further public N16/N32 fixtures propose h=.1, same periodic law and a=.7.
At either physical endpoint the mesh speed is zero and h*.7 exceeds 1/N. The
real face provider emits rejected nonfinite physical amounts. The existing
shared-face guard executes before moving-GCL and refuses them. This control
receives that exact guard, not a claim that an inverted mesh reached GCL.
Expected serial exception: ValueError with exact diagnostic
`moving shared face or fixed-domain boundary is inconsistent`. For MPI2 the
existing Python attempt driver converges the two ValueError diagnostics into
the exact RuntimeError `collective step attempt failed during solve: rank 0
ValueError: ...; rank 1 ValueError: ...`. This wrapper is not changed here.
The exception type/message expectation is source-derived and still needs
native reception. A failure is not admitted as a successful refusal without
the expected diagnostic, exact state/geometry/wire rollback, zero accepted
clock and exactly one scientific publication from the subsequent safe h=.001
run. Before/rejected/retried phases remain separate from positive restart cases.

Every native operation, including global state, moving geometry snapshot,
checkpoint, run and restart, is entered by every rank through collective_call
or collective_attempt. Local assertions and root-only IO converge through
collective_check. Paths are broadcast through collective_directory, artifact
compilation uses compile_resolved_plan_once, model configuration is agreed
before the main chain, and outputs use ROOT mode on an actual MPI world.
The saver extracts the checkpoint's P+1 offsets and all P rank images on root;
it does not assume root's local wire is the global accepted state. These
boundaries cannot repair a collective hang inside a native operation.

## Independent offline equations and wire authority

`tests/review/sol61_moving_interval_offline_oracle.py` imports only NumPy and
stdlib, including separately checked generic evidence primitives from the
existing independent T5 reader. It never imports PoPS or calls NPZ.reopen.
It does not write owner pins or manufacture positive states/checkpoints.

For actual saved endpoints x0/x1 and actual accepted states U0/U1:

* V0=diff(x0), V1=diff(x1), swept=x1-x0; accepted measures equal endpoint
  differences exactly; GCL=(V1-V0)-diff(swept) uses declared tolerance1e-13.
* wg=swept/h, relative characteristic speed a-wg, centered density=(UL+UR)/2,
  Frel=(a-wg)(UL+UR)/2-abs(a-wg)(UR-UL)/2, periodic traces from the actual U0.
* Physical amount=h*Frel+density*swept. The wire physical amount and density
  must match this independently reconstructed numerical face.
* Source amount=h*V0*.05*U0 (or exact zero). The original source is evaluated
  from the previous physical state and the declared (1,0) measure quadrature.
* Q1=Q0-diff(physical_amount-density*swept)+source. Cell/component Reynolds
  residual, reconstructed field and global inventory are checked separately.

The existing arithmetic guard is 64 binary64 eps times max(1, |actual|,
|expected|). Original native fixture tolerances 3e-14/4e-14 stay unchanged.
No target or endpoint tolerance is enlarged. Duration, ticks, clock identities,
frame, measures, densities and ledger quantities that are exact authorities
are compared exactly rather than through a numerical tolerance.

The bounded independent POPSEX03 decoder authenticates one exact carrier,
runtime block0, physical frame, explicit geometry tolerance/clock, ncomp,
rank-space, all global boxes and owners, local patch order, array shapes,
generation/receipt, previous/current states, volumes, nodes, swept volumes,
integrated source, physical face amounts and densities. Every count/length is
bounded by remaining bytes before allocation. A 64MiB reception/decompression
budget is local to these small witness files and does not cap PoPS production.
End-time=t0+h and tick+1=accepted macro-step must match enclosing accepted time.
The inner POPSEX02 ledger is decoded separately; no integral declarations or
consumption are invented for this witness. The exact interval frame, original
source-evaluation identity, quadrature, operation/occurrence, orientation,
component support, amount, unit measure/weight and multiplicity are checked.

Each cell/component has one source occurrence and two amount incidences;
each cell has two geometry incidences: N*(2+3*ncomp) physical incidences. A
distributed carrier must have exact owner-local coverage, including empty
ranks. For a declared replicated carrier every rank-local receipt remains
authenticated and ledger copies must agree; the offline physical count uses
canonical rank0 once. This does not establish that a future replicated
IntegralState consumer would sum these copies only once; that separate
consumer needs its own native contribution proof. No replicated bytes are
silently interpreted as distributed ownership.

True ScientificOutput NPZ files are externally pinned too. The independent
reader verifies their typed array evidence, accepted clock, bind/run provenance,
piece tiling/owners, physical geometry and field bytes against phase checkpoint
data. It requires four actual outputs per restart case, one per safe retry,
and no publication from a refused interval. Immutable detachment/archive
mechanisms have existing source/host probes; no live object lifetime or native
archive qualification is inferred from this offline file comparison.

## External owner pins and native obligations

Run `--describe-contract` for schema `sol61.moving-interval.offline-pins@1`.
It explicitly reports pending_receipts. Owner supplies source commit, native
SHA256, exact Dim1 SDK ABI, identity_file path+SHA256, dimension, actual size,
artifact/platform/moving identity/frame for each case, source flag, velocity,
component order, and external file pins. The complete default campaign is
12 restart cases plus 2 retry cases: 234 phase files (3 per phase), 50 actual
ScientificOutput NPZ files and 1 native identity file, **285 required files**.
The root owner must independently seal those pins from the real run; this
reviewer does not adopt its own computed hashes as native ownership authority.
Build receipts, generated artifact/source manifests, pytest result/log and
the fixture's actual-moving-evidence.json files should also be preserved as
owner provenance alongside that mathematical input set.

Native reception obligations, separately for serial and MPI2:

1. Rebuilt Dim1 extension and JIT artifacts authenticate exact source/native/
   SDK and declared initial bindings. Select POPS_NATIVE_DIM=1 explicitly,
   unset PYTHONPATH, set POPS_ALE_RESOLUTIONS=16,32 and use the installed package.
2. Receive all eight parametrized pytest tests in
   `tests/python/integration/runtime/test_public_moving_interval.py` (six
   positive cases each running two sizes, plus two actual refusal/retry tests).
   No skip substitutes for any test; retain rank-local counts and diagnostics.
3. Receive initial03/schema2, all accepted intervals03, byte-exact physical
   state/geometry/ledger restart and replay, refused candidate rollback and safe
   retry. Check publication count and all source/component permutations.
4. Authenticate P-rank topology (replicated or distributed as actually built),
   empty-rank behavior and absence of collective divergence. MPI2 is pending,
   never inferred from host compilation, replicated copies or serial data.
5. Preserve real files and externally pinned manifests; run this reader without
   PoPS import. Only then may its scientific status become PASS. Later fully
   resealed scientific negative controls must be explicitly labelled harness
   countermodels, never root owner pins or positive native receipts.

Representative root commands, using root's rebuilt installed environment and
normal isolated native-cache fixture (not executed here):

```sh
rtk proxy env -u PYTHONPATH POPS_NATIVE_DIM=1 POPS_ALE_RESOLUTIONS=16,32 /Users/romaindespoulain/miniforge3/envs/pops-api040/bin/python -m pytest -q --tb=short tests/python/integration/runtime/test_public_moving_interval.py
rtk proxy env -u PYTHONPATH POPS_NATIVE_DIM=1 POPS_ALE_RESOLUTIONS=16,32 mpiexec -n 2 /Users/romaindespoulain/miniforge3/envs/pops-api040/bin/python -m pytest -q --tb=short tests/python/integration/runtime/test_public_moving_interval.py
rtk proxy env -u PYTHONPATH -u POPS_NATIVE_DIM /Users/romaindespoulain/miniforge3/envs/pops-api040/bin/python -I tests/review/sol61_moving_interval_offline_oracle.py --pins /absolute/root-owner-pins.json --output outputs/ale-offline-reception.json
```

## Source/host evidence

The independent existing public ALE suite on this base receives original
emitted bodies, signed relative upwind algebra, source permutations, detached
coordinates/archive/writers and the exact URI/rank pair. Those 39 source/host
checks passed together with 13 author codegen checks and 24 new protocol
checks: 76 PASS, 81.61s. No ALE native runtime ran in this checkout.
Final protocol/codegen validation: **43 PASS in 34.24s**, including six
independent exact default-emission comparisons with the frozen original009
fixture. Together with the 39 unchanged independent source/host properties,
82 distinct checks have passed. Ruff and git diff --check pass. Production
remains unchanged; reader positive execution remains pending owner receipts.

```sh
rtk proxy env -u PYTHONPATH -u POPS_NATIVE_DIM PYTHONPATH=python /Users/romaindespoulain/miniforge3/envs/pops-api040/bin/python -m pytest -q --tb=short tests/review/test_sol61_moving_interval_offline_contract.py tests/python/unit/codegen/test_moving_interval_codegen.py tests/review/test_sol61_public_ale_independent.py
rtk proxy env -u PYTHONPATH -u POPS_NATIVE_DIM PYTHONPATH=python /Users/romaindespoulain/miniforge3/envs/pops-api040/bin/python -m pytest -q --tb=short tests/review/test_sol61_moving_interval_offline_contract.py tests/python/unit/codegen/test_moving_interval_codegen.py
rtk proxy /Users/romaindespoulain/miniforge3/envs/pops-api040/bin/python -m ruff check tests/review/sol61_moving_interval_offline_oracle.py tests/review/test_sol61_moving_interval_offline_contract.py tests/python/support/moving_interval_receipts.py
```
