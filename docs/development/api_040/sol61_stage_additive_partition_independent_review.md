# Independent additive Stage and State partition reception

Reviewed immutable production: additive `6528af09105fb625047205b7fc4a0fdf1bda3d95`,
partition `6d88b165fb087ad85791b899cf360ae29391d8f5`, capture-read fix
`41803be2` and fixture/test/doc freeze
`42798c2b3a6f74db622c1c91cec39ea76a3e7bc3`, based on `5708718e`.
The review worktree is `/Users/romaindespoulain/dev/tmp/PoPS-sol61-stage-additive-review`.
Only the two independent tests and this report are delivered by the reviewer.
No MAIN, environment, SDK, header, runtime library, compiler or JIT was mutated.

## Original equation and native route

The public descriptor `pops.evolved-field-rate.spatial-additive@1` separates a
spatial rate from a captured additive source. Projection schema 2 and ProgramIR
14 distinguish it; partition publication `pops.evolved-original-field-stage@2`
requires ProgramIR 15. Unextended stages retain their original bytes.
The equation is `Q(T) - tau*(R_spatial(T) + f(State)) - Qn = 0`.
The existing original RHS provider receives `Qn + tau*f`, while the existing
operator receives the spatial terms with the original sign. Neither division
by T nor a fabricated local reaction is used to represent f.

The independent witness uses three physical unknowns, two nonlinear conserved
components, an auxiliary equation, signed cross diffusion and a separately
owned forcing State. A Fraction AST evaluator checks the full original local
expressions and diffusion matrix, independently of the producer's lowering
helpers. Permutations preserve unknown ordering and conserved component order.
Spatial `None` and exact zero work with both literal and captured f; T=Qn=0
remains finite. Unknown-dependent additive RHS is explicitly refused rather
than silently frozen. Native per-candidate RHS remains a separate realization.

Public validate/resolve, real detachment and emitted C++ are exercised. The
public compile seam is stopped by a sentinel at the first compiler invocation;
no compiler or fake DSO is executed. The generated route retains Kokkos,
original-residual recheck, exact issued `ctx.step_dt()`, attempt/point/lane
checks and collective refusal before publication. This is source reception,
not execution of those native checks.

## Counterexample and correction

At 6d88, replacing the genuine source n-read's point with `TimePoint(clock,1)`
before binding was accepted through public validate/resolve. Such an op=state
still emits `ctx.state(index)` and cannot attest a future native read. Euler
confirmed and fixed this in 418: schema-2 capture graphs recursively require
op=state reads at their accepted TimePoint(clock,0), at bind and revalidation.

The unchanged counterexamples now refuse direct and linearly wrapped relabels.
A genuine `Program.value(..., at=next)` computed from accepted n-reads remains
admitted. A homonymous foreign Program source, foreign Tau, altered previous
endpoint, encoded Tau, original Q, additive body and descriptor are refused.
The read fix is conditional on the new extension; no legacy read policy was
changed. Direct Python mutation is an adversarial source probe, not a public
future-State API or a native result.

## Two actual conserved carriers

An independent public witness owns Q0 and Q1 in distinct Models and Blocks,
with widths 1 and 2. The stage previous tuple is Q0[a], Q1[c], Q1[b]; physical
unknown order is V,T,U. The emitter selects Q0 indices (0) and Q1 indices (2,1),
so the second carrier receives Q(V,T) then Q(U), in its exact declared order.
Fraction evaluation checks both the original equations and published Q.
Missing publication, duplicate previous component, permuted indices, bool
indices and swapped projection expressions fail at source emission before a
compiler or runtime publication. Partition completeness is checked at Program
region validation/emission; the test does not claim that resolve alone rejects
all those mutations.

AMR conservation remains the prior source-only trace: the existing synchronized
finish averages fine Q into covered coarse Q before state publication/history
swap. It does not evaluate Q(mean(T)) in place of mean(Q(T)). No new AMR native
conservation, rollback, regrid or MPI qualification is asserted here.

## Independent manufactured forcing

`test_sol61_stage_additive_mms_math.py` has no PoPS import. Its oracle visits
one oriented physical face and applies opposing budgets to both cells, using
arithmetic D and physical spacing. Independent synthetic tests use a signed,
singular three-component matrix and detect doubled duration, missing forcing,
wrong previous Q and mismatched component permutations.

The eight frozen fixture configurations are also checked: N8/N16, dt .01/.02,
scalar fixed D or two evolved Q and three physical unknowns with candidate D.
The comparison target's three functions are extracted via AST; expected arrays
are separately computed from the declared coordinates, Q0=T0+T0²+.1T1²,
Q1=T1+T1²+.2T0*T1, z=.25T0+.5T1 and signed matrix
[[.012,.002],[-.001,.014]], D column factor 1+.4*Tcolumn².
Expected forcing is (Qtarget-Qinitial)/dt - div(D grad Ttarget).
No author numerical helper contributes to the expected oracle. Fixture history
has two physical slots for max lag 1, latest slot 1; its sample ordinals are 1
for each new physical window, consistent with next_history_sample. This source
reading is not an accepted checkpoint or saved-state proof.

## Reproduction and limits

From the review worktree:

```sh
env -u PYTHONPATH -u POPS_NATIVE_DIM PYTHONPATH=python PYTHONDONTWRITEBYTECODE=1 /Users/romaindespoulain/miniforge3/envs/pops-api040/bin/python -B -m pytest -q tests/review/test_sol61_stage_additive_independent.py tests/review/test_sol61_stage_additive_mms_math.py tests/review/test_sol61_evolved_original_field_stage.py tests/review/test_sol61_evolved_stage_mms.py
env -u PYTHONPATH /Users/romaindespoulain/miniforge3/envs/pops-api040/bin/python -m ruff check tests/review/test_sol61_stage_additive_independent.py tests/review/test_sol61_stage_additive_mms_math.py
```

The coherent selection receives 85 source/math cases (51 independent, 34
producer/legacy), Ruff and diff checks. Four legacy IR/C++ pairs are re-emitted
from a fresh Git archive of 5708718e and the fixed 418 source, using the same
review script/callsite and installed catalog: all bytes/digests match exactly.
They cover ImplicitStage, nonlinear-map ImplicitStage, mixed linear and its
permutation. The temporary exact outputs are
`/tmp/sol61-stage-additive-parity-0.json` and
`/tmp/sol61-stage-additive-parity-final.json`; these are source receipts only.

Native Newton/GMRES, MPI/empty ranks, GPU, prepared callback lifetime, rejected
attempt rollback, checkpoint/restart/replay and saved-state budgets remain ROOT
execution obligations. No native artifact, NPZ, owner approval or CPP-to-DSO
link was invented. This review qualifies neither full M06/M13 nor a continuum
convergence family.
