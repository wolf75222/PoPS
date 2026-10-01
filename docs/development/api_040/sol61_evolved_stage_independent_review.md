# Independent bounded review of EvolvedStage e9

Candidate: `e9fce614e3121275b65f62d4677bf229836b4ce8`.
Parent: `2b4d01d4f2eaeb69bc7f32c7dc063ebde5ca2e81`.
The candidate production tree is unchanged. Only this report and
`tests/review/test_sol61_evolved_stage_independent.py` belong to the review.

The independent public witness uses three physical unknowns in three permutations,
two conserved components, a separate load capture and an auxiliary original
constraint. It declares

- `Q_a(T,Z)=T+T³`, `Q_b(T,Z)=2Z+(2/5)T`;
- `R_a=div((1+Z²/5) grad T)−(1/10) div(grad Z)`;
- `R_b=(7/10) div(grad Z)+(1/5)A`;
- `A−T−2Z=load`, and `tau=(2/3)dt_issued`.

An independently written Fraction evaluator reads the actual lowered original AST.
It verifies the complete local residual, signed cross-diffusion matrix, auxiliary
equation, conserved component order, and the two Q publication expressions.
For `Q−tau div(K grad q)`, the native `−div(D grad q)` coefficient is `+tau K`.
No inverse accumulation, heat capacity substitution, dense solver or model-specific
dispatch is supplied by the review. Fractions passed directly to legacy Reaction
coefficients are opaque to its old identity codec; this witness therefore uses
the existing explicit `Const(Fraction(...))` typed literal API.

Countermodels alter the publication Q, previous slot, component width, target
point, target State or capture order. They are refused before publication, with
the Program value identities, counters and commit identities unchanged. A
homonymous foreign Program duration and a real owned previous State relabelled
to the next point are refused by the public binder. Changing both lowered Q and
the source-contract accumulation is refused by real public validate→resolve
against the registered original equations. A resealed duration factor is refused
against the original declared duration. Tests do not relax production guards or
mask RuntimeError/OverflowError.

The public resolved consumer emits the issued native duration, exact
frame/point/attempt checks, proper candidate/previous widths, and collective
nonfinite-accumulation refusal. A real detached Program retains exactly the same
IR hash and full emitted C++ bytes. Public `pops.compile(resolved)` reaches its
first genuine compiler call with only bootstrap/toolchain/native-dependency
execution substituted; a sentinel stops there before any compiler, binary or
native runtime executes. The existing repository toolchain stub is used only as
that execution boundary, never as a scientific oracle. No fake DSO is written.

## AMR covered cells and the accepted next Q

The projection copies previous Q and overwrites only active cells. This alone is
not a conservation certificate. The actual source chain additionally shows:

1. `program_emit_amr.py` gathers/solves once, observes and publishes each level,
   then calls `advance_synchronized_hierarchy(..., true)`.
2. `amr_program_context_subcycling_runtime.inc` routes that composite callback
   through the real synchronized subcycling engine with detached candidates.
3. `finish_synchronized_records_` in `amr_subcycling_engine.hpp` reconciles fluxes
   and restricts **each conserved candidate State** finest-to-coarsest through
   `execute_average_down_collectively`. `prepare_average_down` selects the actual
   `ConservativeRestriction` transfer. The parent's newer history is replaced
   after restriction.
4. `publish_attempt_` validates/stages these candidates, publishes their State
   carriers, then swaps accepted histories, clocks and ledgers. The next solve's
   frozen Qn captures therefore read the restricted accepted Q.

The physical unknown candidate is synchronized separately by the true provider
`stage_original_field_candidate_collectively` before observation. Restricting Q
is not applying H to a restricted T: for two equal-volume cells T=(0,2) and
H(T)=T+T³, the actual conserved mean is 5, while H(mean T)=2. The review proves
this distinction mathematically and checks the native source ordering, without
executing an AMR hierarchy.

Covered coarse cells are excluded from Newton projections/scalar products and
the original residual norm. The local callback still evaluates their temporary
finite expression before projection; it does not impose Qcoarse=H(Tcoarse) as an
accepted covered-cell equation. Native finite-domain/overflow behavior on such
temporary evaluations remains a reception obligation. The duration callback
checks the live issued dt and the existing provider guards retain lane,
attempt/point, leases, topology and preparation generation; no native regrid or
rollback test was run here.

## Checks and limits

Independent source suite: **18 PASS**. Ruff and `git diff --check`: PASS.

```sh
rtk proxy env -u PYTHONPATH -u POPS_NATIVE_DIM PYTHONPATH=python \
  PYTHONDONTWRITEBYTECODE=1 \
  /Users/romaindespoulain/miniforge3/envs/pops-api040/bin/python -B -m pytest -q \
  --tb=short tests/review/test_sol61_evolved_stage_independent.py
```

A fresh exact Git archive of the parent was compared with the candidate through
the same repository `sol61_spatial_field_legacy_parity.py` callsite and unchanged
installed catalog. Four complete legacy IR/C++ pairs match:

| Case | IR SHA256 | C++ SHA256 |
| --- | --- | --- |
| implicit-stage | `46bc2b8a4015801f7bfb9382decd793720370ffcc7c7967af974ab7e2d2b965f` | `a85a6229731a97d5d38a41a48ca64cff31093ec7e21d9b23423dc021fe1301b4` |
| implicit-stage-nonlinear-map | `fbdbe9184a82270a0ce9081b00bae6579f9b0f53da082befa707eeb5f7f4748c` | `f375c2be1e389f3300d185b70a7bc2d1c9d592c131f13aa7d31b1df0931587d2` |
| mixed-linear | `71d46dd85de78e5aa8ca329a19374fb8baea3fc1693bff5b1dcd03da835e5a1d` | `e868f14782300a0c6dc14aadac8991603832218ac0cae9ce94a497b0701df780` |
| mixed-linear-permuted | `5eca82535cea7bdc1dcc27487706f85aec4456d3f82bbf5929a9f0db544aca73` | `9a29dfffe91349f4ae35d3a24f8fc946b0d35cbea30e909fe318eaeb2baae928` |

No new production blocker was reproduced in this bounded review. It is
source/math/emission reception only: no C++ compilation, Kokkos solve, MPI/GPU,
saved-state conservation, temporal convergence, history/checkpoint/retry/regrid
or scientific M06/M13 qualification is established. The declared cell-volume
publication is the mean of a piecewise constant cell representation, not an
assertion about arbitrary unresolved within-cell reconstructions.
