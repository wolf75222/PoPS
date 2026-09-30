# M04 ForwardEuler: independent regime audit

Base `99d651c182d24aba5aaaf431c5e667f810ad5bb5`. No production runtime,
compiler or header is changed. The linear example and scientific runner add one
explicit regime control. Historical source/native proofs and corpus status are
unchanged. There is no new native execution in this review.

## Actual historical failure re-received

The old owner-published receipt SHA is
`8aab308a08c2a6c6c600db234fc89f3ef37421d9269e1355aea4f12e9b3b1355`,
listed in `evidence/83b2b12/manifest.json`. Its native SHA is
`220b48d3264f413aa5ccd8bba58a8b0d1a6b31bf5bade6faac904b1b7ef8cc87`.
The manifest records historical source `9e9a0e5b` plus its source diff, CPU1,
OpenMP1 and actual Dim2. This is historical artifact evidence; no old DSO is
loaded or accepted as compatible with current source/SDK.

Read-only input was `outputs/m04-xonly-220b-openmp1/states`. The externally
pinned receipt authenticates the three actual NPZ hashes, which are preserved:

| N | Actual NPZ SHA256 | Recomputed L1 | Native versus independent stencil |
|---|---|---|---|
| 32 | `76e616b8c019b9942e8376fb0ac11f5a16d2e0718190b72742e55f61ce409151` | .00348358996373690 | 4.45e-16 |
| 64 | `91b4da3779628840f540508ab7d540927f0d224db18c52c828e65ee5ba5aa2f6` | .00228821307001481 | 2.23e-16 |
| 128 | `e7845d68d0f8cb366b3a9b14c715ac6616a9ddf56946b1308c852dbd77cad649` | .00140683697276028 | 2.23e-16 |

The independent oracle integrates the continuous cosine over each cell and
propagates the **actual initial native row** through a periodic upwind/central
ForwardEuler stencil. A second route multiplies the three stencil-weight mode
amplifications, including the terminal shortened step. This route does not
import the example's Fourier oracle or PoPS. The maximum mode/native defect is
8.89e-16. Recomputed orders are `.6063534245372155` and `.701766241133947`.
Mass, maximum principle, y-invariance, physical inputs, final time, declared dt
and step counts also pass their original checks. The first order guard remains
**failed**. Switching the mathematical temporal equation to SSPRK2 at the same
dt produces native discrepancies `.00622/.00229/.000742`; omitting diffusion
or reversing advection also produces large discrepancies. These countermodels
cannot reinterpret the actual trajectory as another physical/time method.

## Cause and bounded correction

For smooth modes the leading extra upwind diffusivity is `a*h/2`, whereas
ForwardEuler contributes temporal antidiffusion `-a*a*dt/2`. With the original
combined-bound step, `a*dt/h=.9/(1+.02*N)` changes from `.54878` to `.39474`
to `.25281`. The leading cancellation therefore changes substantially across
the grids. The finite-grid error ratio does not measure an unchanged leading
coefficient times h. Exact mode/stencil computations, rather than a truncated
modified equation, establish the numerical errors. Following the same original
step formula to finer grids gives mathematical orders tending toward one; this
is not a new native convergence campaign.

There is no demonstrated native mathematical/source defect: the actual states
realize the requested method. The honest correction is an additional experiment,
**`m04-fe-fixed-courant`**, whose predeclared policy is:

```text
dt = min(.9/(N + .02*N*N), (1/8)/N)
```

This controls the cancellation across the prescribed meshes while honoring the
combined stability bound. It keeps `a=1,D=.01,A=.2,t=.1,N=32/64/128`, the full
tensor `diag(.01,0)`, first-order Rusanov/two-point diffusion, ForwardEuler,
the exact cell-average input, all original L1/conservation/state/time guards,
and order `.7`. It does not reduce the guard, discard N32, add an ODE substitute
or replace ForwardEuler by SSPRK2. A smaller dt can increase the PDE error by
removing part of the accidental cancellation, while making its convergence
ratio interpretable. The predicted results are:

| N | Declared dt | Courant | Combined frequency × dt | Mathematical L1 | Steps |
|---|---|---|---|---|---|
| 32 | .00390625 | .125 | .205 | .00641684745196225 | 26 |
| 64 | .001953125 | .125 | .285 | .00325626666740281 | 52 |
| 128 | .0009765625 | .125 | .445 | .00164122164708620 | 103 |

Predicted orders are `.9786458344358433/.9884487638693287`. These are explicitly
mathematical predictions. They do not close the old failed `m04`, whose receipt,
default step formula and corpus status remain unchanged.

The source checks execute the actual public `author_case` AST and actual step
preflight for all six mesh/policy combinations, resolve the public Case and emit
C++ without compiling it. They receive the actual FixedDt, one RHS evaluation,
one state commit, and the combined diffusion/transport lowering. The unchanged
`author_case` body is pinned against base99d651c1 by normalized AST SHA
`ceee2c39d507204fd316409b8db3a9d0fd874067821bf064d9b12e604f58f3e5`.
The new policy rejects SSPRK2/isotropic selection before authoring. Every old
runner name explicitly selects its original policy, so ambient environment
cannot convert an old campaign into this control.

## Reproduction and future native obligations

Source/unit command (no PoPS build, JIT, installation or runtime):

The coherent source suite received **44 PASS** (6.33s), Ruff passed on all four
Python files, and `git diff --check` passed. Public source emission is evidence
about the selected Program, not evidence that its generated C++ executed.

```bash
rtk proxy env -u PYTHONPATH -u POPS_NATIVE_DIM /Users/romaindespoulain/miniforge3/envs/pops-api040/bin/python -I -c 'import sys; from pathlib import Path; sys.path.insert(0,str(Path.cwd())); sys.path.insert(0,str(Path.cwd()/"python")); import pytest; raise SystemExit(pytest.main(["-q","--tb=short","tests/review/test_sol61_m04_forward_euler_regime.py","tests/python/unit/numerics/test_api040_m04_oracle.py"]))'
rtk proxy env -u PYTHONPATH /Users/romaindespoulain/miniforge3/envs/pops-api040/bin/python -I tests/review/sol61_m04_forward_euler_regime_oracle.py --historical /Users/romaindespoulain/Documents/Codex/2026-09-28/dans-le-d-p-t-pops/outputs/m04-xonly-220b-openmp1/states --output /PRIVATE/m04-audit.json
```

Root owns future native execution with a coherent installed Dim2 module/SDK and
authentic before/after source/native identities, using the existing runner:

```bash
rtk proxy env -u PYTHONPATH /Users/romaindespoulain/miniforge3/envs/pops-api040/bin/python docs/development/api_040/run_scientific_checks.py --case m04-fe-fixed-courant --ranks 1 --threads 1 --output /ROOT/EMPTY-m04-fe-fixed-courant
```

No native control data exists in this review. The future independent reader
requires externally supplied owner receipt SHA and expected native SHA, checks
the three actual native state hashes, original physics and guards, declared dt
policy, maximum principle/conservation/support, original continuous error and
ForwardEuler stencil/mode, and refuses the SSPRK2 counter-equation:

```bash
rtk proxy env -u PYTHONPATH /Users/romaindespoulain/miniforge3/envs/pops-api040/bin/python -I tests/review/sol61_m04_forward_euler_regime_oracle.py --control /ROOT/m04-fe-fixed-courant/states --control-owner-sha256 OWNER_RECEIPT_SHA --control-native-sha256 OWNER_NATIVE_SHA --output /PRIVATE/m04-control-received.json
```

The oracle does not manufacture external owner pins or positive saved native
states. It emits only mathematical reference metrics without those real files.
The report JSON stores this distinction and the old failed receipt. This lot
does not qualify current native execution, MPI2, AMR, GPU, native Dim1 or general
anisotropic/nonlinear diffusion, and never reclassifies historical M04 as passed.
