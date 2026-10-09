# Independent M10 review of 2355636

Reviewed by Astra, independently of the author, against the mathematical
handoff `notes_on_the_abstraction_of_pops.tex:1246–1276` and the original M10
registry entry. This review changes tests and their scope documentation only.

## Finding corrected before native reception

`System::program_exchange_records()` returns the local accepted ledger
(`src/runtime/system/system.cpp:251`), and
`PreparedDiffusion::stage_accepted_exchanges` traverses only local owned cells.
The original native test incorrectly required all `2*N` incidences on each MPI
rank. It also wrote the same JSON file from every rank and asserted local
acceptance before subsequent collective state/history reads.

The test now gathers the ledger without deduplication, checks exactly one lower
and upper incidence per cell and converges assertion/I/O failures through the
artifact's actual execution context. Only rank zero writes the receipt. The
independent control tests exercise partitioned and empty-rank lists, duplicate
ownership, orientation, multiplicity, temporal weight and receipt I/O error.
They are Python control tests, not an MPI execution claim.

## Equations, representation and discrimination

The handoff uses `j=D/h*(B(delta)*nL-B(-delta)*nR)` and `n_t=-div(j)`.
The author's returned face value is `G=-j`, used under `+div(G)`. The signs are
consistent. The prescribed background satisfies `-Delta_h psi=n*-b`; the
perturbed load uses the actual density occurrence and the unchanged background.
Their identities and stage zero are checked in the public Program, including
the field component stored as history.

The independent oracle solves the full centered periodic Poisson matrix with a
mean-zero constraint. It solves the fitted edge relation using quadrature of
the integrating factor, rather than the author's Bernoulli/eigenvalue formulas
or any PoPS expressions/code generation. At the frozen N=32, D=.1, dt=1/(16N²):

| Check | Measured source-only value |
| --- | ---: |
| Poisson potential max error | 4.9960e-15 |
| Discrete equilibrium max flux | 4.4195e-15 |
| Fresh versus stale one-step state gap | 7.4232e-7 |
| Wrong Poisson sign state gap | 2.9494e-4 |
| Added duplicate diffusion state gap | 2.1593e-4 |
| Mean periodic charge | -6.9389e-16 |

The original `2e-11` state/history tolerance and `1e-8` stale discrimination
threshold remain unchanged and are checked before native compilation. The stale
gap is more than 37,000 times the state tolerance. Incidence checks additionally
authenticate face values, signed time integration and exact joint coverage.

`n*_i=exp(-psi_i)` prescribes discrete cell averages of a piecewise constant
density, not exact cell averages of the smooth exponential profile. An independent
Gauss rule distinguishes those arrays by more than 1e-4; substituting the smooth
profile's true averages no longer yields zero fitted flux. The documentation now
states this explicitly. The fixed `b` is signed charge (minimum about -14.9933),
not a positive material density. Periodic neutrality is closed, with Poisson
coefficient one. No quasi-neutral limit, continuum convergence, wall condition,
arbitrary positivity or entropy property follows from this variant.

## Actual evidence and remaining reception

Command, from the isolated checkout:

```sh
env -u PYTHONPATH /Users/romaindespoulain/miniforge3/envs/pops-api040-c11/bin/python \
  -m pytest -q --tb=short -o pythonpath=python \
  tests/python/unit/codegen/test_m10_independent_review.py \
  tests/python/integration/runtime/test_m10_self_consistent_sg.py -k 'not native_self'
```

Result: **11 passed, 1 deselected**, 2.65 s. This includes the author's complete
public source validation/resolution/emission and the independent tests.
No JIT, installed simulation or MPI run was performed. Native Dim1 serial and
MPI reception, including a rank with no boxes, remain required. One accepted
step reads the stored stage observation in slot 1; it does not qualify history
rotation over multiple distinct observations. The unperturbed equilibrium here
has a source-only zero-flux proof, not a separate native equilibrium run.
