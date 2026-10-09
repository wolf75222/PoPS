# M22/H05: closed normalized two-reservoir subcase

Status: source authoring and oracle tests executed; **native science not_executed**.
This slice does not close or qualify full M22 radiation coupled to matter.

The exact closed data are in the handoff's
`reference/PoPS_API_v0.4.0/baseline/test_registry.json`, M22, and the historical
`legacy/v0.2.0/tests/run_revision.py`, H05. They prescribe the linear normalized
exchange, initial `(E,T)=(2,0.5)`, `k=0.8`, and the backward-Euler test `dt=0.4`:

```
E' = -k (E-T),  T' = k (E-T).
E1-E0 + dt*k*(E1-T1) = 0,
T1-T0 - dt*k*(E1-T1) = 0.
```

Here T is the second normalized energy reservoir, not an asserted material
temperature/EOS. The equations conserve E+T and preserve nonnegative energies for
k>=0. There is no clipping. Native Program guards check the initial and candidate
energies and the rebound rate. Numerical nonfinite failures remain hard failures.

`examples/migration/scientific/api040_m22_h05.py` authors two distinct Models and
Blocks with homonymous scalar state declarations. The joint `LocalResidual`
captures old E, old T and the fixed k field independently of its Newton seeds.
The existing generic per-cell source evaluation materializes k as an equation
input; it adds no extra physical source to either reservoir. The shared prepared
nonlinear provider evaluates both original equations before outcome consumption
and atomic commit. No model-specific emitter, enum or solver was added.

The oracle imports no PoPS code. Each discrete step solves the independent 2x2
matrix with NumPy; the corpus one-step answer is exactly `(70/41,65/82)`. The
continuous solution uses conserved mean and exponentially decaying difference,
independently checked against SciPy `expm`. Runs at t=0.4 with 1,2,4,8,16 steps
measure the expected first temporal order. Constant initial data are exact cell
averages on the 4x4 periodic grid; there is no spatial convergence claim.

Criteria fixed before native execution:

| Measurement | Criterion |
|---|---:|
| State versus discrete 2x2 oracle, max norm | <=2e-11 |
| Cellwise E+T defect | <=2e-11 |
| Minimum reservoir energy | >=0 |
| Bound initial state discrepancy | <=1e-14 |
| Final time discrepancy | <=1e-14 |
| Observed temporal order at successive refinements | 0.8 to 1.1 |
| Canonical/reversed Block-order counterpart discrepancy | <=2e-11 |
| Accepted/rejected steps | exactly prescribed count / zero |

The script gathers real accepted native states, saves and reopens NPZ files before
computing metrics, records their hashes, source semantic identities, native hash,
ABI and exact execution contexts, and compares both permutations. Root-only
archive/check failures are broadcast before another collective operation.

The installed pytest witness `test_api040_m22_h05_runtime.py` separately prepares
two permutations, rebinds k=0.8 and k=0, and injects k=-0.8. Invalid k must refuse
on every rank with both states and time unchanged. This witness was collected,
not executed; root must rebuild the T3 product provider and run it serial/MPI2.

Scalar closure tests are deliberately separate. The corpus specifies
`chi(f)=(3+4*f*f)/(5+2*sqrt(4-3*f*f))`, f=|F|/(cE). An independently rationalized
formula checks isotropy, a nearly limiting beam, the limiting beam, rotations and
component permutations. Domain: finite E>0,c>0, finite F, |F|<=cE. Vacuum and
non-realizable data are refused, without a projection back into the cone.
These tests qualify only the scalar closure identity, not a pressure-tensor flux,
transport PDE, diffusion limit or matter interaction.

Missing full M22 specification in the supplied corpus: material energy/temperature
relation, radiation constant and dimensional normalization, exact matter/radiation
source (the corpus explicitly prohibits replacing physical T^4 by this linear
exchange), absorption/scattering opacities and discontinuity data, complete moment
flux/tensor convention, Marshak initial and boundary data, and the reference
transport/diffusion limiting problem. H05 does not supply any of those choices.

Executed: 13 oracle/source tests, canonical/reversed validate→resolve→model-graph
emission, Ruff. Prepared: two installed integration variants. No shared environment,
production core or native installation changed for this science slice.

Source check:

```sh
env -u PYTHONPATH /Users/romaindespoulain/miniforge3/envs/pops-api040/bin/python -m pytest -q \
 -o pythonpath=python tests/python/unit/physics/test_api040_m22_h05.py
```

Native reception, only after the centrally authenticated rebuild:

```sh
env -u PYTHONPATH python examples/migration/scientific/api040_m22_h05.py
```
