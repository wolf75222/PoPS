# M03 Euler/EOS - source and oracle receipt (pre-native)

Date: 2026-09-29. This is a source/analytic receipt, **not** an installed native
Euler run or a Dim=1 qualification. The independent review belongs to GPT-6 Sol;
the additional adversarial review by Astra Protocols is separate.

## Provenance and decision

- Corpus: `docs/development/api_040/corpus.json`, M03. Prescribed ideal-gas Sod
  states `(rho,u,p)_L=(1,0,1)`, `(rho,u,p)_R=(.125,0,.1)`, `x in [-.5,.5]`,
  `N=100/200/400`, `t=.15`.
- Reference N05: `PoPS_Codex_handoff_0.4.0/reference/PoPS_API_v0.4.0/examples/cases.py`
  `euler_entropy` defines `p=(gamma-1)(E-|m|²/(2rho))-gamma*p_inf` and
  `c²=gamma(p+p_inf)/rho`. `tests/check_numerics.py` tests smooth entropy
  waves, including `(gamma,p_inf)=(1.4,.3)` in 2D; it does not qualify Sod.
- Second EOS choice: **stiffened gas `(gamma,p_inf)=(1.4,.3)`**, the N05 2D
  parameter pair. N05 also tests `(1.6,2)` in 1D, but its right Sod shock has
  speed above 5, crossing `x=.5` before `t=.15`. The chosen shock speed is
  `2.5635505795`; its location at `t=.15` is `0.3845325869`, inside the
  prescribed domain. This decision was made before native execution.
- `examples/migration/scientific/api040_m03_euler_eos.py` authors all four Euler
  conserved components, both physical fluxes, EOS, and eigenvalue tuples through
  public Python `Model`, `DiscretizationPlan`, `FiniteVolume`, `SSPRK2`,
  `AdaptiveCFL`, `TransportBoundarySet`, `Outflow`, and `PeriodicAxes`. The state
  and component identifiers are noncanonical and scenario dependent. Both EOS
  coefficients are actual `ConstParam` handles (`adiabatic_index` and
  `background_pressure`), consumed through `model.value`. Environment settings
  may rename the state, all four components, both parameters, and the block;
  an alternate-name source resolution was checked. The
  solver is Rusanov with first-order reconstruction. Physical x boundaries are
  outflow and y is periodic. Native array order is `(component,y,x)`.
- `examples/migration/scientific/api040_m03_riemann_oracle.py` contains an
  independent exact self-similar Riemann solution. It solves the pressure-star
  residual by monotone bisection. For stiffened gas the shock and rarefaction
  relations use shifted pressure `P=p+p_inf`; it integrates conservative
  variables with Gauss-Legendre quadrature after splitting each cell at every
  wave and contact. The authored PoPS flux does not import this oracle.

## Predeclared criteria

The example records these thresholds in `CRITERIA` before compilation: density
L1 cell-mean error at N=100/200/400 below `.09/.07/.055` and strictly
decreasing; mass and total-energy defects below `2e-10`; longitudinal
momentum balance defect below `3e-9` versus the exact boundary-flux integral
`t(p_L-p_R)`; transverse momentum and y-variation below `2e-11`; initial
bind error below `1e-12`; final-time error below `1e-12`. Every saved state
must be finite with positive density, internal-energy density, and pressure.
No clipping or floor is used. The bounds are acceptance choices awaiting an
installed run; they are not claimed observed results.

Each full-domain state is saved as `state_N.npz` before statistics are computed.
The statistics reopen those bytes. The receipt includes the saved file hash,
actual native rank/context/run report, and Dim=2 scope. Both EOS need separate
invocations (`POPS_API040_M03_CASE=ideal` and `stiffened`). At N×N, the y-invariant
exact solution exercises a Dim=2 implementation and does not establish Dim=1.

## Checks completed

- Pure oracle `pytest`: **4 passed**, both EOS. Ideal Sod star pressure and
  velocity match standard exact values `0.303130178050647` and
  `0.92745262004895`; each EOS's integrated exact state obeys mass, energy,
  and boundary momentum balances. Quadrature orders 24 and 48 agree within
  `2e-12` at N=200.
- Independent adversarial oracle checks from Astra Protocols: **21 passed**,
  covering Rankine-Hugoniot including stiffened total energy and reversed
  waves, both rarefaction invariants, pressure shift, Galilean transformation,
  and cells split by the initial diaphragm. Combined oracle total: **25 passed**.
- Installed public Python authoring: `pops.validate` and `pops.resolve` pass
  for both EOS on 8×8 with the mixed outflow/periodic layout. This authoring
  probe does **not** compile a native artifact or run one. The ideal case also
  resolves with alternate names `w/(r,px,py,total)/(kappa,offset)` and block
  `another_fluid`.
- Independent code-generation inspection emitted C++ headers for both EOS with
  renamed state/components/parameter handles. It found the intended four
  conserved slots, `p=(1.4-1)(E-|m|²/(2rho))-1.4p_inf`, and both directional
  wave expressions in `outputs/astra-m03-ideal-renamed.hpp` and
  `outputs/astra-m03-stiffened-renamed.hpp`. This is generated-source evidence,
  not a C++ compile or native solution.
- Python compilation of example, oracle, and test files passes.

The pure test command was
`env -u PYTHONPATH /Users/romaindespoulain/miniforge3/envs/pops-api040/bin/python -m pytest -q -o pythonpath= tests/python/unit/numerics/test_api040_m03_riemann_oracle.py tests/python/unit/numerics/test_api040_m03_riemann_oracle_adversarial.py`.
`POPS_API040_M03_AUTHORING_ONLY=1` runs the source-only public validation and
resolution path for either value of `POPS_API040_M03_CASE`.

Pending root-owned installed Dim=2 compile/runtime reception at all three
resolutions for each EOS. A failed predeclared threshold must be reported as a
failure with the persisted state; it is not permission to relax the threshold.
