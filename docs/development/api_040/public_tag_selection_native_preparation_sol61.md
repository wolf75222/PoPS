# Public TagBuffer versus nesting Native test preparation

Read the integrated `hooke_tag_buffer_nesting_causal_review.md`: explicit Buffer is selection policy, while derived nesting buffer/lookahead are parent stencil/transfer coverage. The N8 partial-coverage science guard remains untouched. This private test-only tranche is based on MAIN304fb372 and coordinates Galileo's separate production TagSelection contract; it changes no production file or environment.

Physical declarations are visible at module scope: generic two-component stationary conservation law ddt(U)=-div(0), initial constant1 and cos(2pi x), periodic unit rectangle. The independent mathematical oracle uses the exact conservative cell average sinc(1/nx)*cos(2pi(i+.5)/nx), threshold0.7. On nx8 it selects columns0,7. Buffer0 preserves them; explicit Buffer1 gives columns0,1,6,7 with periodic wrap. Numerical nesting is not read to determine selection. This is a ratio2 rectangular-strip witness: BergerRigoutsos efficiency1/minimum box1 and aligned rectangles justify exact union coverage. It does not assert tags equal patch coverage for arbitrary sparse tags or different clustering policies.

Native node: `tests/python/integration/amr/test_public_tag_selection_native.py::test_public_tag_buffer_preserves_periodic_selection_and_parent_ghosts`. Eight parameter cases: shape8x8/8x12, Buffer0/1, ConservativeLinear+CoarseFineGhostInterpolation or ConservativeInjection+CoarseFineInjection. The rectangular geometry and independent transfer policies exercise ranked numerical coverage without deriving the oracle from their config. Representative node suffix `[linear-0-shape0]` retains N8 with nonzero coarse and fine active cells.

Each Native case requires the actual installed package and validate→resolve→compile→bind with the authentic artifact execution context; no direct-runtime bootstrap, synthetic compiled model or staged NumPy seed is used. Actual coarse marker cells must match the mathematical tag set. Actual native patch union and composite-active cells must equal the selected periodic strips. One actual Euler step exercises spatial halos/coarse/fine ghost machinery. Native POPSCAR1 full-storage bits then require constant1 in every valid and ghost cell, finite marker values, components2 and at least one grown cell per axis. The marker ghost interpolant is not claimed identical to analytic continuum values. Patch topology and stationary clocks remain correct. ROOT receives the live new checkpoint_tag_selection_contract as diagnostic JUnit property alongside package/DSO paths and artifact identity; that config is not the physical selection oracle.

Source verification only: **10 PASS in11.17s**, actual source import path asserted and pops._pops/dimension-native modules absent. Eight real public cases validate/resolve; two NumPy-only mathematical tests prove periodic dilation and reject unwanted full refinement or ignored Buffer1. Source-only collection admitted all eight Native nodes (18 total including10Source). No Native test ran, no JIT/build/runtime/env mutation occurred. ROOT must rebuild Galileo's new native ABI/header contract then execute this node in Serial/MPI2; source success cannot receive it.

Limits: public Buffer currently accepts a scalar cells count, so independent per-axis anisotropic Buffer authoring is unavailable; rectangular8x12 geometry is covered. This bounded fixture uses two levels and deliberately does not claim multilevel>2 nesting or arbitrary sparse cluster coverage. Those require an additional independent parent-domain geometry oracle rather than weakening exact N8 selection or inventing a 3-level result. No actual ghost, backend, performance or MPI qualification exists before ROOT runs it.

Representative installed-package command (ROOT's authentic env/PYTHONPATH policy applies):

```sh
rtk proxy env -u PYTHONPATH /actual/installed/env/bin/python -m pytest -q 'tests/python/integration/amr/test_public_tag_selection_native.py::test_public_tag_buffer_preserves_periodic_selection_and_parent_ghosts[linear-0-shape0]'
```

The env path is an explicit descriptive placeholder, not an installed runtime identity.
