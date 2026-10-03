# M19 AMR consumed Field witness preparation

This fixture is a Source preparation for the installed ABI10 capability
`mapped_consumed_field_output_amr`. The Source tests passed; Native execution,
MPI vote/rollback reception, and SDK28 qualification remain to be run against
the newly built installed package. Source emission is not a Native result.

`tests/python/support/m19_amr_consumed_case.py` declares a non-kinetic producer
with two State components and the actual equation-owned Field problem
`phi - 0.125 Delta(phi) = heat_load`. The initial load is
`2 + 0.1 cos(2 pi x) + 0.07 cos(2 pi eta)`. A local Source increases it by
`32 dt`, with `dt = 1/64`. Two independent reservoir blocks consume the solved
Field through public `solve -> observe -> mapping_port -> publish_mapped -> rhs
-> commit` nodes. The support order is `(eta, x)`, while native storage is
`(x, eta)`, and the target rod is stored on native axis 1. Reversing the two
publications is a parameterized test case.

The two physical quadratures use four eta bins and weights `(1/8,1/8,1/4,1/2)`
and `(-1/4,0,1/4,0)`. Both integrals are nontrivial; the positive output varies
along the target rod and the signed output is nonzero. Spectator/marker State
components must retain their exact values. The source grid starts at `(3,4)`
and refines fully to `(6,8)` after the first accepted step because its load
crosses the declared threshold 2.2. The reservoir grid `(1,3)` is already
fully refined to `(1,6)`. Both hierarchies use synchronous common-time execution,
two configured levels and regrid schedule `every(1)`. State transfer explicitly
uses conservative injection and coarse/fine injection.

The independent NumPy reader in `tests/review/sol61_m19_amr_consumed_offline.py`
assembles the periodic two-dimensional cell-centered Helmholtz matrix from
saved source State and computes exact interval overlaps and four-bin weighted
integrals. Its bounded composite scope requires complete finest-level coverage:
the covered coarse level contributes zero physical measure, so the composite
equation reduces to the finest periodic stencil. Partial refinement is rejected,
not approximated by a finest-grid extension. CP12 full carrier geometry, native
refinement boxes, distribution owners, durable valid State bits, NPY bits, clocks,
and topology epochs are checked before numerical comparisons. This scope does
not qualify a mixed coarse/fine interface stencil or retained Field history.

The fault DSO is derived from the genuine generated physical support integral
component. It preserves the real callback and authenticated physical-map
signature, declares its test extension, and only after successful integral
evaluation writes a NaN candidate on the elected observed rank. Serial and MPI
tests must observe that actual callback, elect the highest observed rank and
record agreement. They require every rank to report finite-guard rejection and
retain byte-identical State carriers, clock, epoch, topology, accepted contract,
temporal state, auxiliary/Field payloads and mapping evaluation authority.
There is no mock runtime/backend and no product specialization for this case.

The runtime test uses the actual public `validate -> resolve -> compile -> bind
-> run` chain. Every phase (`initial`, `accepted1`, `accepted2`,
`before-failure`, `after-failure`) retains a composite checkpoint containing
CP12 children, explicit finest-level NPY observations, their SHA256 pins,
common clock, level/epoch authority, and consumer cursors. Compilation/native
identity and a C25-before-bind receipt are saved separately. SHA256 pins
establish integrity; ROOT reception supplies external runtime/source authority.

Source reproduction from the checkout (the Python environment's installed
package can predate this source, so the insertion below deliberately selects
checkout Python only for this preparation):

```sh
rtk proxy env -u PYTHONPATH /Users/romaindespoulain/miniforge3/envs/pops-api040-ir17/bin/python -c 'import sys; sys.path.insert(0,"python"); import pytest; raise SystemExit(pytest.main(["-q","tests/review/test_sol61_m19_amr_consumed_preparation.py"]))'
```

Result: **5 passed**, covering both publication orders and manufactured periodic
stencils on `(4,3)` and `(8,6)`. Adversaries reject constant replacement,
unsolved load, transposed axes, stale Field values, wrong State component and NaN.
The preparation does not compile a DSO or bind/run a native runtime.

After ROOT has installed and authenticated SDK28, its selected runtime Python
must import PoPS from the environment prefix and expose ABI10 plus the AMR
consumed-Field capability. Run both serial and MPI2 with its normal compiler and
MPI fixtures; do not insert checkout Python for these installed-package runs:

```sh
rtk proxy env -u PYTHONPATH "$POPS_RUNTIME_PY" -m pytest -q tests/python/integration/runtime/test_m19_amr_consumed_runtime.py
rtk proxy mpiexec -n 2 env -u PYTHONPATH "$POPS_RUNTIME_PY" -m pytest -q tests/python/integration/runtime/test_m19_amr_consumed_runtime.py
```

For each receipt directory, an independent saved-data audit is available:

```sh
rtk proxy env -u PYTHONPATH "$POPS_RUNTIME_PY" -m tests.review.sol61_m19_amr_consumed_offline /absolute/path/to/amr-consumed-field
```

The reader never marks `Native_received` or `ROOT_approved` true. ROOT must bind
the resulting actual serial/MPI receipts to the exact SDK/source/artifact pins,
inspect the real failure/injection records, and decide the qualification claim.
