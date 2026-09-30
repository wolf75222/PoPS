# Per-block local runtime observations

Production fix `a6cf5c34f529c82a835984d80645ffadb33bb9d7`, based on
`ab1248be84b53118d30ffc798ab673c2218cb747`, received in a private checkout on
1 October 2026. Only `_multi_layout_executor.py` and `_runtime_instance.py`
change (23 insertions, 5 deletions). No C++/SDK, physical descriptor, mapping,
oracle, numeric value, installed environment, MAIN checkout, build or JIT changes.

Root's actual SDK375f M19 serial run compiled/bound six public cases, then failed
at the initial `_save`: `RuntimeInstance.local_boxes(block)` reported that its
provider lacked local boxes. This is the observed failure supplied by root;
this fix's source checks are not a replay of those native cases.

There were two Python seams. The multi-layout executor had no `local_boxes` or
`local_state` relay, despite already holding the exact block-to-layout-to-child
registry. After adding a relay alone, RuntimeInstance would still ask for a global
`spatial_shape()` to validate the box dimension. The composite intentionally
refuses such a global shape because its child layouts have distinct geometry.

The composite now relays both methods through `executor_for_block(block)`, passing
the original block name and box index to that child. RuntimeInstance selects the
same child once for each observation. For local boxes, both the provider and its
strict spatial-shape validation refer to that selected child. It retains the
existing exact integer/positive shape (rank 1/2/3), exact box rank and integer
half-open bounds with positive extents. Global `spatial_shape()` still refuses
multi-layout geometry. A single-layout provider without a block selector retains
its existing direct route and the same spatial-shape validation.

Local state remains an opaque native result: no copy, reshape, transpose, inferred
component zero, dtype conversion, coordinate change or synthesized rank-zero piece.
The native child retains index-range/storage checks and its existing exception
contract. RuntimeInstance retains its pre-provider rejection of bool, negative and
noninteger box indices. Unknown blocks/missing registered children fail through
the existing exact route; missing observation capabilities refuse explicitly.
No global-state or unrelated-child fallback is introduced. Empty native ownership
is an empty box tuple, and an attempted local piece remains the child's refusal.

## Source checks

The added `test_multi_layout_local_observation.py` executes the actual
RuntimeInstance and composite classes at the published-child authority seam.
Only child storage/query providers are substituted. This deliberately does not
exercise install, transfer preparation, MPI, compilation, evolution or checkpoint.
The source seam includes independent rank-1/2/3 providers to check dimension
routing; it does not qualify mixed native dimension ABIs in one public runtime.

Its **29 checks** cover independent child shapes/boxes, different component widths
1/3/5 and native axis order, exact ndarray object identity, two blocks sharing one
child layout, a selector that would return another child if called twice, unknown
block/missing layout/wrong-child owner, malformed shape, malformed/noninteger/empty
box, empty ownership, invalid index, missing capabilities and original child
exception identity. No fabricated native-positive observation is saved.

Actual commands in the private checkout:

```sh
PY=/Users/romaindespoulain/miniforge3/envs/pops-api040/bin/python
rtk proxy env PYTHONPATH=python PYTHONDONTWRITEBYTECODE=1 "$PY" -m pytest -q tests/python/unit/runtime/test_multi_layout_local_observation.py tests/python/unit/runtime/test_multi_layout_runtime_authorities.py --tb=short
rtk proxy env PYTHONPATH=python PYTHONDONTWRITEBYTECODE=1 "$PY" -m pytest -q tests/python/unit/runtime/test_runtime_instance_gate.py -k local_boxes --tb=short
rtk proxy "$PY" -m ruff check python/pops/runtime/_runtime_instance.py python/pops/runtime/_multi_layout_executor.py tests/python/unit/runtime/test_multi_layout_local_observation.py
rtk git diff --check
```

Results: **54 PASS in 1.60 s** (29 added + 25 existing authority checks), then
**4 existing local-box checks PASS in 1.43 s**, 118 gate cases deselected.
Ruff/diff PASS. These are **58 source checks**, not installed/native reception or CI.

A broader attempted source-only selection included the entire RuntimeInstance
gate: 110 PASS / 64 FAIL. It requires native bootstrap (`from pops import _pops`)
and a selected native catalogue in tests that build install plans; those imports
and preconditions were unavailable in this source-only run. That command is not
qualified as a green gate. No fake catalogue/module or private environment
installation was added to turn it green. The four affected existing local-box
regressions were received separately as above.

Root owns the matching installed Python replay: all six real M19 serial cases,
then all-rank MPI cases, including initial observations, mapping trajectories,
checkpoint/restart and saved-state scientific authentication. This source repair
does not qualify the previously failing native campaign or complete M19 physics.
