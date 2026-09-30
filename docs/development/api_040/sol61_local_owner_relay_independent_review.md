# Independent local-owner relay review

Production candidate: `a6cf5c34f529c82a835984d80645ffadb33bb9d7`.
Exact parent: `ab1248be84b53118d30ffc798ab673c2218cb747`.
This review changes tests/documentation only. No author checkout, MAIN, installed
package, SDK/header, native/JIT/build, communicator or physical equation was
modified. The reviewer uses the frozen Git objects, not a moving author tree or
an installed Python package with old methods.

## Actual failure and bounded correction

The genuine closed ROOT SDK375f Dim2 Serial M19 campaign at
`outputs/installed-m19-product-serial-dim2-sdk375f-codec-20261001` has six failures.
Each genuine compilation/publication and bind completed. `_save(initial)` failed
at `RuntimeInstance.local_boxes(block)` with the exact collective failure row:
`('NotImplementedError', 'this runtime provider does not expose rank-owned local boxes', True)`.
The existing coordinate-provider API had no relay on MultiLayoutExecutor. This
was not an observed reduction/lift/PDE or checkpoint numerical failure: execution
never reached those operations. The failed campaign is preserved and cannot be
made green by skipping ownership or modifying expected states.

Candidate production adds `local_boxes` and `local_state` to the coordinator,
forwarding through its exact `executor_for_block`. RuntimeInstance selects that
same child for its local observation and exact spatial-dimension validation.
Single-layout calls still target their direct executor. Ambiguous global
`spatial_shape()` remains a refusal in multi-layout; the fix does not invent one
common geometry or rank0 box. Local methods remain local: their surrounding
caller owns any collective error convergence, as in the authentic fixture.

## Independent checks

`tests/review/test_sol61_local_owner_relay.py` extracts and executes the actual
public method bodies and their exact integer/iterable helpers from the frozen
candidate and parent. Only the Native call boundary is replaced by explicit
rank/owner probes. This is host protocol reception, not execution of Kokkos, MPI,
or a native RuntimeInstance constructor.

**43 independent checks PASS**:

- the exact parent's missing-relay error is reproduced;
- four logical blocks select their native owners, including two sharing a child;
  unrelated native children receive no call;
- shape/dimension comes from the selected child, without a call to the ambiguous
  global getter. Host providers exercise dimensions 1/2/3 and different extents;
  this does not claim a real artifact accepts mixed native compile dimensions;
- ranks 0/1/2 are preserved verbatim. Empty peers return empty native ownership;
  local-state access on an empty peer propagates the native IndexError;
- component-major arrays of widths 1/3/5 and native reversed axis extents return
  by exact object identity, without reshape, transpose, component reduction,
  copy, or invented rank;
- foreign block/missing layout errors occur before a Native call; invalid
  dimension, non-int/bool shape, malformed-rank/int/half-open box and invalid
  index guards remain fail-closed;
- native RuntimeError and OverflowError are the identical exception objects,
  not converted into fabricated success, empty ownership or a fallback state;
- single-layout parent/candidate call traces and raw array bytes are identical;
  existing get/global-state/block/layout-selection bodies are byte-identical.

Boxes are the Native provider's ownership authority. This relay preserves their
native coordinates rather than deriving boxes from an array shape. Legacy box
validation checks dimension, plain integers and increasing half-open bounds;
it does not newly add domain containment or pairwise-overlap validation. The
independent saved-state checker validates those global ownership conditions.
This review does not claim new protection against arbitrary mutation of private
executor ownership dictionaries or concurrent regrid while reading an array.

Commands (no PoPS import, native execution or installation):

```sh
env -u PYTHONPATH /Users/romaindespoulain/miniforge3/envs/pops-api040/bin/python \
  -m pytest -q tests/review/test_sol61_local_owner_relay.py --tb=short
/Users/romaindespoulain/miniforge3/envs/pops-api040/bin/python -m ruff check \
  tests/review/test_sol61_local_owner_relay.py
git diff --check
```

## M19 origin/phase status

The six actual directories currently contain provenance only: there are no
initial/accepted/restored/replayed state or receipt files from that failed
campaign. The closed 13-file requirement remains unchanged (provenance plus four
triples). JUnit is six FAIL and cannot satisfy offline reception.

Actual provenance records have four distinct System binary hashes, for
population/integral/weighted/extended. The two retained fields share a layout,
not a binary package: there are three layout engines/checkpoint children but
four block model DSOs. ROOT execution origins must preserve those four actual
System files; no three-DSO consolidation is justified. This is based on genuine
partial provenance, not on a fabricated future phase. Actual CPP mapping/link
metadata still needs owner-supplied retained paths; absent CPP/link records stay
explicit gaps.

No blocker was demonstrated in this candidate's two production files. Genuine
post-fix Serial/MPI2 execution, actual ownership arrays/checkpoints and finite
product state reception remain pending ROOT. No Vlasov/BGK or continuum field
solve qualification follows from these host checks.
