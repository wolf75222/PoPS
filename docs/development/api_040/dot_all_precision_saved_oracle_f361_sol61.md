# Independent saved-state oracle for dot_all precision

Exact external native receipt: source
`f36176fa2a822c67b5ccc114bd21de1bef4be23e`, SDK headers
`506dce78009f1c0822b40ae1c38d9849eb9fbc2b65051906c4b2de4377039aaf`,
Dim2 native extension SHA256
`121bac0fd893427742e02a31437451a15f9ac7099274993c72d73c52fd7082f7`.
Evidence directory:
`outputs/installed-computed-dt-bound-public-dim2-sdk506-20260930`.
Exclusive review tree `work/PoPS-sol61-dot-all-precision-review`, branch
`codex/api040-sol61-dot-all-precision-review`.

The independent script imports NumPy and standard Python only; it does not
import or execute PoPS. It authenticates both complete NPZ hashes, the source
receipt, their exact SDK ABI equality, state shape/dtype/component names,
finite values, macro step and requested/reached duration receipts. It derives
the second Heun increment from the genuine stored first accepted state using
the authored operation order. It then forms ordinary individually rounded
binary64 products and sums their exact rational values with `Fraction`.
`math.fsum` agrees, but is not the independent reference implementation.

| Saved state | NPZ SHA256 | Requested h | Old endpoint ULP | Oracle endpoint ULP |
| --- | --- | --- | --- | --- |
| ro0, t=.8 | 6f99f83b77ab0971be3c97134276ec52dd067e09df5ccd7b8433babc33d8247c | 1 | -5 | -1 |
| ro1, t=.47058823529411764 | 9c1fef4ee223dd73a527926772993e9aaf0ba77fab2ee0ec0e708e8ccb582138 | .5 | -10 | 0 |

The targets remain respectively `1.6` and `0.9411764705882353`, and the actual
consumer's four-ULP endpoint rule remains unchanged. The checkpoint fields
are `(2,8,8)`, with first-cell values `[.6,.8]` and
`[.8823529411764706,.47058823529411764]`.

The authentic old source accumulates 64 cell products *separately per
component*, then merges the component/patch sums with ordinary `+=`. This
grouping reproduces both external errors exactly. A single naive loop over
all 128 products instead yields different endpoint errors, so that loop must
not be substituted for the actual source reduction tree. Python 3.12's
`sum(float)` is compensated; the old-order replay deliberately uses explicit
`s += value` loops.

`dot_all_precision_saved_oracle_f361_sol61.json` saves all products in hex,
both contractions/endpoints, checkpoint and identity hashes, native receipts,
and the independent synthetic counterexamples. Those cover six permutations
of `[1e16,1,-1e16]`, intersection of active/coverage masks, ignored invalid
values outside the domain, invalid masks even when the other mask excludes
the cell, invalid active inputs, finite-input product overflow, and finite
products whose exact sum overflows binary64.

A concrete remaining MPI limit is recorded separately: rank0 products
`[1e16,1]`, rank1 products `[-1e16]`. Each input/product is finite; each exact
local sum rounds to finite binary64. Finalized local scalars are therefore
`[1e16,-1e16]`, and ordinary MPI SUM returns zero although the exact sum of
the products is one. Improving local accumulation alone cannot retain that
lost cross-rank low part. The initial author proposal bounded its correction
to local Kokkos/patch/component/AMR-level accumulation and left MPI SUM
unchanged. After this counterexample, integration authorized carrying both
high/low words through the exact execution lane and merging them by rank.
That actual implementation requires reception after its frozen SHA; no MPI
partition invariance or globally compensated native contraction is qualified
by this offline oracle.

Command, run in the exclusive review checkout:

```sh
rtk proxy env -u PYTHONPATH \
  /Users/romaindespoulain/miniforge3/envs/pops-api040/bin/python -I \
  tests/review/sol61_dot_all_saved_state_oracle.py \
  --evidence /Users/romaindespoulain/Documents/Codex/2026-09-28/dans-le-d-p-t-pops/outputs/installed-computed-dt-bound-public-dim2-sdk506-20260930 \
  --output outputs/sol61-dot-all-precision-independent/oracle.json
```

Exit 0: two authenticated saved cases; old ULP `[-5,-10]`, exact-products
oracle ULP `[-1,0]`; the all-finite MPI cancellation witness remains `[0,1]`
(ordinary rank sum, exact sum). These are offline algebra/source findings,
not a new native reception. No tolerance, model equation, legacy dot contract,
MAIN, SDK, environment, C++ header, build or JIT was changed. Review of the
author's actual reducer/join implementation must follow its frozen SHA.
