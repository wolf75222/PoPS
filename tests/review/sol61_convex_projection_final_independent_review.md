# Final independent Source review of convex Gauss projection

Author gel reviewed: `e75ea17fb9bcc9015e2de55d360f4f1237edcb9e`, on
`ab026d467b31c84233b46548fec8ccf310d0145f`. The prior affine overflow refusal is
preserved separately in independent gel `453012a94d2c1a2563452ee89d8cc0e7bf14aa55`.
This review worktree changes only an independent test and this note.

**Source review passes; no remaining blocker found within this scope.**

`next_weight = total_weight + weight` is computed at every quadrature sample,
and `total_weight = next_weight` executes outside both the initial-sample and
equal-value branches. Equal samples therefore retain their weights even when
the numerical mean update is skipped. The independent adversary sets every
sample to `0.6` except the final sample `-0.2`; its weighted result matches a
full normalized quadrature reference in dimensions 1, 2 and 3.

For finite same-sign values, the subtraction does not have the previous
opposite-sign overflow. The fraction is between zero and one since every fixed
Gauss weight is positive and the cumulative sum includes it. For opposite signs,
the implementation forms two bounded products with opposite signs before
addition. The original `1.5e308*x` and reordered-expression regressions now give
finite results within their independent exact-zero reference bounds in all
three dimensions. No numerical tolerance or model constant was added to the
production code.

The actual-header-body independent probe checks binary64 maxima, opposite-sign
maxima, adjacent maximum values, arbitrary constants, minimum subnormal and
signed zero. Identical samples preserve the reference bit pattern; mixed zero
signs remain numerical zero (no canonical zero sign is promised for varying
expressions). An independent scaled, full quadrature reference checks 1,000
deterministic random tables per dimension at ordinary and near-maximum amplitude.
Scaling belongs only to this host reference. The author probes independently
retain antiderivative degree1..7 checks and the two original large-amplitude
expressions. Source Gauss nodes, weights, sample order, geometry and exact-integral
branch remain unchanged from the initial proposal.

```sh
env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 \
 /Users/romaindespoulain/miniforge3/envs/pops/bin/python -m pytest --noconftest \
 tests/review/test_sol61_convex_mean_final_independent.py \
 tests/review/test_sol61_constant_projection_host.py \
 tests/review/test_sol61_constant_projection_overflow.py -q --tb=short
```

Result: **3 Source/host tests pass in 3.24s**, no skips. The newly added probe
extracts the actual production quadrature body, substituting only POD evaluator
and geometry interfaces. Host compilation uses C++20 and
`-Wall -Wextra -Werror`. Scalar control flow/arithmetic remains under the existing
`POPS_HD` annotation; device compilation has not been executed.

No Native/JIT/installed environment/build/scientific reception occurs here.
ROOT owns rebuilding the exact artifact and rerunning genuine C++/native tests.
This review does not close the independent Tag/ghost readiness scope or qualify
any current nonconstant runtime checkpoint.
