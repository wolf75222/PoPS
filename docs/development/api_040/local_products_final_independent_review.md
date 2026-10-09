# Independent final review of local residual products

Reviewed production commits `14cd8d2` and `16a14a9`, with the prior review
`d2f607e` as context after reading the implementation. The review checkout
preserves C38 `797abce`; local cherry-picks `1d94437` and `1461461` contain the
same T3 changes. This review changes no production code.

Unknown keys are frozen and sorted before the callback. Each unknown keeps its
exact block and component space. Capture keys are independently sorted; their
SSA occurrences remain separate even when two captures share a block. The
closed DAG maps unknown leaves to the current candidate and capture leaves to
their own field views. A capture is never replaced by the seed or by another
occurrence with the same block. Residual cardinality and role indices are
rechecked in lowering. There is no leading-block elimination.

The callback is inside the existing atomic authoring transaction. A rejected
row or undeclared input restores the Program image. At execution the layout,
distribution and local-rank vote precedes the first `fab(li)` access. Outputs
are private scratches, followed by the existing collective nonlinear report
and explicit consumption before projection. Inspection does not establish
native allocator failure convergence or MPI rollback; those require the
installed acceptance witness on the rebuilt package.

The separate test `test_local_product_independent_review.py` uses three
unknown blocks of widths 1, 2 and 4, a dense nonsymmetric seven-dimensional
matrix, reversed unknown insertion order, and two different SSA captures from
the same block. Its oracle is NumPy's solve of the independently fixed matrix
and an explicit check of every original residual row. The host probe uses
the production product residual emitter and actual C++ nonlinear provider,
without a MultiFab or generated runtime. A separate authoring failure checks
that the full Program hash is unchanged.

The new probe passed **5/5 in 1.89 seconds** after the central performance
window was lifted: one atomic-authoring check and four C++ provider executions
(two capture pairs, two insertion orders). Compilation uses C++20, O2 and
`-fno-fast-math`. Reproduce from the checkout with:

```sh
env -u PYTHONPATH /Users/romaindespoulain/miniforge3/envs/pops-api040-c11/bin/python -m pytest -q -o pythonpath=python tests/python/unit/codegen/test_local_product_independent_review.py
```

No new native hierarchy, MPI, device or installed acceptance is claimed.
No concrete semantic blocker has been established by this source review.
The documented open obligations remain: Program source/apply nodes inside a
product body, multiple unknowns owned by the same exact block, nonlocal and
rectangular problems, and AMR scientific acceptance. The existing Uniform
installed witness must still establish state/time rollback on every rank.
