# Independent review of affine moment library `8bc8406`

I reviewed the seven-file change integrated at `005d287` without modifying
its implementation or scientific fixtures. The recurrence expands
`(b_i + Σ_j A_ij v_j)^α_i` one parent multi-index at a time; independent
weighted-particle tests verify 1V/2V order six and 3V order three. The
zero-index output returns its input object by identity, including signed zero,
without reading the matrix. No model name or fixed moment width appears in the
body. Ordinary expression DAG rules govern lazy branches and active nonfinite
values. I independently ran the two new source/host files: **22 passed in
4.73 s**. The native Uniform and MPI fixtures remain for the central rebuilt
package; the original author's MPI assumptions are being corrected separately.

One bounded counterexample was reproduced. `polynomial(index)` was recursive and
memoization depends on traversal order. With the complete 1V basis through
degree 1000, input raw moments `[1,0,…,0]`, `A=1`, `b=0`, ascending indices
returned 1001 outputs. Reversing exactly the same indices raised
`RecursionError: maximum recursion depth exceeded` before constructing a
result. The physics, basis and coefficients are unchanged; only storage order
differs. This was an implicit Python recursion ceiling inconsistent with the
documented arbitrary-order/permutation surface, even though the tests at
orders five and six pass. The follow-up replaces only the recursive parent
walk with an explicit post-order stack. It keeps the same `axis=max(...)`,
per-parent iteration, term accumulation and final lexicographic sum. A new
test checks exact output equality for ascending and descending order-1000
bases. The initial counterexample failed before this change; the follow-up
test and original source/host suite then passed **23/23**. This is an
authoring correction, not a native-kernel or moderate-order numerical change.

Reproducer (run against integrated source, no native build):

```python
from pops.moments import affine_push_forward
n = 1000
ascending = tuple((i,) for i in range(n + 1))
descending = ascending[::-1]
assert len(affine_push_forward([1.] + [0.]*n, indices=ascending,
                               matrix=((1.,),), offset=(0.,))) == n + 1
assert tuple(reversed(affine_push_forward([0.]*n + [1.], indices=descending,
                    matrix=((1.,),), offset=(0.,)))) == tuple([1.] + [0.]*n)
```

I did not run installed PoPS, MPI, GPU or a complete acceptance campaign.
