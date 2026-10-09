# Local original-residual products, contract v1

`LocalResidual(residual, initial={"a": seed_a, "b": seed_b},
captures={"old_a": a.n, "old_b": b.n})` authors one square local equation over a
product of complete State values. The body receives `(P, z, **captures)` with an
immutable named mapping `z`. It returns exactly the same keys, each containing
the corresponding tuple of scalar residual expressions. The historical single
State API and its optional seed argument remain unchanged.

Packing is lexicographic by explicit unknown key, then the declaration's component
order. Captures are sorted by explicit key and bound to exact Program inputs,
independently of the seed. There is no limit of two blocks or five components.
An unknown must have a distinct exact BlockHandle. Shapes, units and names may
differ; support, frame, sampling, centering and clock must agree. Unknown seeds
share an exact evaluation point; frozen captures may refer to earlier points on
that clock. Every output retains its own StateSpace, point and owner.

At execution the generated coupled kernel collectively compares the full
MultiFab layout, distribution and local rank of every seed and capture before
indexing any `fab(li)`. Mismatches require an explicit map/redistribution; no
implicit interpolation or densification is performed. The same check now protects
the existing coupled implicit Euler shell.

The body executes once at authoring. Its closed DAG uses the Program's common
checked expression emitter, including lazy `where` and `rounded` barriers. Hidden
outer Program reads are rejected: equation inputs must be explicit captures.
All original residual components go to the existing prepared local nonlinear
provider. No block is assumed invertible and no Schur reduction is inserted.
The provider's original-residual convergence criterion remains authoritative.
Outputs occupy attempt-local scratches; the collective report is consumed before
any public projection/commit. The consumed result is indexed by exact BlockHandle,
as for existing coupled solves.

This is a local provider for the named `LocalResidual` protocol, not a claim that
the general `SolveRequest` unknown-product/global adapter is complete. Current
implementation gaps: product bodies containing `P.source`/`P.apply` or other
Program nodes are explicitly refused; only direct component DAGs are lowered.
These are remaining implementation obligations, not physical restrictions.
Rectangular least-squares equations, arbitrary elimination/reconstruction,
nonlocal residuals, compatibility and joint nullspaces are not supplied by this
slice. AMR coarse/fine scientific qualification is not inferred from the shared
emitter. The installed acceptance witness initially targets Uniform 2D.

Evidence: `test_local_residual_product.py` exercises authoring and detached
validate/resolve/model-graph emission; `test_local_residual_product_host.py`
executes the actual emitted residual and native prepared provider, including a
singular leading diagonal block and invertible full 2+3 system, changed seeds,
capture rebind, an unsolvable last row and a masked invalid intermediate.
`test_local_residual_product_runtime.py` is the installed Uniform/MPI witness for
block permutation, rebind and no partial publication after failure. Its execution
requires the centrally rebuilt artifact and is not implied by source/host tests.
