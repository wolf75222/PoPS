# Failed AMR cache candidate: independent source critique

Exact target: `69e29d44e480e4583c73d9af25d5bb3e0c78197d`. This commit
changes only the integration fixture, not the cache or prepared diffusion.
Review was source-only: no native/MPI, build, SDK or environment mutation.

The corrected expectation matches the actual cache atomicity contract.
`PreparedResourceCache::acquire_lease` retains the old holder while constructing
a separate candidate. Candidate allocation and construction each have their own
collective error vote. Only after both succeed does it revoke the prior version
and publish the candidate holder. A failed construction must therefore preserve
the old object identity, buffer identity, version and evaluated contents on all
ranks, including a peer whose candidate constructor succeeded locally.

The existing native unit test
`PreparedResourceCache.FailedConstructionPreservesAcceptedBufferOnEveryRank`
already asserts that exact rule with the same old resource pointer, old buffer
pointer and values 17 at both ends of its buffer, followed by a successful new
construction. Expecting the AMR integration fixture's old frequency read to
throw after the failed replacement contradicted that rule.

The revised fixture first tests successful reconstruction after a rank-local
cache clear: the new resource is unevaluated and its frequency read must throw.
It then evaluates coefficient .1, requiring twice the old .05 frequency.
After the invalid-ghost candidate refuses collectively, reacquisition of the
original valid preparation must return the exact `rebuilt` object with its last
successful `.1` frequency. Finally it explicitly evaluates `.05` again and
requires the original frequency. No tolerance is enlarged; object identity and
both numeric equalities remain strict.

Preserved storage is not permission to publish a new law without evaluating it.
The cache header explicitly states the caller's reevaluation obligation. The
generated `_emit_diffusive_rhs` reacquires storage, prepares state/providers and
calls `apply` before reading the new frequency or issuing accepted exchanges.
`PreparedDiffusion::apply_impl_` clears `evaluated_` immediately, even before
input/output storage validation, and marks it true only after face preparation,
frequency reduction and their collective error fences have succeeded. A failed
**evaluation** thus differs from failed **replacement construction**: the former
invalidates the attempted image; the latter preserves the prior valid image.

This fixture correction is not a weakening of that source sequence. The cache
itself does not enforce freshness of a law merely by handing out a cache hit;
manual low-level callers retain the documented obligation. This source review
does not establish that every possible handwritten native caller obeys it.
No new production defect was demonstrated in the reviewed generated path.

Root's native169 serial/MPI2 receipts are external evidence supplied to this
review, not rerun or newly qualified here. Their post-fix reception remains root's
responsibility. Useful native targets remain `test_prepared_resource_cache` and
`test_amr_program_diffusion`, including failed first construction, peer failure,
successful replacement, failed evaluation and accepted ledger rollback cases.
