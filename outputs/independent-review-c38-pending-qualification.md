# Independent review: C38 retained pending history

Reviewed the frozen C38 core commit `797abce` in `PoPS-principal-group`, with
the separate real-regrid adversarial fixture commit `454ebb7`. This is a
source/host review. Installed hierarchy, MPI and rollback tests await the
central rebuilt package.

The original failure was an unconsumed child-1 lag created at epoch E+1 that
was rejected when a subsequent child-2 publication moved the hierarchy to
E+2. The change keeps the creation `prior_*` and `published_*` edge immutable,
adds `qualified_*` for the current hierarchy, and advances it only across an
immediately adjacent engine-prepared publication. `pending_qualification_valid`
requires topology and materialization distances from the creation edge to
agree. A retained marker is requalified only if its child is at or below the
new parent, its ring is absent from the affected plan, its old qualification
matches the descriptor's prior edge, and its exact live ring contract and
slot-1 publication sample are unchanged. The staged map is private until
collective validation and accepted-image publication succeed.

The captured ring contract includes owner, state/space/clock/interpolation
identity, both slot indices/dt/fill/sample values and accepted clock. It
excludes topology generations deliberately. The serializer rejects a marker
whose captured contract or sample differs from the accepted image; import
checks current qualification, slot-1 sample and live restored history
provenance; the deferred reader checks current qualification and the retained
slot-1 sample while allowing a fresh slot-0 store. The exact cross-rank
contract includes operation identity and every marker field. These are
fail-closed source guards, not a proof that a later native remap succeeds.

Native accepted-state wire changes from `POPSAND8` to `POPSAND9` and rejects
AND8 instead of inventing origin/qualification. The Python checkpoint envelope
is unchanged; the test-only inspector recognizes both formats. The added
binary ring contract is length-framed. Capacity uses the same field encoder
through `CountingWriter`; AstraProtocols separately checked three rings, five
levels, long identities and overflow in host harness `74686bb`. The author
reports 3/3 C++ host tests, 16/16 Python inspector tests and an instantiated
AMR context syntax-only check on the matching SDK. None is a true MPI run.

My independent fixture `454ebb7` extends the actual three-level synthetic
loader, rather than a pure candidate-only helper. An exported ordinal hook
refuses rank zero's second resource refresh after the first child publication.
It requires every rank to reject, rolls back to exactly the pre-regrid boxes,
epoch, accepted wire/revision, exchanges, flux shard, clocks, cell states and
history rings, then retries the same restoration without the hook and retains
the original positive descendant coverage/value assertions. The old boolean
hook keeps its behavior. This fixture passed `git diff --check`; it has **not**
been compiled or executed yet, so its rollback and retry claims are test
requirements, not observed results.

Verdict: the frozen core's authority split and checks are coherent at source
level; no concrete bypass was found in this bounded review. Acceptance remains
open on the real three-level serial/MPI test and the prior red restart receipt
on a rebuilt artifact. A failure there must be diagnosed rather than
interpreted as an approved relaxation of retained-history authority.
