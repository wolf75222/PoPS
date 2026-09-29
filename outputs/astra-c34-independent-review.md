# C34 independent publication review

Reviewed Sol's original witness and in-progress reservation/CAS changes in the isolated
`PoPS-numerical-bodies` checkout. No MAIN/core changes and no native build by reviewer.
New reviewer-owned file: `tests/python/unit/runtime/test_consumer_publication_c34_adversarial.py`.

## Observations and changes made by Sol during review

- Shared `ConsumerCursorAuthority` is mandatory in `ConsumerTransaction`; the sole production
  constructor in `RuntimeInstance._stage_consumers` supplies its controller-owned authority.
  There is no implicit private-authority fallback. Reservation compares all selected cursor
  values under one lock before any `publish`, and remains held through acceptance until seal
  or successful compensation. Disjoint cursor updates merge into the current shared set.
- Initial implementation lacked an incarnation: reset to identical cursor values could leave
  an already-prepared old payload eligible to publish. Review requested a non-restorable epoch;
  Sol added epoch capture and reservation validation. The actual prepared-root/reset/retry
  test passes. Sol applied the change before the test's first run, so no executed red is claimed.
- `_restore_step_envelope` initially restored the entire snapshot cursor set after local
  compensation. This could erase the cursor of a sealed disjoint NPZ publication. Sol changed
  it to read the surviving shared authority, and preserved disjoint reports too. The reviewer
  test calls the real controller methods, authentic cursor authority, transactions and NPZ
  writer, with an empty executor; it therefore proves the envelope seam, not PDE concurrency.

## Independent evidence

Ten tests passed in 0.71 s using `env -u PYTHONPATH` and the dedicated `pops-api040-bodies`
interpreter with `pytest -o pythonpath=python`. A later added assertion that the disjoint report
survives controller rollback also passed in its targeted replay.

Tests cover actual NPZ bytes and cursors for:

1. Disjoint publication retained when an earlier root compensates.
2. Conflicting root rejected before its NPZ becomes visible, both while the winner remains
   reserved and after its seal (stale cursor).
3. Two simultaneous threads: exactly one visible conflicting publication.
4. Checkpoint reset revokes prepared old-incarnation roots even with identical cursor values;
   a freshly constructed attempt succeeds.
5. Compensated attempt followed by fresh retry advances the cursor once.
6. Missing explicit shared authority rejected before preparation/publication.
7. Failure injected after the second real NPZ publication compensates both outputs of that
   root, preserves the sealed disjoint artifact/cursor, and permits a fresh successful retry.
8. Real controller envelope rollback preserves the disjoint artifact, cursor and report.
9. Incomplete compensation retains reservations; another conflicting NPZ and reset are blocked.

`git diff --check` passed. No installed-native, MPI, distributed CAS, checkpoint-native restore,
or concurrent-PDE qualification is inferred from these tests.

## Follow-up closure: terminal compensation and checkpoint CAS

Root requested an explicit outcome for incomplete compensation. Sol chose a terminal consumer
contract: future publication and checkpoint reset report `incomplete consumer compensation;
recover artifacts and recreate RuntimeInstance`. Artifact recovery is deliberately not permission
to reuse the compromised consumer. This preserves the reservation instead of silently blocking or
unsafely releasing it. The strengthened test first failed against the generic `reserved` message;
it now passes and also checks that an unaffected consumer still publishes its real NPZ.

Two further tests cover checkpoint CAS. `snapshot()` returns the cursor set and monotone revision
under one lock; `reset(..., expected_revision=...)` compares atomically and returns its new revision
under that same lock. Production checkpoint callbacks retain this returned revision, so rollback
cannot accidentally authenticate an intervening publication by reading a later revision. Tests
confirm that both installation and rollback refuse to erase newly sealed disjoint publications,
preserving actual NPZ bytes and cursors. The complete independent suite passed 12/12 in 0.79 s;
the final unaffected-consumer assertion passed in a targeted replay.

Source audit found one production `ConsumerTransaction` constructor, in `_stage_consumers`, with
mandatory shared authority. Scheduled scientific output, diagnostic, monitor and checkpoint
publications remain under its `PreparedPublication` dispatch; internal `.publish()` calls are
delegations within that transaction. Ordinary step rollback now retains the current authority and
reports; only explicit checkpoint install/rollback uses global reset, guarded by revision CAS.

`RuntimeInstance.checkpoint(path)` is a separate manual snapshot/write/finalize route and does not
advance ConsumerGraph cursors. It has no shared cursor reservation gate in this reviewed change;
concurrent manual checkpoint consistency is not qualified here. Raw System checkpoint APIs also
sit below the ConsumerGraph layer. No concurrent PDE, MPI, or native restart qualification follows
from this review.
