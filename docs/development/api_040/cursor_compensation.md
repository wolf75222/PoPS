# Exact cursor compensation

The installed 38faddb reception exposed a C34 regression: compensation of a first
publication restored a stored empty cursor, although the accepted checkpoint had
no row for that consumer. Both have the same lookup value, but their serialized
checkpoint representations differ. The existing compensation test failed.

`ConsumerCursorAuthority.commit` now retains row membership with each reserved
root. Compensation restores the affected rows and removes only rows that this
root introduced. It preserves unrelated rows and publications by disjoint roots,
including publications committed before or after the rejected root. Explicitly
stored empty rows remain stored. Cursor authority revisions stay monotone.

This repairs the existing rollback contract; it does not change the cursor schema
or establish general scientific read-set concurrency. The regression and four
membership/order combinations belong to the normal runtime test suite. Source
checks and the installed rerun are recorded separately. The original failed
receipt is retained at the parent workspace's `outputs/installed-38faddb-unit/`.
