# Independent source review: three-level C38 owner alignment

Reviewed MAIN commit `ab0007a` after the serial and MPI copies of
`RestartRegridRetainsDescendantHistoriesAcrossTwoParentReplacements` reached
`rebuild_hierarchy` and failed with `boxes and owner ranks must align exactly`.
The native public contract states that `patch_boxes()` contains **fine patches
only** (`level >= 1`), and `rebuild_hierarchy(boxes, owner_ranks)` requires one
owner per box in the same order; level zero is implicit. The prior fixture
concatenated owner ranks for levels `0..2` against the level-`1..2` box list.

`ab0007a` leaves the saved conservative states at levels `0..2` intact, but
collects owner ranks only for levels `1..2`. It adds a size assertion before
restart. No product code, source model, checkpoint image, or scientific
threshold changes. After `rebuild_hierarchy`, the test still restores all
three states and all six history rows, checks both parent/descendant layouts
change, requires descendant coverage not to shrink, compares retained numeric
history values on old descendant cells, and commits/finalizes the transaction.

The source correction matches the exact public geometry/ownership contract.
It does not prove the subsequent assertions pass. Root's rebuilt serial and
MPI2 runs with finite timeouts remain the necessary reception; this review
used no native compilation or execution.
