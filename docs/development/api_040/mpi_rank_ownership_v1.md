# AMR level-0 rank ownership diagnostic v1

`runtime.amr.coarse_local_box_bounds()` returns this MPI rank's current valid
level-0 boxes as `((lower, upper_exclusive), ...)` in the grid's native axis
order. It reads the built hierarchy's level-0 `MultiFab` local storage directly;
it does not reconstruct or predict the distribution from `PatchLayout`. The
number of rows equals `runtime.amr.patch_table().coarse_local_boxes`.

The result is a read-only snapshot for the **current hierarchy epoch**. Query
again after regrid, restart, hierarchy transfer, or a step that may change the
layout. It is neither a checkpoint ownership map nor a promise that a later
step retains the same rank assignment. It supports every compiled dimension
through the exact-ranked native `Box<Dim>` projection.

The MPI witness in
`tests/python/integration/runtime/test_rank_one_owned_amr_invalid.py` uses
`PatchLayout(distribute_coarse=True, coarse_max_grid=4)`, allgathers these
native-owned rows, checks exact disjoint coverage of the level-0 domain, and
injects an active invalid User numerical body at an interior cell of a rank-1
box. Both ranks must reject the step, preserve the accepted time/step, and
retain the original global states. This qualifies **rank-1-owned** failure
only after the installed native test succeeds under two real MPI processes.
