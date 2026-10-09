# Program norm2 physical ownership @1

The public `Program.norm2(value)` remains component-zero Euclidean algebra over
its existing owned active-cell scope: the same local products, active masks,
local accumulation order and final square root. Its active mask is the existing
pointwise/EB mask; covered coarse AMR samples are not excluded. The sum therefore
retains samples from every selected level and is not a composite-cover or
volume-weighted norm. Correcting MPI replica participation does not change that
per-level mathematical scope. It introduces no volume/kappa
weight, component-wide pairing, normalization by the number of ranks, threshold
change, model selector or altered physical RHS. `dot_all` remains a separate
explicit vector contraction contract.

AMR's level-zero storage can be replicated, while fine levels are partitioned.
A replicated level contributes its finite local square once through rank zero
of the authenticated prepared execution lane; every partitioned local tile
contributes normally. Ownership is selected separately on each actual level.
Every rank still evaluates its local data. Nonfinite local contributions are
retained, including a non-root replica, so the previous norm/guard rejection
path remains observable. Empty rank-local shards contribute zero.

This internal ownership contract corrects double participation in AMR norm2.
It changes neither public syntax nor native data layout/ABI number; its header
content changes the SDK signature and requires rebuilt matching consumers.
The generic all-component contraction already uses the same physical replica
ownership policy; norm2 does not call or acquire its different arithmetic.

Durable tests append to `test_mpi_amr_spatial_norm.cpp`: distributed/replicated
coarse storage on one/two ranks, component-zero contrast with an unread second
component, last-rank nonfinite propagation, independent reversed block owners,
an empty partitioned rank and mixed replicated-coarse/partitioned-fine levels.
The original MPI2 retry and populated-history guard failures are retained.
A bounded JIT-only member-specialization experiment against the original matching
988/f8d/de2 installation changes only this generic context primitive: both public
cases pass with their unchanged `.1` and `30` guards. It is distinct from the
future official rebuilt SDK/runtime reception and does not qualify GPU or 3D.
