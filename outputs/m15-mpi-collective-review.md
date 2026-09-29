# M15 owner-only face rejection, MPI2

Base source: `4bb0639`; installed genuine Dim=1 native SHA-256
`84b7d44400c40218d91c90d5cdfaa31a0860f6205eaa1a9fd5c4f709af3bac6c`.
The original campaign evidence is
`../../outputs/m15-4bb0639-dim1-mpi2/{result.json,interruption.json,rank0.sample.txt,rank1.sample.txt}`.
It completed the first N32 trajectory and then stalled on the first inadmissible-state
case. Rank 0 waited in `WorldCommunicator::allgather_bytes`; rank 1 waited in
`System<1>::step`'s `collective_step_rejection_phase`. The campaign was interrupted,
so it proves a hang, not a scientific M15 pass.

A separate MPI2 diagnostic used the same installed package and M15 N32
`negative_density` input. Both ranks compiled and bound successfully, then a
deliberate allgather of bind errors completed with `('', '')`. `pops.run` then
reproduced the same mismatched collective stacks and was terminated by a
90-second timeout. This excludes bind failure as the initiating divergence for
this case. No package was rebuilt for this diagnostic.

The FE body emits `ctx.rhs_into`, whose periodic route calls
`System::block_rhs_into_at` then `block_rhs_group`. Bound preparation built a
periodic transport session but published `prepared_boundary_group_core_` only
when a physical boundary existed. A periodic group therefore selected route 0,
and `SystemBlockStore::evaluate_rhs_core` invoked the generated, unprepared
`full`/`flux` callback. A face-domain error can arise on the owning rank after
the common halo operation, while a peer with no local face continues to later
collectives. The outer step rejection phase is too late to align those calls.

The proposed repair publishes the prepared group core for a complete periodic
pair of flux/full callbacks. The core then uses the existing
`periodic_{flux,full}_at_point_prepared` callbacks and their collective
face-materialization boundary. Blocks without these callbacks retain their
unprepared route inside a per-block collective error boundary. The new installed
MPI regression separates bind and step, requires the genuine native face
rejection on every rank, and checks no state or clock publication.

Current verification: source-only Python syntax and `git diff --check` pass.
The regression is intentionally unqualified until root rebuilds the native
Dim=1 package and executes it with MPI2. Existing serial M15 results do not
establish MPI rejection convergence.
