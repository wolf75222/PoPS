# Prepared geometry for accepted exterior exchanges

The installed Dim2 production artifact at source `f36176fa2a822c67b5ccc114bd21de1bef4be23e`,
SDK `506dce78009f1c0822b40ae1c38d9849eb9fbc2b65051906c4b2de4377039aaf`,
native SHA-256 `121bac0fd893427742e02a31437451a15f9ac7099274993c72d73c52fd7082f7`
passes ten serial public transport/diffusion/restart cases. The isolated AMR1
two-rank run times out at 360 seconds, with source and installation unchanged.

A second run uses a read-only PMPI interposer which forwards every call to the
original PMPI function. Both ranks finish the same first 3483 observed collective
entries. At entry 3483, rank zero enters
`AmrSystem::prepared_amr_level_geometry -> Impl::ensure_engine` on the package
communicator; rank one has already entered the exchange preparation vote on the
prepared hierarchy communicator. The owner selection exposed a collective
geometry getter inside the producer, whose record count is rank-local.

The two contexts now provide `prepare_external_trace_face_predicate()`. It
copies the prepared domain and boundary topology before the producer; its result
only reads those owned values. Transport also snapshots the geometry used for
face measures before the owner branch. PreparedDiffusion obtains the same
predicate in its existing mask preparation phase. Every rank still enters
preparation, error votes and staging, including ranks with zero records.

This is an additive inline C++ preparation seam. No context member layout,
native ABI number, equation, face orientation, quadrature, ledger wire format or
acceptance authority changes. The SDK header signature must be rebuilt. Neither
the serial receipt nor the diagnostic timeout qualifies this correction:
independent review, rebuilt serial/MPI and public installed reception are pending.

The immutable local runs are
`outputs/installed-integral-transport-diffusion-restart-dim2-sdk506-20260930`,
`outputs/installed-integral-transport-amr1-mpi2-dim2-sdk506-20260930` and
`outputs/installed-integral-amr1-mpi2-collective-diagnostic-sdk506-20260930`
under the task workspace. The interposer source and per-process call stacks are
in `outputs/mpi-amr1-collective-order-sdk506-20260930`. These are real production
PoPS artifacts and authentic failures, not prototype evidence.

