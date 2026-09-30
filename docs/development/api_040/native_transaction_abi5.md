# Native transaction ABI 5

The moving-interval carrier adds an owning geometry map to `ProgramRuntimeState`.
Accepted snapshots and prepared restores now copy that map and publish it with
the state, histories, diagnostics and exchange ledger. This changes the C++
object layout, so the central native ABI advances from 4 to 5. The generated
release contract and `pops::kAbiVersion` advance together. All native extensions,
generated Programs and external native consumers must be rebuilt against the
same shipped SDK. Header signatures authenticate the exact build within ABI 5.

The public API and semantic IR remain at version 3. Existing transport-only
Programs retain serialized version 5; selecting an accepted constitutive face
trace uses serialized Program version 6. External computed-grid frontiers use
their explicit version 2 policy, and a Program-returned Scalar duration uses the
`ComputedDt` version 1 policy. Neither changes the earlier stage coordinates.

The first moving-interval C++ carrier has a separate, bounded reception. Its
native state ownership does not qualify the complete public ALE route, AMR,
geometry checkpoint codec, or devices. Numerical preparation failures must be
voted before any subsequent collective; prepared physical recovery must accept
the candidate before its ledger and state/geometry swaps. Subsequent owning
evaluation packets must authenticate their time, duration, geometry generation
and live attempt, rather than accepting stale integrated amounts.

Receipts carrying SDK `fc0bfd9a0ab1a6036a2e941dd16fdcbfc486125d31b8679d465d68111d16e959`
belong to ABI 4. They are retained as evidence for their recorded source and
backend and do not qualify ABI 5. Local Kokkos 5.2.0/MPICH 4.1.2 reception also
does not substitute for the declared Kokkos/OpenMPI release matrix or GitHub CI.
