# AMR global field observation, contract v2

The exact `pops.amr-field-gather` contract advances from 1 to 2. All global AMR
field getters transport the prepared double buffer as object bytes on the
authenticated ExecutionLane. They do not sum physical data. The existing
collective preparation vote and exact agreement on semantic identity, level,
components, domain, patch layout and distribution still precede transport.

The counterexample is the installed SDK18 second-State fixture on two MPI ranks
(ROMEO job 732028, Source c6b7a600). Its complete native carriers retain a stored
negative zero in the spectator State, while `block_level_state_global` returns
positive zero. Both ranks reject the initial bit comparison after the actual
step and captures. The serial fixture passes. No mathematical tolerance or
signed-zero guard is changed.

Each valid patch is contributed once by its owner. Replicated distributions use
their existing canonical contributor. Non-contributors start with zero bytes,
so the common bytewise OR transport preserves every contributed double's bits,
including negative zero; uncovered cells of a partially refined level retain
positive zero. The double conversion from native Real remains the established
Python API. This change introduces no model-dependent branch and applies equally
to State, Field and history getters using the same preparation mechanism.

Byte count overflow is checked during collectively voted preparation, before
the MPI payload. Serial transport remains the identity. The native artifact must
be rebuilt and received before claiming the corrected MPI behavior. Source
inspection or previous SDK18 passes do not qualify the new artifact.
