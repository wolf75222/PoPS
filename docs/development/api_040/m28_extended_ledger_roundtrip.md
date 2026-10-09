# Metadata-only nested POPSEX02 roundtrip

The real native reception of the moving checkpoint baseline exposed a writer /
reader disagreement: ALE explicitly selects `ledger.checkpoint(true)` to retain
trace support and `source_evaluation_identity` in nested POPSEX02, even when no
IntegralState is declared. The reader required a strictly positive integral
count and therefore rejected its own authentic empty or metadata-only image.

The parser now accepts the genuine zero count. The remaining wire extent still
bounds every nonzero integral count; truncated consumption lists, unknown
consumed traces, malformed records and trailing bytes continue to refuse.
No default POPSEX01/02 writer bytes change and no new synthetic integral is
declared to work around the disagreement.

`moving_extended_ledger_host.cpp` includes the actual native header. Its old
baseline mode reproduced the exact `invalid integral count` refusal, then the
fixed-header host binary passed empty and qualified-metadata exact roundtrips,
three independent corruptions and the existing nonempty IntegralState roundtrip.
This is a small host receipt seam check, not a System/PDE/MPI runtime claim.
An additional actual-header probe demonstrates that the prior parser accepted
an exact consumed exterior key with zero declared integrals. This image cannot
be emitted by the native API: consuming a trace requires a declared integral.
The parser now refuses that impossible combination explicitly while retaining
the valid unconsumed exterior metadata baseline. Four corruptions are checked.
Neither the default POPSEX01/02 writer nor its bytes changes.

The same assertions are registered in the real ProgramContext contract target;
the authentic moving codec and staging fixtures still require central native
reception after rebuilding the changed header.
