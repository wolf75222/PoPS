# Independent review: suspended Program cadence port

The native CTest failure at MAIN `565efe0`, filter
`ProgramRuntimeStateCadence.SuspendedRegionsKeepOneCadenceAndQualifiedScratchPort`, reported
`Program map read/write does not match its suspended qualified port`. This is a
fixture error, not evidence that the production port identity comparison is
wrong. The test called `advance_cadence_region` while a qualified port remained
outstanding. The entry preflight rejects that call; its catch rejects resources,
restores accepted time and macro-step, and cancels the continuation, including
borrowed fields. Subsequent reads/releases of that cancelled port must fail.

I independently reviewed author commit `fb0bbfc` in `PoPS-principal-group`.
It changes only the C++ fixture and its report. The success path now probes
foreign identity/target while the port remains live, then releases it and
completes both substeps. A separate rejection test suspends at the second
substep, proves that premature resume restores time `2.25 -> 2.0` and step
`7`, revokes the port and stage generation without running its callback, then
retries with a different scratch field and generations `3/4` to commit
time `2.5`, step `8`. This preserves the rollback contract; no production
code or guard changed.

An additional test, `ForeignPortReleaseCannotConsumeSuspendedStage`, lives in
this review branch. It checks the complementary operation: two wrong
`release_program_map` calls cannot clear the still-live qualified port, whose
field and generation remain available to the correct release and resume.
It was authored independently from `fb0bbfc`; it is not part of the author's
7/7 executed TU receipt. I compiled/linked this TU in the isolated worktree
against the existing root build libraries, with outputs only under
`outputs/cadence-port-independent/`, and ran this filter: **1/1 passed**.

The author's isolated translation-unit rebuild/link using the existing Ninja
flags reports 7/7 passing host tests, after reproducing the old binary's
failure. I inspected its source diff and the recorded receipt but did not
rebuild the native package or run MPI. These are host TU tests with an empty
communicator. Root's rebuilt exact-head CTest remains the integration
gate.
