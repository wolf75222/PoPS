# Suspended cadence port reception repair

Base: `565efe0`. Production code is unchanged. Independent review: GPT-6 Sol,
`sol6_reference`, confirmed the rejection boundary and reviewed the exact diff.

The original `SuspendedRegionsKeepOneCadenceAndQualifiedScratchPort` fixture
called `advance_cadence_region` while its qualified port was still outstanding,
expected rejection, then tried to release that same port. Since collective
cadence rejection (`e3f2cf5`), this rejection restores the accepted cursor and
cancels the continuation, including its borrowed fields and callback. The outer
`System::advance_program_region` also cancels the attempt and rolls back field
publication. Retaining the port after that rejection would violate the current
transaction boundary.

The successful suspension test now tests wrong identity/target reads while the
port is alive, then releases the valid port and finishes both substeps. The new
`PrematureResumeRevokesSuspendedPortAndRetryStartsFresh` test injects the rejected
resume at the second substep: time has advanced from 2 to 2.25, but rejection
restores time 2 and macro-step 7. It verifies that the suspended callback was not
executed, the port and generation are inaccessible, all cadence leases are
cleared, and a retry uses another scratch field and fresh stage generations
before committing time 2.5 and macro-step 8. No qualification check was weakened.

## Actual bounded execution

Original root-built executable, same failing filter: exit 1, same exception
`Program map read/write does not match its suspended qualified port`.

Only the changed test translation unit was compiled and linked in this isolated
checkout, with the root Ninja command, existing gtest libraries and the existing
Kokkos/MPI dependency libraries. No library rebuild, install, or shared build
directory mutation occurred. The isolated executable passed all 7 tests in the
translation unit. `git diff --check` passed. Compiler output contained only the
existing gtest character-conversion warning and duplicate-rpath linker warning.

Reproduction command for the root's rebuilt target:

```sh
rtk proxy build-mpi/bin/test_program_context_schur_free
```

Local evidence under `outputs/cadence-port-review/`: `red.log`, `commands.txt`,
`compile.log`, `link.log`, `green.log`, and `green.xml`. The exact compile/link
commands are recorded there; generated objects and executable are not committed.
This is a host execution of the cadence/header mechanism with an empty
communicator. It is not a new MPI, AMR, PDE, or installed-package qualification.
Root owns the integrated rebuild and complete CTest reception.
