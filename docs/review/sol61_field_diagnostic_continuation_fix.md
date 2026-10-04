The real SDK29 CPU Vlasov–Poisson job 732949 failed while compiling its generated
Program, before bind or saved-case publication. Its preserved CPP
`5308c36c76416403a974079fcd0a5cb0e99760825057f427b5a1606f5a90526f.failed.cpp`
has SHA256 `819333a97d6ae9a5d4a2c483e9110927c8559fe2cd29636c80de11bee96c7967`.
Line 492 increments a scalar captured by value into a const deferred map lambda.
The later missing saved-case file is a consequence of that compilation failure.

Field solve/reuse diagnostics are mutable Host effects of one Program invocation.
Their emitted storage is now an owning `shared_ptr<array<Real, 2>>`, initialized to
zero inside each step invocation. Ordinary value captures carry the same payload
through nested map continuations after the creating frame returns. All owners are
prepared at invocation entry before the first solve callback or map suspension;
allocation errors join the existing execution-lane collective exception vote.
The counters are updated after the accepted solve guard, and their values reach
the existing runtime diagnostic records. They never enter per-cell device kernels.
New invocations and retries allocate fresh owners; accepted-state rollback uses the
existing `ProgramRuntimeState` diagnostic restoration and continuation cancellation.

This changes no physical equation, solver, tolerance, field reuse key, map, cadence,
or ABI. The emitter recognizes the typed field diagnostic effect, with no model or
variable-name dispatch in continuation capture. A `mutable` value capture would
update a separate scalar; a reference capture would borrow an expired frame.

The dedicated review tests use the public mapped two-stage field composition and a
distinct public two-state variable-coefficient field composition. A small Host
probe compiles effects extracted from actual emission against the actual cadence,
diagnostic and accepted-restore headers. It verifies nested suspension ownership,
visible solve/reuse counts, completed Kokkos Host work, rejected diagnostics,
restoration, fresh retry authority and an injected allocation refusal before any
callback. The exact historical failed CPP remains a negative compilation witness.
These Source/Host checks do not qualify installed Native, MPI, CUDA/HIP or VP physics;
the correction needs a new authenticated source/build and installed execution.
