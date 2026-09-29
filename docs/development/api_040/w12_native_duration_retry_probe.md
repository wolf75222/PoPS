# W12 native duration retry probe

`ProgramRuntime.NestedChildCommitThenParentRejectRestoresDurationAndExchangeMailbox`
now exercises three distinct native step durations, 0.1, 0.2, and 0.3, with a
previously accumulated skipped interval of 0.25. The Program body reads the
actual `ProgramContext::cache_effective_dt` result and uses it both in the
state update and as the numerical flux of a real `ExchangeRecord`. It stages
the record through `ProgramContext::stage_exchange`, which qualifies its
identity with the authenticated runtime point, including the binary64 step
duration. The test checks the integrated amount `effective_dt * dt`, the
history slot, cached state, accepted state, clock, and the exchange mailbox.

The first child rejects after staging. The second child accepts with another
duration, then its parent rejects. A new root attempt accepts with a third
duration. The first and second exchange contexts must not survive rejection;
the third context must encode its own duration. A repeated occurrence in the
third attempt must be refused without replacing the first accepted record.
The oracle fixes these values before native execution and reads the actual
System cache, history, state, and ledger rather than an author-side dictionary.

This is a transaction and duration-identity probe, not a qualification of the
whole W12 application. The current `AccumulateDt` code generator calculates
`_effdt<node>` before the due body but does not consume that variable. A
scheduled rate can legitimately be multiplied by the current step duration
at a later consumer, so substituting the accumulated duration there without
an explicit temporal-use contract could double-count skipped steps. This
separate semantic gap needs an observable authored Program counterexample
before changing the lowering. The AMR schedule cache remains explicitly
unsupported; this test proves the Uniform native path only.

Run centrally after rebuilding the changed C++ test target:

```text
ctest --test-dir build-mpi -R '^test_program_runtime_np2$' --output-on-failure
```

The native test has not been executed in this isolated checkout.
