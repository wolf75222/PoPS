# Exact provenance of the public ALE MPI refusal

Read-only production review pinned to `29daef24292f17c3f11b0b6c8c9357523e951600`.
Correction is confined to test fixtures/support, independent probes and this
report. No production header/source, MAIN, SDK or environment was modified.

## Diagnosis

The N16/N32 MPI2 fixture asserted the wrong native category and omitted the two
native collective wrappers. Six other cases passed in each authentic rank log;
these two cases stop at the diagnostic assertion before their after-rejection
snapshot, safe retry and output-inventory assertions. No production defect is
demonstrated by these two failures, and their rollback/retry scientific reception
remains pending the central rerun.

Actual production chain, inspected in small ranges:

1. `program_context_moving_interval.inc:309` calls
   `all_reduce_max(shared_mismatch ? 1L : 0L, lane)` and then throws
   `std::invalid_argument` on every rank when the global result is nonzero.
   The call and throw lie outside the local-patch loop. Empty ranks therefore
   receive the same global refusal; physical patch ownership is not the rank
   selected for the subsequent exception diagnostic.
2. `program_cadence_continuation.inc:115` uses
   `collective_step_rejection_phase` with the exact failure label
   `Program cadence phase failed collectively`.
3. `collective_step_rejection.hpp:189` routes ordinary exceptions to
   `collectively_rethrow_exception`. `collective_exception.hpp:28` chooses the
   lowest failing communicator rank, broadcasts its exact diagnostic and throws
   `std::runtime_error` on every MPI rank. Since the globally voted ALE guard
   throws everywhere, this rank is zero, including when rank zero owns no patch.
   Serial execution rethrows the original invalid_argument, mapped to ValueError.
4. `system_impl.hpp:865` wraps the cadence failure once more with exact label
   `System step failed collectively`; the same native rank-zero policy applies.
   Its catch performs native transaction revocation/restore before rethrowing.
5. `_step_strategy.py:194` consumes every receiving rank's RuntimeError and
   produces the deterministic Python diagnostic with each outer receiving rank.

For MPI2, each receiver consequently reports exactly:

```text
collective step attempt failed during solve: rank 0 RuntimeError: System step failed collectively; rank 0: Program cadence phase failed collectively; rank 0: moving shared face or fixed-domain boundary is inconsistent; rank 1 RuntimeError: System step failed collectively; rank 0: Program cadence phase failed collectively; rank 0: moving shared face or fixed-domain boundary is inconsistent
```

These rank-zero native authorities and the outer Python ranks are distinct
pieces of provenance. Expecting `rank 1` in either native wrapper, expecting a
native ValueError in MPI, or removing a wrapper loses that provenance.

## Strict correction and independent probes

`require_exact_moving_interval_refusal` compares the complete tuple on every
rank: exact type name, full message and actual RuntimeError classification.
It checks rank count and envelope types, preserves the serial ValueError case,
and performs no substring, regex or numerical-tolerance admission. The native
fixture retains all original equations, guards, snapshots, scientific tolerances,
output inventory, refusal timestep and safe retry timestep unchanged.

22 independent source tests PASS. They execute unchanged AST-extracted bodies
of `_phase`, `_attempt_error_record` and `_collective_attempt_error` from the
pinned source, avoiding a native bootstrap import. Only world/allgather transport
is substituted. The nominal typed rejection class is never instantiated: these
probes receive ordinary fatal errors only. All receiver ranks are exercised for
serial, MPI2 and MPI3 envelopes, including zero local patches on either rank.
Wrong category, native rank, wrapper/phase, extra message, runtime classification,
missing/successful rank, and coerced rank counts are strictly refused.

The two authentic failure records in each actual rank log are parsed with
`ast.literal_eval` and received by the corrected exact assertion: four complete
N16/N32 records PASS. Original log paths, digests and complete records are
preserved in `sol61_moving_mpi_diagnostic_receipt.json`; no native result is
replaced. Ruff and Python parsing/diff checks PASS.

Command, using the source checkout explicitly:

```sh
env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 /Users/romaindespoulain/miniforge3/envs/pops-api040/bin/python -c 'import sys;sys.path.insert(0,"python");import pops,pytest;print(pops.__file__);raise SystemExit(pytest.main(["tests/review/test_sol61_moving_mpi_diagnostic.py","-q","-p","no:cacheprovider"]))'
```

No native/JIT/build or MPI execution was performed by this review. Empty-rank
native behavior is inferred from the inspected unconditional voted throw and
received Python envelopes; a genuine empty-rank native campaign remains unrun.
The corrected N16/N32 checkpoint/rollback/retry and output checks must still be
executed centrally against the rebuilt/authenticated runtime.
