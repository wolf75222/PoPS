# Candidate7 validation: screened/MMS, primitive and provider-sensitive scripts

Frozen source: `bd583faf196f3c1faeedec489e04d24e959ed00b`, clean qualification8 checkout at completion.
Private interpreter invoked directly: `/Users/romaindespoulain/dev/tmp/PoPS-migration-20260907-candidate7-env/bin/python`.
Native SHA256: `d1eb7da32a5c0a7b2e7433dfdd309d03d4b19a8afaab5f426d03ad5ee875f518`, checked before the gate, in every script subprocess, and after completion.

Dim2, OMP_NUM_THREADS=2, OMP_PROC_BIND=false, native tests required, signed private package include/native trees, base environment Kokkos headers. PYTHONPATH was unset. Independent caches and logs were written outside the frozen checkout. Candidate6 was prepared but never launched in this lane; all results below are actual candidate7 executions.

Full `tests/python/unit/solvers/test_poisson_screened.py` and `tests/python/unit/physics/test_primitive_state.py`: **16 passed, zero failures/errors/skips in413.27s**. No marker exclusions, reduced cases, or maxfail limit. Exact individual test names/timings and the empty skip list are recorded in `candidate7-mms-primitive.results.json`; command provenance and fixture hashes are in the matching metadata JSON. Full output and JUnit records are in `.log` and `.xml`.

All five full scripts passed without skip markers:

- `time/test_time_multielliptic.py`:133.468s.
- `time/test_time_solve_fields_from_state.py`:74.830s.
- `runtime/test_predictor_corrector.py`:124.770s.
- `runtime/test_projection_eig_predicate.py`:2.341s.
- `runtime/test_projection_eig.py`:54.531s.

Every script used its unchanged frozen fixture bytes, recorded SHA256, and complete existing numerical assertions. The replay covers named/default and scaled field parity, fields derived from state at distinct stages, complete predictor/corrector comparisons, compiled spectral predicates, and native projection comparisons. Per-script logs and exact outcomes are retained in `candidate7-provider-fixtures/`, including `results.jsonl` and `summary.json`. The external full-script runner is `run_candidate7_provider_fixtures.py`.

No source, header or package edits/installations were performed. Both test processes completed; this lane no longer uses private7. The clean checkout SHA and native hash were verified again at the end. The separate Roe/compiler lane, LocalLinear/LocalNewton lane and root's full profiles were not duplicated. These results qualify the exact requested scope, not an unrelated complete suite or additional dimensions/backends/ranks.
