# Independent selected-history source reception

Pinned author production: `b91cbf5291f7e61238e542d6813c712c3ad271ce`,
parent `13a3cabe5c1d6830e3d6a9fe3357c945d243c0eb`; author tests/docs
`1c184a3ed760b29231c175746da08bce90c9f562`. This review changes tests/docs only.
It receives source, actual-header host and emitted C++ syntax, not a PoPS DSO,
MPI run, GPU run or a scientific native dataset.

The new HistoryComposite contract is conditional `pops.spatial-interaction@2`,
IR18, with `pops.spatial-interaction-history-source@1`. Direct issued/accepted
sources retain @1/IR17. The keeper closure must be its original, canonical
State.n expression with exact StateHandle, space, clock and point. Derived
T and Q=T+T² cannot be substituted simply because their storage owner matches.
The source probes issue a real Case/State/keep_history lag2, then replace the
keeper through the real store_history API or mutate the descriptor. They refuse
seven cases: derived T, Q closure, direct and wrapped point relabel, foreign
state, boolean seed ID and boolean lag. Both serialization and the live contract
validator refuse these cases. Positive Uniform and AMR emit the dedicated
history entrypoint and retain exact lag/seed metadata.

`sol61_spatial_history_independent.cpp` includes the real new header, real
HistoryManager, MultiFab, distribution and Kokkos. Ten assertions run for each
`POPS_REAL_TYPE=float` and `double`: cold selection does not initialize or publish
history; already cold-stored T=2 matches the seed while Q=6 refuses; a mature
selected lag remains valid after pending slot0 store; altered owner, selected
duration and unauthenticated sample refuse. This host executable has one
replicated lane and does not test transport or rank-local failure consensus.
The production call sites place these helpers inside existing interaction votes;
AMR checks pending remap and selected sample/duration across all levels before
snapshot and rechecks complete lifecycle before detached scratch publication.

Read-only provider tracing confirms `prepared_amr_block_state` delegates to
`block_state`; the accepted-carrier separation was already independently received
on 13a. The author actual AMR-engine coarse-before-fine host was replayed here.
No new claim of an installed facade lifecycle or conservative PDE is made.

Checks:

```
rtk proxy env -u PYTHONPATH -u POPS_NATIVE_DIM PYTHONDONTWRITEBYTECODE=1 \
 /Users/romaindespoulain/miniforge3/envs/pops-api040/bin/python -B -m pytest -q \
 -o pythonpath=python tests/review/test_sol61_spatial_history_independent.py \
 tests/review/test_sol61_spatial_interaction.py --tb=short
```

66 PASS in 45.40 s: 11 independent cases plus 55 author cases. These include four
real public Case→resolve→emit C++ syntax checks for Uniform/AMR with/without
history and actual-header host checks at both Real widths. The author host has
71 assertions per Real; our additional host has 10 per Real. No native/JIT
package execution or environment mutation occurred.

Four independently regenerated non-history profiles (issued/accepted ×
Uniform/AMR) compare exact serialized IR and emitted interaction C++ to a fresh
13a Git archive at the same callsite: all eight SHA256 values agree. The old AMR
entrypoint bytes remain SHA256
`ca23e5b348a2bb8bde50e25cd0f58764057660dd7bf4284d9abe069b7cae0e28`.
The source parity records remain in `outputs/history-review/{parent,current}.json`;
these outputs are not native evidence. Ruff and diff checks pass.

No blocker is demonstrated in this bounded reception. Installed Uniform/AMR,
MPI empty ranks, rank-local refusal, regrid, restart and actual history read-point
snapshots still require ROOT's rebuilt native campaign. The offline reader must
receive those snapshots under its own two external ROOT seals; this review does
not issue either seal.
