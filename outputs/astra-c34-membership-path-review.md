# Independent C34 membership and AMR Path test review

Reviewed MAIN `193775875cb3ee781756a1df0d6fb29db774d004` plus the pending
`_consumer_transaction.py` membership delta and `test_amr_path_rhs_barrier.cpp`
adaptation. Production files were read only. Reviewer: GPT-6 Astra/high.

## C34 result

No additional defect demonstrated in the bounded delta. `commit` records row
membership under the authority lock; `release(rollback=True)` starts from current
cursors and restores only committed effects owned by this root. Reservations
remain held until compensation/seal, so the affected IDs cannot legitimately be
advanced by another root while restoration is pending. Disjoint publications are
neither removed nor restored from the earlier global snapshot. A failed cursor
check does not publish the partially rebuilt local row map.

New independent test `test_consumer_membership_independent.py` uses actual NPZ
publications and a two-effect root mixing absent and explicitly empty rows. A
disjoint root publishes recursively between outer reservation and commit. Two
variants either seal that inner root or compensate it; a subsequent root then
advances/recreates that disjoint cursor before outer compensation. Exact cursor
serialization and artifact presence are checked, followed by an actual retry
of the outer root to verify reservation release. Both variants pass.

Command (run from `work/PoPS-principal-group`):

```sh
rtk proxy env -u PYTHONPATH /Users/romaindespoulain/miniforge3/envs/pops-api040-c11/bin/python -m pytest -q -o pythonpath=/Users/romaindespoulain/Documents/Codex/2026-09-28/dans-le-d-p-t-pops/work/PoPS/python tests/python/unit/runtime/test_consumer_membership_independent.py
```

Result: **2 passed in 0.75 s**. The test file is isolated; the explicit source
path selects the reviewed MAIN Python implementation. This is a source protocol
test with real serial NPZ artifacts, not installed-native, MPI, simultaneous
thread scheduling, or concurrent PDE qualification. The independent test commit
requires the C34 authority implementation and the reviewed membership fix.

## AMR Path test result

The old rejection of every FixedDt invocation contradicts the current
consumer-stability v2 unit budget: `numerical_face_courant_()` returns the active
authored Courant or 1, and `path_rhs_courant()` delegates to it. Outside an active
invocation, returning 1 is therefore consistent. The numerical path guard still
reduces actual face speeds and rejects nonfinite frequency or `dt*frequency`
above this budget; the test uses its non-deferred guard and a forward-Euler
continuation with coefficient `level_dt`.

HotTrace increases Gaussian variance to `1e8`, preserving the finite positive
rho/Theta fixture domain while raising wave speeds. The revised test now requires
the exact CFL refusal diagnostic (an arbitrary exception cannot satisfy it),
checks no publication/continuation, time zero, macro-step zero, accepted-state
identity and every level's cell values, and retries the same FixedDt with the
ordinary stage. The success helper checks the independent residual/state oracle.
This strengthens the distinction between invalid numerical budget and formerly
unsupported invocation metadata; no hidden failure was demonstrated by reading.

No native C++ build or execution was performed in this review. Parent's central
build and serial/MPI reception remain necessary.
