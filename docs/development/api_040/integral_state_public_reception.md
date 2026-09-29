# Public IntegralState reception prepared after the source review

This test-only increment builds on `88eba605`, independent tests `5a99502` and
review `1d88d39`. The embedded-NUL authoring repair is separate (`d424b77`).
No installed native test or JIT was run while preparing this increment.

## Fixed experiment and oracle

The new public fixture uses actual Dim2 Uniform, cells `(8,1)`, domain `[0,1]^2`,
periodic y and outflow x. It transports the true cell averages of `1+0.2x` with
physical velocity `(1,0)`, FirstOrder reconstruction and an explicitly authored
User Rusanov face with stability equal to its speed. This is the ordinary upwind
flux for this scalar system. SSPRK2 uses `dt=0.01`; `q0=0.7`, `C=1`, and the
selected external x+ contribution is delivered with scale `-1` because the
ledger's incidence points into the cell. The tolerance is fixed at `3e-13`.
The oracle uses NumPy incidence and two separate Euler evaluations, without
reading PoPS expressions, generated kernels or native states as its input.

The first and predictor boundary fluxes are different for this linear profile.
Their two half-dt deliveries are verified separately in the saved ledger. A
single physical x+ face gives exactly two selected records across all ranks;
in MPI2 at least one rank must have no selected record. This is a true Dim2
mechanism test, not a claim of spatial two-dimensional convergence.

## Cases and persisted evidence

`test_public_ssprk_integral_restart_in_fresh_instance` checks the initial scalar,
advances one step, checkpoints, and compares continued execution with restart
in a fresh bind of the same artifact. States and exchange bytes must be bitwise
identical at restart and after continuation; q must also match the independent
oracle. The checkpoint decoder separately checks POPSEX02 records, selected
support, temporal weights, consumed keys and replicated scalar values.

`test_public_integral_refusal_restores_envelope_and_safe_retry` has two variants.
The periodic-y selector has no external trace and must reject with its specific
diagnostic. The unsafe-dt variant proposes `0.5`, requires the authored User CFL
refusal, and retries on the same runtime with public endpoint clipping to
`0.01`. Both verify initial time, step, q, state bytes and exchange bytes after
rejection. The retry compares against the same SSPRK2 oracle. This rejection
occurs before delivery; it does not replace the C++ child-accept/parent-reject
receipt for rollback after provisional consumption. A foreign IntegralState
handle also must be refused without publication.

The existing `test_native_accepted_face_amount_and_initial_read` is unchanged
scientifically. Its AMR2 variant continues to require some active coarse cells
and an entirely fine-owned x+ boundary, then checks `q=0.7+1.2dt`. It now saves
initial/accepted public checkpoints and all explicitly gathered level states,
as well as its coarse active mask.

Every saved phase contains a public checkpoint, `*-state.npz` and
`*-receipt.json` under the pytest temporary directory. The receipt records the
artifact identity, platform, dimension, time, macrostep, exact quantity identity,
checkpoint SHA256, per-rank exchange-image SHA256 and decoded records/scalars.
All root-only file work and local assertions converge before subsequent native
collectives. The installed runner's `--basetemp` preserves these files in its
output directory. No private runtime mutation or injection hook is used.

## Commands and unreceived scope

Run the two runtime files through the repository's installed-check runner in
serial and MPI2, on the rebuilt Dim2 package:

```sh
rtk proxy env -u PYTHONPATH /path/to/pops-env/bin/python docs/development/api_040/run_installed_checks.py --output /path/to/receipt --test tests/python/integration/runtime/test_integral_state_trace_runtime.py --test tests/python/integration/runtime/test_integral_state_public_restart.py
```

Source preparation checks are `test_integral_state_public_receipts.py` and
collect-only of the two runtime files. The resulting native checks are pending;
this increment does not qualify all W11, M14, GPU or arbitrary AMR schedules.

The parent rebuild already exposed a separate release-contract defect after the
prior source/host review: `module_capabilities.hpp` still had `kAbiVersion=3`
after native release ABI became 4, causing a real static_assert failure. Root
owns that repair and release preflight. The small ledger/declaration host TUs
did not cover the complete package build, and this limitation is retained here.

## Initial installed AMR2 failure retained

While these tests were being prepared, the parent reported the initial author
fixture's installed AMR2 result `q=0.724` instead of the unchanged oracle
`0.7+1.2*0.01=0.712`; Uniform and AMR1 passed. The log is
`outputs/installed-integral-native-author-serial` in the workspace. The cause
was not yet established by this review. The strengthened fixture writes its
accepted checkpoint/ledger/state receipt before asserting q, so this failure
will retain the actual records rather than only its scalar assertion. The
expected q, dt, face ownership condition and tolerance remain unchanged.
