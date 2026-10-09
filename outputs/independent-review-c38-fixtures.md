# Independent source review: three C38 native fixtures

Reviewed the three uncommitted **test/support-only** changes in MAIN after the
failures in `outputs/native-c17-c38-3d06cab-ctest.log`. No production guard was
changed and no native build was run by this reviewer.

The two `test_amr_history_ring` restart tests failed with `installed AMR Program
has no prepared flux-expression budget`. They constructed `Fixture` with a
provider only, then asked restart to seal Program checkpoint capacity. The new
`Fixture(true)` installs the existing Forward Euler hierarchy body and its
prepared flux budget before registering histories. Its explicit `clock.macro`
matches the tests' registered history clock. The helper's default clock remains
`test.clock.macro`, preserving its other callers. Non-restart fixture cases keep
their previous provider-only setup.

The three-level synthetic loader test failed at its second refined level with
`AmrSystem bootstrap level requires an active transaction`. The helper committed
inside the level loop, ending the plan after level one. Moving its single
`commit_bootstrap_level()` after all materializations matches the public Python
executor's one `finalize_bootstrap()` after the plan, and keeps the transaction
live for level two. The one-level case now also commits its plan explicitly.

These are source-supported fixture repairs for the exact logged failures. They
do not establish that the regrid/history assertions pass, that the same
artifact is used, or that MPI ranks converge. Root's exact rebuilt serial and
MPI2 receptions remain required.
