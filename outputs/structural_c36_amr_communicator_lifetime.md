# AMR completion communicator lifetime regression

## Reproduced failure and source cause

Root reproduced the public installed integration test `test_nonnegative_diagonal_diffusion_runtime::test_periodic_amr_x_only_diffusion_keeps_composite_mass_and_refines` alone. The retained original log is `../../outputs/installed-degenerate-amr-isolated.log`: MPICH aborts in MPI_Allreduce with invalid communicator `0x84000003`. Installed native SHA256 was `220b48d3264f413aa5ccd8bba58a8b0d1a6b31bf5bade6faac904b1b7ef8cc87`, source MAIN `9e9a0e5bd56ebca6e0f678485f02f1d2b7ecfa78`.

The newly introduced outer completion vote borrowed `prepared_hierarchy->lane` through `require_prepared_engine_lane()`. `complete_program_step_()` can perform scheduled `regrid_parent()`, whose `refresh_prepared_hierarchy()` replaces and destroys that graph and its owned MPI communicator. `collective_step_rejection_phase()` votes *after* the operation returns. Its stored CommunicatorView was then a freed handle. Both `step()` and `advance_program_region()` had this lifetime violation. Numerical diffusion is not the cause of this source defect.

A read-only LLDB attempt never advanced past `run` into Python during more than four minutes; its dedicated debugger/debuggee processes were terminated. `../../outputs/structural_c36_amr_lldb.log` is retained. No stack trace or debugger reproduction is claimed.

## Correction and authority

Both entry points still authenticate the prepared hierarchy before dispatch. Outer cadence, continuation and completion votes now use `require_package_assembly_lane()`, the durable shared RuntimeInstance-owned ExecutionLane. Numerical operations continue to obtain their own prepared hierarchy lanes.

Ownership and participant proof in `src/runtime/amr/amr_system.cpp`: install_prepared_boundary_execution_context requires owned active authority and matching PreparedExecutionContext, rejects replacement or installation after engine materialization, and stores the shared pointer on Impl. ensure_engine calls multiblock_type::prepare_collectively(package_lane,...). prepare_hierarchy_graph duplicates candidate_multiblock.lane(). MPI_Comm_dup preserves group and rank ordering. AcceptedSnapshot restore rebuilds child graphs without replacing package_assembly_lane. Thus the parent remains valid through both successful regrid and rollback. There is no MPI_COMM_WORLD fallback in production.

## Regression and checks

Added `test_amr_synthetic_program_loader_transaction.ScheduledRegridCompletionUsesDurableRuntimeLaneAfterMoveAndRollback` and its permanent MPI2 CTest filter entry. It uses the existing authenticated source-built synthetic Program DSO and real native AMR carrier:

- scheduled regrid_every=1, real topology epoch, regrid count and patch coverage changes;
- step and advance_program_region paths;
- MPI embedding parent obtained by MPI_Comm_dup, then freed before stepping; the RuntimeInstance owns its duplicate;
- move construction and move assignment after package authority installation, before Program captures bind the facade; AmrSystem is noncopyable, checked statically;
- rank-zero rejection after successful actual regrid, public transaction rollback on all ranks, exact accepted bytes/coarse and fine state/topology restoration, then retry and committed time/step.

This synthetic fixture qualifies lifetime/transactions only, not public Python numerical-method semantics. A live move after Program capture installation is not tested: AmrProgramContext stores a facade pointer and this is a separate possible contract issue. The borrowed communicator API currently accepts congruent world rank spaces; this test makes no subgroup/reversed-rank support claim.

Actual local check: the complete changed C++ regression translation unit passed `/usr/bin/clang++ -fsyntax-only` using MAIN build-mpi compile command with isolated source/include paths. Only an external GoogleTest char8_t warning was emitted. See `outputs/structural_c36_amr_regression_syntax.log`. No native rebuild, link, serial execution or MPI2 execution was performed in this worker. The new native regression has not itself been executed red/green; the existing public Python failure is the retained red evidence.

Root acceptance: rebuild pops_runtime_amr and test_amr_synthetic_program_loader_transaction, run that test serial and MPI2 with timeout, plus existing ScheduledHistoryRegridRefreshesCapturedBodiesAndRollsBackPublication. Rebuild/install the matching package and rerun the original installed degenerate-diffusion AMR test. The correction does not assert that arbitrary rank-local exceptions inside unrelated regrid collective sequences are safe.
