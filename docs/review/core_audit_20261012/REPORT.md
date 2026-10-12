# Core audit fixes: local native qualification

Four reproduced defects are corrected: a variable tensor stencil can lose ellipticity, stationary AMR tags can repeatedly expand retained buffers, Richardson can reject convergence reached by its last allowed correction, and a checkpoint path error on one MPI rank can leave peers blocked. The changes and their regression tests are confined to the core audit branch; the separate solver research checkout was preserved.

This report records received local evidence from 12 October 2026. It does not assert a GitHub CI result. CI must be assessed separately on the final PR commit and its required aggregate gate.

The PR is stacked on `codex/adc-euler-comparison-20261011` (#685, head `29a8852d`). The audit starting point `424344f7` contains eight inherited prerequisite commits after that public head. Those prerequisites are distinct from the two audit commits below. The six later commits of the separate solver research branch are absent from the audit branch; no solver checkout or branch is modified by this PR preparation.

## Source and installed runtime

| Identity | Recorded value |
| --- | --- |
| Production fixes | `a17fe0e98955220bc0a1719a981d8a80a40c3743` |
| Qualified fixture follow-up | `319647003a1148d2d7d0808f0051ee2c41306753` |
| Branch | `codex/core-audit-20261012` |
| Installed native specialization | Dim2, C++/Kokkos/MPI/parallel HDF5, ABI 13 |
| Native SHA-256 | `dbf1f7de0d4309ae5607028432604493f2a4c4f7140a8d0f94446a1438909d08` |
| Header signature | `f395e0ac974a8c9ea34c30fb5bd6edcb337a3b0e11feaad8bc35e63d66045172` |
| Richardson test SHA-256 | `a0abd0a87b61ed631475c5f25523ce3fa466628cb7acab799c5049c3dc444f38` |

The public tests used the installed `pops` package in the dedicated `pops-core-audit-20261012` environment, with `PYTHONPATH` removed, `OMP_NUM_THREADS=2`, and `OMP_PROC_BIND=false`. Rank receipts authenticate the selected extension. The recorded ABI identifies Apple LLVM 21, C++20, libc++, MPI and Kokkos support. These are local runtime results; no substitute runtime or reference SDK supplies this qualification. See [native identity](evidence/native-identity.json), [native capabilities](evidence/native-contract.json), and [final source/fixture verification](evidence/finalization-verification.json).

The installed production runtime is attributed to `a17fe0e9`. The sole change from that revision to `3196470` is the constant RHS in `tests/python/integration/runtime/test_public_krylov_lifecycle.py`. The qualified fixture uses sixteen values equal to one, giving an exact dyadic L2 norm of four; this stabilizes the exact equality oracle after normalization. Solver code, assertions and tolerances (`rel_tol=0`, `abs_tol=1e-12`) are unchanged. The test-file hash already matches the six completed public Richardson qualifications. This fixture-only commit did not require a native rebuild.

## Corrected behavior and direct coverage

| Defect | Before and after | Received qualification |
| --- | --- | --- |
| ALG-001: tensor stencil | A periodic checkerboard of valid SPD tensors combined harmonic normal faces with incompatible arithmetic cross terms, producing negative discrete energy. Cross gradients now use the congruent tensor `D A D`, with `D_ii=sqrt(H_i/A_ii)`, retaining harmonic normal faces and positive periodic energy. The exact center diagonal includes cross terms and physical boundary ghost contributions. | [8 stencil tests](evidence/test-tensor-final-stencil.log) and the [FAC positive-energy regression](evidence/test-fac-regression.log) pass. Coverage includes discontinuous SPD energy, representable extreme contrast, weighted gradients, symmetry/exact diagonal, physical corner ghosts, second-order smooth consistency, complete homogeneous Neumann flux, and mapped conducting-disk modes. |
| AMR-01: buffer growth | Buffer/clustering coverage retained from a previous regrid was reused as a dilation seed, so fixed tags expanded the hierarchy at each regrid. Retained coverage remains available for Keep/hysteresis, while only effective accepted Refine requests seed the new buffer. | [10 public variants, 15 rank results](evidence/amr-public-qualified.json) pass with no failures, skips or timeouts: five cases on MPI1 and MPI2. Each checks four accepted steps, unchanged patch boxes, unchanged state and composite integral within `2e-14`. |
| ALG-002: terminal Richardson convergence | With `A=I`, RHS one, zero seed and `omega=1`, a budget of one correction reached the solution but returned an iteration-limit rejection. Richardson now measures the candidate's true physical residual after the final correction before deciding budget exhaustion; nonfinite values and acceptance controls remain guarded. | [4 C++ regressions on MPI1](evidence/test-richardson-regression.log) and [4 on each MPI2 rank](evidence/test-richardson-regression-mpi2.log) pass. [6 public variants, 9 rank results](evidence/richardson-public-summary.json) pass: identity budgets one and two, plus repeated nonconvergent attempts that preserve accepted state/time. |
| CP-001: collective checkpoint paths | Local normalization could raise before a rank entered the snapshot collective. Checkpoint/restart path preparation and canonical target agreement now converge across the communicator before snapshot capture or root I/O. | The [public MPI2 receipt](evidence/checkpoint-public-mpi2.receipt.json) records return code zero, both ranks passing, no timeout, and empty stderr. It covers NUL on either rank, divergent valid directories/filenames, root-only I/O failure, canonical aliases, rollback, and fresh-instance bitwise AB2 continuation. |

Relevant production source files are `include/pops/numerics/elliptic/nd/cartesian_tensor_operator.hpp`, `include/pops/numerics/elliptic/linear/generic_krylov.hpp`, `src/runtime/amr/amr_system.cpp`, `python/pops/output/_checkpoint_collective.py`, `python/pops/output/_restart_provider.py`, and `python/pops/runtime/_runtime_instance.py`.

The public AMR matrix uses `(buffer, coarsen, cadence, levels)` values `(0, false, 1, 2)`, `(1, false, 1, 2)`, `(1, true, 1, 2)`, `(1, false, 2, 2)`, and `(1, false, 1, 3)`. The test is `tests/python/integration/amr/test_public_tag_selection_native.py::test_stationary_regrids_do_not_redilate_retained_tag_buffer`. MPI2 results are received independently on both ranks; the matrix includes the fifteen rank results, and [ten consolidated watchdog receipts](evidence/amr-process-completion.json) record successful process-group completion.

The public Richardson tests are `test_public_richardson_accepts_identity_solution_at_the_iteration_cap` (budgets one and two) and `test_public_richardson_rejects_an_unconverged_last_correction_without_publication` in `tests/python/integration/runtime/test_public_krylov_lifecycle.py`. [Six consolidated launch receipts](evidence/richardson-process-and-rank-completion.json) confirm zero return codes and no timeout; their nine rank receipts confirm installed native identity and successful pytest exit codes.

The checkpoint test, `tests/python/integration/mpi/test_uniform_history_checkpoint_mpi.py`, follows `Case -> validate -> resolve -> compile -> bind -> run` on the real runtime. Failure checks preserve native state, dense AB2 history and accepted time; restart refusal additionally preserves archive bytes. Successful restart in a fresh instance continues bit for bit with the uninterrupted public AB2 runtime to six steps at `t=0.06`. The recorded duration is 83.832 seconds, including cold compilation, under a 180-second process-group watchdog. See the [checkpoint output](evidence/checkpoint-public-mpi2.stdout.log).

## Broader regression evidence and superseded attempts

The C++ evidence consists of several received selections, not one fresh global run:

- [Initial final selection](evidence/test-cpp-final.log): 56 passes, three MPI1 rank-required skips and two historical failures.
- The failed FAC test `#591` was rebuilt and [passed individually](evidence/test-fac-regression.log). The failed provider-action oracle `#757` was corrected, rebuilt and [passed in the four-test Richardson selection](evidence/test-richardson-regression.log). Both historical failures are superseded for the final source.
- [Complementary selection](evidence/test-cpp-additional.log): 28 passes and one further MPI1 rank-required skip.
- Combining unique received results gives **86 distinct MPI1 passes** and **four MPI1 rank-required skips**. The skipped cases are `#748`, `#764`, `#765` in the Krylov workspace target and `#597` in the coarse GMRES target. Both corresponding [MPI2 CTest targets pass](evidence/test-cpp-mpi2.log); the four dedicated Richardson regressions also have separate MPI2 results. These counts must not be described as a fresh global `90/90` run.

Supporting received checks are [76 Python checkpoint unit tests](evidence/test-checkpoint-unit.log), [55 CI-inventory tests](evidence/test-ci-inventories.log), [documentation lint](evidence/check-docs.log), and the [packaging manifest check](evidence/check-packaging.log). Documentation lint passed with four recorded dependency-staleness warnings; the bundled log retains them.

Initial grouped AMR/Richardson attempts timed out after 240 seconds and are excluded from successful coverage. The initial Richardson whole-file attempt showed `..FF`: two unchanged legacy cases passed, then the identity fixture's exact oracle failed before interruption. The constant-RHS fixture was subsequently received in the six complete individual MPI1/MPI2 cases. The summary preserves this historical failure/timeout description. None of the partial attempts counts as a completed qualification.

The verification recorded no missing requested qualification, no production changes after the received native campaign, and no rerun of previously received tests during finalization. This report does not qualify unrelated scientific campaigns, GPU configurations, or all distributed AMR behavior.

## Portable evidence

[evidence-manifest.json](evidence-manifest.json) records the selected small receipts/logs, their original SHA-256 values and the SHA-256 values of these public copies. Personal absolute paths are replaced with portable environment, repository or evidence references; test outcomes, runtime hashes and revision identifiers are preserved. The large CTest catalog, build directories, binaries, and full compilation logs are excluded. References to nonbundled private launch scripts or JUnit files are historical metadata, not additional claimed evidence.
