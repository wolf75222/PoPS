# SDK14 initial Field/Ghost bounded reception preparation

Preparation only, private Source base e659a29a8292bd7d1e71ceb8231becb9cde7e41d. Existing source gels: FAC50655394 (independent e390e3a1), initial failure497212a8/fd6032c3/5e474fd7/6698efa2 (independent7e1a500c), doc correctionc8c91f2a. Production companion corrections and final SDK14 SHA/DSO/header signature are ROOT-owned and not yet assigned by this plan. Run only after that final combined Source is installed and identity checks pass; do not reuse SDK13 proofs or failed getter observations as endpoint-consumption proof.

Minimal existing preparation inventory (Source tests, not Native claims): tests/review/test_sol61_initial_fac_accuracy.py; tests/review/test_sol61_accepted_initial_ghost_contract.py; tests/review/test_sol61_accepted_initial_field_point.py; tests/review/test_sol61_initial_field_point_independent.py; tests/review/test_sol61_initial_ghost_failure_preparation.py; tests/review/test_sol61_initial_ghost_selection.py. Relevant checkpoint/budget unit boundaries: tests/python/unit/runtime/test_bound_initial_checkpoint_envelope.py, test_initial_checkpoint_strategy.py, test_amr_checkpoint_contract.py, test_multi_layout_checkpoint_diagnostic_capacity.py, test_field_provider_install.py. Their fake/Source fixtures are explicitly not receipt of a Native initial CP or Field solve.

Genuine installed Native nodes, both Serial and MPI2, unchanged equations and OriginalF1e−10:
- tests/python/integration/amr/test_public_initial_field_ghost.py::test_public_initial_field_fresh_before_ghost_and_positive_point
- tests/python/integration/amr/test_public_initial_field_ghost_failure.py::test_public_initial_ghost_rank_fault_keeps_prepublication_owner

Positive board includes actual initial, positive endpoint, checkpoint12/accepted9 and exact restart. Failure board proves same native owner at initial commit boundary only: global fullgrown state/registry, Field manifest and clocks before/after actual callback error, voted/agreed greatest real xmin owner and strict target parser. Neither current failure snapshots nor counters certify every Field cache/history byte or the outer abort after that captured boundary. Both Native positive and failure images must be saved before numerical/rollback assertions. Endpoint Ghost consumption point must be supported by actual callback/producer evidence; a Field accessor's last SolveOutcome may legitimately be at ForwardEuler t_n.

ROOT command template (no install/setup/build, use actual SDK14 values):

```bash
cd "$FINAL_SOURCE_CHECKOUT"
test "$(git rev-parse HEAD)" = "$SDK14_SOURCE_SHA"
test -z "$(git status --porcelain --untracked-files=no)"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 POPS_NATIVE_DIM=2 POPS_REQUIRE_NATIVE_TESTS=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CMAKE_BUILD_PARALLEL_LEVEL=1 POPS_HEAVY_MODULE_TU_POOL=1
# ROOT configures real compiler/Kokkos/MPI/include origins for this SDK, then authenticates them.
env -u PYTHONPATH "$SDK14_ENV/bin/python" docs/development/api_040/run_installed_checks.py --identity-only --output "$OUTPUT/before"
env -u PYTHONPATH "$SDK14_ENV/bin/python" -m pytest -o pythonpath= -p no:cacheprovider -q "$POSITIVE_NODE" "$FAILURE_NODE" --basetemp="$OUTPUT/serial-tmp" --junitxml="$OUTPUT/serial.xml" > "$OUTPUT/serial.log" 2>&1
# Separate execution, after positive Serial receipt: same-node two genuine ranks.
export OUTPUT POSITIVE_NODE FAILURE_NODE SDK14_ENV
# MPIEXEC is the actual launcher matching loaded libMPI; never a fake wrapper.
env -u PYTHONPATH "$MPIEXEC" -n 2 "$SDK14_ENV/bin/python" -c 'import os,pytest; from pops._native_selector import select_native_dimension; world=select_native_dimension(2).mpi_world(); assert world.size==2; rank=world.rank; out=os.environ["OUTPUT"]; raise SystemExit(pytest.main(["-o","pythonpath=","-p","no:cacheprovider","-q",os.environ["POSITIVE_NODE"],os.environ["FAILURE_NODE"],"--basetemp="+out+"/mpi-tmp-rank"+str(rank),"--junitxml="+out+"/mpi-rank"+str(rank)+".xml"]))' > "$OUTPUT/mpi.log" 2>&1
env -u PYTHONPATH "$SDK14_ENV/bin/python" docs/development/api_040/run_installed_checks.py --identity-only --output "$OUTPUT/after"
cmp "$OUTPUT/before/source-files.json" "$OUTPUT/after/source-files.json"
```

This is a command template, not a executed runner: ROOT must capture terminal code and postcheck even when pytest fails (existing ROOT trap/receipt protocol). Set the exact node variables above, new OUTPUT path, and per-run TMPDIR/POPS_CACHE_DIR/POPS_CODEGEN_DIR/XDG_CACHE_HOME before running. Scope cannot be received if postcheck/export or either rank XML is absent. Authentication includes actual pops/native paths inside SDK14_ENV, Source/installed header/Python equality, DSO hash, SDK signature, ABI, doctor, MPI2 real rank/size and loaded dependency origins before/after. MPI worker obtains rank/size from the actual selected Native communicator in the same interpreter; it assumes no launcher-specific environment rank variable and contains no cell loop. No commands were executed in Native/ENV by the author.
