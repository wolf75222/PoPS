# SDK2e4 corrected diffusion and Uniform Stage reception

This follow-up preserves the first SDK2e4 attempts. [Machine evidence](sdk2e4_diffusion_stage_followup.json) pins actual execution result/XML, source origins, owner inventories, external ROOT approvals and independent scientific results. Native success does not close the full original corpus.

The Python Stage repair is installed through `scripts/build_python.sh`; its receipt is `native-python-stage-repair-1807/installation-reception.json` under the evidence directory below. Five Python production files changed, zero headers/C++, Ninja had no work, all three installed DSOs are exact. 1,120 Python/header files authenticate. Dim2 native SHA is `b8a5166c3ebe0a64e370be233bb2e987cbd7324f2b64dfac782718ead03f234b`, SDK/header signature `2e4d69fd3574933f4b83cf014ad2d0159902fea6197d74623bee4b4ee6cf1675`. The separate retained SDK manifest file SHA is `4942e284b6200142ad3bde5462c6a6c908f68267a278f9a31f7393c5248bf26d`. Original CPP build source is e01c12a7, common Python installation source1807a16b. Actual Stage fixture execution source is135179aa. These identities describe different things.

| Selection | Serial | MPI2 per rank | Scope |
|---|---:|---:|---|
| Corrected Frozen/Candidate archives | 4 PASS | 4 PASS | Uniform nonconstant MMS, scalar/signed nonsymmetric permuted3 |
| Corrected diagnostics | Uniform2 PASS; earlier AMR2 PASS | Uniform2+AMR2 PASS | Native bits, capacity faults, joined rollback, legacy absence |
| Evolved original Stage | 8 PASS | 8 PASS | N8/N16, dt .01/.02, 1/2 evolved Q carriers, 1/3 unknowns |

The Serial diffusion/Uniform-diagnostics batch has a seventh source/native-geometry admission. Each MPI diffusion/diagnostics XML has eight clean cases. Each Stage XML has eight clean cases. All MPI source/installation before/after pins and rank parity pass. MPI scientific archives are collective rank0 publications: two per diffusion family or eight Stage archives, never doubled by rank count.

Frozen@3 and Candidate@2 keep distinct native checkpoints and observations, actual retained Program CPP and compiler-carried IR. ROOT scientific reception is reproduced independently in Serial and MPI2: max original-F relative residual 2.908e-15 Frozen and 1.198e-11 Candidate, original guard3e-8. Uniform8 histories/clocks and POPSDIA1 per-rank diagnostics are received, with exact replay. [Frozen copied attacks](captured_diffusion_saved_reception_sol61.md) refuse15 MPI2 corruptions, [Candidate](candidate_diffusion_sdk2e4_real_countermodels_sol61.md) nine per backend; fake seals occur only on marked negative copies. Donors remain unchanged. These checks execute no PoPS or Native code.

Stage restart correctly carries a distinct continuation RunIdentity. Every checkpoint envelope is authenticated against its own real runtime creator; all other manifests/arrays/history/diagnostics/clock payloads compare exactly. [Independent comparator](sol61_stage_continuation_candidate_independent.md) receives8 real pairs and refuses48 resealed mutations. The real Stage fixture original equations and guards now pass, and separate independent Stage scientific ROOT owner seals are closed for both modes. [Stage reader](sol61_stage_saved_state_independent_reception.md) receives eight actual archives per mode, exact rational IR, 64 phases and 48 CP total; max original F relative 1.954e-11, guard3e-8. It refuses18 resealed contremodels per mode and three forged-rank XMLs. AMR Stage needs an explicit history storage owner port and remains unreceived.

The reader links documentary Program IR/hash, CPP export, receipt and checkpoint. Candidate sidecars and binary/artifact digests are recomposed. A component compound semantic digest is a different domain from documentary Program hash; its model/Program/ResolvedSnapshot payload is absent. No compound semantic recomposition, artifact aggregate recomposition, CPP-to-DSO graph proof or private capture-lease persistence is claimed. ROOT component association is an explicitly external authority, not a reconstructed cryptographic payload.

## Reproduce native executions

Use a fresh output for every command. setup_env.sh has already been attempted once; do not repeat setup on this checkout. Use the repository build scripts for installation as described in [the prior build receipt](sdk2e4_native_reception.md). Do not add the reference prototype to Python paths.

```sh
cd /Users/romaindespoulain/dev/tmp/pops-api040-native-reception-20261001
POPS_RECEPTION_ENV=/Users/romaindespoulain/miniforge3/envs/pops-api040
POPS_RECEPTION_EVIDENCE=/Users/romaindespoulain/dev/tmp/pops-api040-native-reception-evidence-20261001
export CONDA_PREFIX="$POPS_RECEPTION_ENV" Kokkos_ROOT="$POPS_RECEPTION_ENV" POPS_KOKKOS_ROOT="$POPS_RECEPTION_ENV" CMAKE_PREFIX_PATH="$POPS_RECEPTION_ENV"
export POPS_INCLUDE="$POPS_RECEPTION_ENV/lib/python3.12/site-packages/pops/include"
export PATH="$POPS_RECEPTION_ENV/bin:$PATH" POPS_NATIVE_DIM=2 PYTHONNOUSERSITE=1 POPS_REQUIRE_NATIVE_TESTS=1 POPS_KEEP_GENERATED=1
export OMP_NUM_THREADS=1 POPS_THREADS=1 OMP_PROC_BIND=false FI_PROVIDER=tcp

env -u PYTHONPATH "$POPS_RECEPTION_ENV/bin/python" docs/development/api_040/run_installed_checks.py \
  --output "$POPS_RECEPTION_EVIDENCE/reproduction-stage-serial-FRESH" \
  --test tests/python/integration/runtime/test_public_evolved_original_stage.py::test_public_evolved_original_stage_saved_and_exact_replay

env -u PYTHONPATH "$POPS_RECEPTION_ENV/bin/python" docs/development/api_040/run_installed_mpi_checks.py \
  --output "$POPS_RECEPTION_EVIDENCE/reproduction-stage-mpi2-FRESH" --dimension 2 --ranks 2 --threads 1 --timeout 1800 \
  --test tests/python/integration/runtime/test_public_evolved_original_stage.py::test_public_evolved_original_stage_saved_and_exact_replay
```

For the diffusion/diagnostics run use the same runners with these four repeated `--test` values:

```text
tests/python/integration/runtime/test_public_captured_diffusion.py::test_public_captured_diffusion_nonconstant_saved_and_exact_replay
tests/python/integration/runtime/test_public_captured_diffusion.py::test_public_candidate_diffusion_nonconstant_saved_and_exact_replay
tests/python/integration/runtime/test_program_diagnostic_checkpoint_runtime.py::test_native_diagnostic_bits_checkpoint_restart_replay
tests/python/integration/runtime/test_program_diagnostic_checkpoint_runtime.py::test_native_diagnostic_capacity_faults_and_joined_restore
```

Historical failed attempts remain under `installed-sdk2e4-evolved-stage-serial-dim2` (actual compiler defects), `installed-sdk2e4-evolved-stage-corrected-serial-dim2` (fixture envelope criterion) and `installed-sdk2e4-diffusion-and-diagnostics-serial-dim2-corrected-selection` (Uniform fixture accessor). Actual sources are archived as immutable execution Git blobs under `diffusion-source-archive-457e`, `diffusion-source-archive-1807` and `stage-source-archive-1351`; IR/CPP were saved by the original execution, not rebuilt later.

## Reopen received archives

Independent readers require separate external pin and ROOT-approval hashes; missing/mismatched seals fail. The proof paths and four independent owner/approval pairs are pinned in the machine receipt. For example:

```sh
env -u PYTHONPATH "$POPS_RECEPTION_ENV/bin/python" -I /Users/romaindespoulain/Documents/Codex/2026-09-28/dans-le-d-p-t-pops/work/PoPS/tests/review/sol61_candidate_diffusion_checkpoint_reception.py receive \
  --pins "$POPS_RECEPTION_EVIDENCE/diffusion-candidate-mpi2-scientific-root-owner-v1/owner-pins.json" \
  --pins-sha256 aab5be7c9d554a6701f67a2b2ad2315126ac08aee99a99d564eb615530aeeed7 \
  --approval "$POPS_RECEPTION_EVIDENCE/diffusion-candidate-mpi2-scientific-root-owner-v1/root-approval.json" \
  --approval-sha256 7eac5cb845bcdc2913b59cc46275f84daa06889364fa7ff454b9409ed629aacc
```

No GPU, ROMEO, official OpenMPI/Kokkos4.4 or GitHub CI result follows. Full kinetic/thermal/Hall/Marshak/nonlocal/Cahn–Hilliard and broader original corpus obligations remain open, with their existing equations and missing inputs preserved.
