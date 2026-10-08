# SDK9f57 actual Dim2 build reception - 2026-10-01

The repository build at source `0abbe25395a44e570d8d5525693b8e2dcbf4d387`
completed and installed the real PoPS Dim2 Kokkos/MPI module in
`/Users/romaindespoulain/miniforge3/envs/pops-api040-ir17`.
This receives compilation and installed identity. Scientific execution remains
separately required for the newly integrated contracts.

The [verbatim build receipt](sdk9f57_dim2_build_reception.json) is also retained
at the original evidence location and has SHA256 `99ef67b4fefbf54fbbbbb0af35b746e6472a91bfa2d76f00b943908d52df12fa`.
It pins the actual command, build log, CMake cache, exported compile commands,
Ninja commands, retained wheel, SDK manifest and installed native identity.
The wheel's selected native member equals the installed module byte for byte.

- Source production inventory: 1,223 files, unchanged before/after the build.
- Installed Python/header inventory: 1,128 authenticated files.
- SDK/header signature: `9f57def42c90cfc0e7ee127afa5050d95f27ecbc65b4ed198f788a097e4b4bad`.
- SDK manifest file SHA256: `64801a0868bed87060d969ebe1911b4c5937bbe72f55c96000e6c9ae60d9dbac`.
- Dim2 native SHA256: `c8a45b013042f53a76ca6754a0285203aa5a9fda6a18f1d580e181dec99642fe`.
- Compiler: actual Apple LLVM 21.0.0; C++20, libc++, Kokkos5.2.0,
  MPICH4.1.2, parallel HDF5, CPU arm64. Doctor passes for the selected Dim2 module.

The new source includes global-field-history-storage@1/IR16,
spatial-interaction@1/IR17 and spatial-interaction-history-source@1 with
conditional IR18, plus the authenticated multi-layout checkpoint diagnostic
inventory correction. Non-history profiles retain their earlier IR and CPP.
Neither source/header tests nor this build receives distributed execution,
AMR Stage, nonlocal mathematics or a complete corpus model.

SDK2e4 in ENV `pops-api040` is preserved for its sealed historical receptions.
The new ENV still contains the copied SDK2e4 Dim1/Dim3 modules at this point;
those dimensions must be rebuilt before using the new SDK with them.
No CPP-to-DSO cryptographic graph, remote CI, GPU or ROMEO result is asserted.

Reproduction uses the versioned
[build driver](build_repository_sdk_reception.py), an already prepared isolated
ENV, the exact source commit, a fresh output directory, and the mandatory
preserved-package source/copy receipts. The receipt's `command.json` contains
the exact successful invocation of `scripts/build_python.sh --dim 2 --mpi`,
with a single heavy compilation slot and exported compile commands. Setup is
not repeated in this checkout. Use `env -u PYTHONPATH` for all installed checks;
`run_installed_checks.py` accepts repeated `--test` selectors.
