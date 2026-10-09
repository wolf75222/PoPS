# Accepted exchange ownership on replicated carriers

The actual installed Dim2/MPI2 reception on 2026-09-30 exposed a production
ownership defect. For a replicated AMR root carrier, both ranks staged the same
physical right-face current. `consume_external_trace_collectively` then correctly
summed the local ledgers, producing `q=.72` instead of `.71` for `q0=.7`,
`F=1`, face measure `1/8` and one accepted interval `dt=.01`.

The isolated receipt is
`outputs/installed-integral-transport-amr1-mpi2-abi5-307-isolated-20260930`
in the task workspace. Both ranks fail the unchanged `2e-13` scalar guard.
The saved checkpoint contains two identical POPSEX02 images with SHA256
`dad50f7c904dcc00da8134e65ab3c9a9c4cd85eb66370d70ec613736579da9be`:
each has 256 records, eight right exterior faces, and total right amount `-.01`.
The installed native SHA256 is
`d80e633cb18d352643091fa9677ec35b4851bcade6653f9fa0ad64d5566fa744`,
SDK `307f61570e511eae829151680b4dd2e24425907803f4ec78b2dbb32a19b82540`.
The preceding eight-test campaign reached the same AMR1 failure, then timed out
in AMR2 after 1200 seconds. Its incomplete rank results are retained and are
not qualification of the other cases.

`accepted_exchange_contributes` now uses the native carrier's distribution and
the prepared lane's rank: distributed fields contribute their local owners;
replicated fields contribute on lane rank zero. Transport emission and prepared
diffusion use the same rule. Every rank still enters mask preparation, error
votes and batch staging, including ranks whose batch is empty. Coarse/fine
coverage and embedded-boundary activity remain separate selectors. This fixes
ownership without changing PDE evaluation, temporal weights, IntegralState
reduction or POPSEX01/02 encodings.

The public reception additionally checks that the union of selected face keys
over saved rank images is unique, and checks its exact physical face count.
Host probes execute both production producers with replicated rank-zero/rank-one
carriers and a distributed rank-one carrier. They supplement actual native/MPI
reception; they do not substitute for it. Reception of the rebuilt SDK is pending
at this integration freeze.

Reproduce after an incremental Dim2 MPI build in the project environment:

```sh
env -u PYTHONPATH POPS_ENV_NAME=pops-api040 bash scripts/build_python.sh --dim 2 --mpi
env -u PYTHONPATH FI_PROVIDER=tcp OMP_NUM_THREADS=1 POPS_THREADS=1 \
  python docs/development/api_040/run_installed_mpi_checks.py \
  --dimension 2 --ranks 2 --threads 1 --timeout 360 \
  --output /absolute/new/receipt \
  --test 'tests/python/integration/runtime/test_integral_state_trace_runtime.py::test_native_accepted_face_amount_and_initial_read[amr1]'
```

Use the same `CONDA_PREFIX`, `Kokkos_ROOT`, `POPS_KOKKOS_ROOT`, `CMAKE_PREFIX_PATH`
and `POPS_INCLUDE` as the build; the receipt authenticates the installed package,
extension and shipped sources before and after execution. The historical image
does not qualify a later SDK. Native artifact and checkpoint authority remain
bound to their compiled identities.
