# SDK9f57 actual MPI2 non-regression

All thirteen selected cases pass on each of the two real MPI ranks, with zero
failures/errors/skips, matching test inventories, no timeout, and unchanged
installed production and test sources before/after. Duration: 1,229.58 seconds,
Dim2 CPU Kokkos, MPICH with `FI_PROVIDER=tcp`, one OpenMP thread per rank.

The six finite-product Reduce/Lift/initial-checkpoint/restart variants no longer
exhibit the former diagnostic-capacity failure. The remaining seven cases cover
twenty entropy interior targets and outside-cone refusal, four original-source
integral-feedback/exterior/restart variants, and two rejected-transport
capture-preservation/retry variants. Each rank repeats the shared experiment;
this is thirteen collective cases, not twenty-six independent experiments.

[The immutable machine receipt](sdk9f57_product_entropy_feedback_mpi2_nonreg.json)
pins complete raw rank XMLs/logs/identities, the launcher result and source
inventories. Its exact test source is `cfd5b01e`; installed Python is `378f2908`
and original native compilation remains `0abbe253`. All 1,128 installed
Python/header files and the Dim2 DSO are authenticated before/after. New
headers, later source, full kinetic equations, GPU and GitHub CI do not inherit
these results. Independent saved-state reception is pending separate external
ROOT owner seals on the preserved new-SDK archive.

Reproduce from that source/installation using the repository runner and a fresh
output directory; the raw receipt retains the exact MPI child command.

```sh
env -u PYTHONPATH OMP_NUM_THREADS=1 POPS_THREADS=1 OMP_PROC_BIND=false FI_PROVIDER=tcp \
  /Users/romaindespoulain/miniforge3/envs/pops-api040-ir17/bin/python \
  docs/development/api_040/run_installed_mpi_checks.py \
  --output /tmp/pops-product-entropy-feedback-mpi-fresh --ranks 2 --dimension 2 --threads 1 --timeout 2100 \
  --test tests/python/integration/runtime/test_m19_product_support_runtime.py::test_native_product_reduce_lift_restart \
  --test tests/python/integration/runtime/test_m18_discrete_entropy_runtime.py::test_twenty_interior_targets_and_outside_cone_refusal \
  --test tests/python/integration/runtime/test_public_integral_feedback.py::test_public_feedback_original_source_real_exterior_and_byte_exact_restart \
  --test tests/python/integration/runtime/test_public_integral_feedback.py::test_feedback_rejected_transport_preserves_capture_integral_and_safe_retry
```
