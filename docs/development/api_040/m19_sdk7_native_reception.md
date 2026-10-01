# SDK7 finite-product native reception

Six finite-product cases pass in installed Serial and on each of two MPI ranks, with zero failures/errors/skips. The two scientific ROOT receptions independently check saved states, exact checkpoints/replay, rank ownership, mapping cadence and clocks; maximum original reduction/extension error is 2.842170943040401e-14. Twelve genuine copied/resealed countermodels per backend are refused. These controls run offline and do not count as new native injections.

The authoritative machine receipt is [m19_sdk7_native_reception.json](m19_sdk7_native_reception.json). The C++ DSO was built at bfae73f3. The MPI campaign uses Python b3108c73 and the same C++/SDK bytes after the explicit component-content bind projection repair. Paths remain physical provenance and still undergo byte/symbol verification. Collective plan equality remains enforced. The earlier six-case MPI failure is preserved.

The physics here is a finite reduction with signed weights and a constant lift, at (nx,nv,width)=(4,3,3),(2,5,1),(7,3,5), in both declaration orders. No Vlasov/BGK/Poisson, AMR, GPU, official OpenMPI, or full-family acceptance follows. The real artifact aggregate is recomputed; the original resolved-plan payload and System CPP-to-DSO build graph were not retained. No filenames substitute for these missing proofs.

Reproduce on a freshly rebuilt installed package from the desired checkout, with fresh output directories:

```sh
env -u PYTHONPATH POPS_NATIVE_DIM=2 POPS_REQUIRE_NATIVE_TESTS=1 POPS_KEEP_GENERATED=1 OMP_NUM_THREADS=1 POPS_THREADS=1 OMP_PROC_BIND=false FI_PROVIDER=tcp   /Users/romaindespoulain/miniforge3/envs/pops-api040/bin/python   docs/development/api_040/run_installed_checks.py --output /tmp/pops-m19-serial-fresh   --test tests/python/integration/runtime/test_m19_product_support_runtime.py::test_native_product_reduce_lift_restart
env -u PYTHONPATH POPS_REQUIRE_NATIVE_TESTS=1 POPS_KEEP_GENERATED=1 FI_PROVIDER=tcp   /Users/romaindespoulain/miniforge3/envs/pops-api040/bin/python   docs/development/api_040/run_installed_mpi_checks.py --output /tmp/pops-m19-mpi2-fresh   --dimension 2 --ranks 2 --threads 1 --timeout 1800   --test tests/python/integration/runtime/test_m19_product_support_runtime.py::test_native_product_reduce_lift_restart
```

A new build receives its own SDK/native/source receipts; these commands do not grant it historical SDK7 acceptance. Historical ROOT live and countermodel recipes are preserved in /Users/romaindespoulain/dev/tmp/pops-api040-native-reception-evidence-20261001. The new archived-backing reader uses owner-pins/approval@2 and independently checks all backing bytes while leaving recorded origin paths unchanged. ROOT archive approval is a separate action. Raw states and DSOs remain outside Git.
