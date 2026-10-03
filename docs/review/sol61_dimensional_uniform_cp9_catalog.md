# Uniform CP9 dimension catalog

Historical private fixture SHA256 `16c11e77a4dc0cb54c48485d73539c3c3b763960e1a67314267268a8ed659965` remains byte-preserved externally. At initial gel478, its scientific body was AST-identical in the shared support function. The current delta changes only package-origin authentication, as detailed below; the scientific body is preserved. Parametrization over all three dimensions is replaced by three explicit file entries; no equation, storage/clock/checkpoint guard, refusal, replay, or C25 requirement changes. Existing dimensional receipts are not reruns of this new catalog.

`tests/python/conftest.py` selects POPS_NATIVE_DIM once before collection. `scripts/ci_python_dimensions.py` partitions files through native_dimensions.json, default Dim2 with explicit Dim1/Dim3 entries. The existing tests/test_manifest.toml integration/runtime suite already covers these files. Each entry refuses a missing/wrong launcher dimension without skipping or switching to another DSO. The genuine support then selects that exact already-authorized specialization and requires installed origin plus native/compiler/Kokkos fixtures. Missing Native in required-native launchers fails through existing repository guards; Source collection is not Native execution.

ROOT runs separately for D=1,2,3 against the corresponding genuine installed package:

```
env -u PYTHONPATH POPS_NATIVE_DIM=$D POPS_REQUIRE_NATIVE_TESTS=1 python docs/development/api_040/run_installed_checks.py --output "$RUN" --test tests/python/integration/runtime/test_uniform_cp9_dim${D}_runtime.py::test_installed_dimensional_cp9_two_states
```

For MPI2, use `run_installed_mpi_checks.py --ranks 2 --dimension "$D" --threads 1 --timeout 6000` with the same exact node and output. Configure Kokkos roots to that activated clone. No job or installed environment is modified/submitted by this patch.

Seven Source tests authenticate the real CI partition, refusal before helper invocation for wrong launchers, and actual public validate/resolve in all dimensions. Three nodes collect only. Official setup ran once in dedicated pops-sol61-dimensional-cp9-catalog (Dim2), exit0; PoPS is not installed there and no Native build/run was performed. Source tests use existing ir17 with env-uPYTHONPATH, absolute WT/python and WT, --noconftest -p no:cacheprovider. External original/proof/XML/setup receipts: sol61-dimensional-cp9-catalog-preparation-20261003.

## CI route correction

The negative review of478 is preserved: CI initially built/downloaded only dimensions1/2 and rejected its legitimate downloaded package outside sys.prefix. Current ci_plan admits dimension3 through the existing dynamic build/prewarm matrix, and the shard downloads its exact Dim3 artifact. The runner declares its exact package root. The witness separately authenticates CI artifacts at checkout/.pops-ci/python-packages/dimN: exact Python file coverage and hashes against checkout, imported package origin, existing variants manifest/DSO hash/capability verifier, and nonmissing baked native header signature against checkout headers. This is not permission to use arbitrary Source/prototype packages. Historical installed executions still require the active prefix, with no CI declaration. No Native compiler/backend capability is fabricated by this new test-origin route.

The only scientific-helper body difference is replacement of the installed-prefix-only assertion by this origin authority call. Therefore receipts from private SHA16c11 do not automatically qualify this changed fixture/CI workflow. CI CTest still has its separate existing ownership; these entries belong to the manifest-driven pytest dimension partitions. Local Source checks do not constitute a GitHub CI result. Tiny source authority seam stand-ins are explicitly SourceOnly; existing manifest/extension verifier tests cover its strict admission.
