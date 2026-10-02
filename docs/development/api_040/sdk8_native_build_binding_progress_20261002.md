# SDK8 native construction and bind reception — 2026-10-02

The [pinned receipt](sdk8_native_build_binding_progress_20261002.json) records
actual repository builds and installed-package tests. Native scientific
acceptance remains false. The following are separate, bounded results:

| Execution | Actual result | Scope and limit |
| --- | --- | --- |
| Build `bee0787c` | FAIL | New Newton consensus header absent from packaging manifest; fixed by `e19ce632`. |
| Build `e19ce632` | FAIL | Full AMR translation unit exposed `levels` outside its scope; fixed by `8ba956b8`. |
| Build `8ba956b8` | PASS | ABI8, Dim2 CPU Kokkos/OpenMP, MPICH and parallel HDF5; 1,135 installed sources authenticated. |
| Public typed Original Newton, Relative/Uniform | 1 PASS | Genuine installed SDK8, two native triplets and independently reconstructed original norms from saved NPZ; other policies/backends pending. |
| Repository C++ `test_mpi_amr_spatial_norm` | 5 collective cases PASS | MPI2/OMP1; five cases on each rank, zero failure/error/disabled. Ownership, GMRES, grown-provider refusal, policy drift/overflow voting and rank-local publication damage. |
| First public Tag bind | 1 FAIL | Real pybind setter required an unregistered `Extent<2>`; fixed by the existing ranked converter. |
| Build `aabaa816` | PASS | Corrected pybind property; source/HEAD stable during build, immutable wheel/native equal, old SDK2e4 preserved. |
| Real corrected extent setter | 1 PASS | Anisotropic tuple roundtrip and seven malformed-input refusals leave the prior value intact; no mesh or JIT needed. |
| Second public Tag bind | 1 FAIL | A later runtime-authority consumer still requires execution schema2, while the optional Halo effect emits schema3. Correction and full public reception pending. |
| Shared execution contract, `f1b32ede + 69ec8e3f` | 245 Source PASS independently | Exact v2/v3, validation before callbacks, declared/runtime drift, detached metadata, integer checkpoint schemas, dimension and native integer bounds; no runtime qualification from spies. |
| Build `3d8481c9` | PASS | Corrected Python chain installed: 1,136 authenticated sources, unchanged DSO `7fd8c7fe`; genuine public Tag reception in progress. |

The independent SDK audit verifies the actual ABI8 import, setters and 1,135
installed source files. The primary-clock symbol is confirmed by `nm` in the
already compiled **program component**, not as a method on the core SDK classes.
The independent binding review verifies the precise ranked converter and
43 Source tests. Source fakes alone had missed both integration seams.

Successful builds retain their wheel, command, CMake cache, Ninja commands,
compile commands and full before/after production inventories. The old DSO
`33eedc3d` is retained inside the `8ba956b8` wheel; the isolated target prefix now
contains DSO `7fd8c7fe` from `aabaa816`. Historical origin paths in receipts
describe the execution time and are not immutable SDK copies. SDK2e4 and SDKbb416
use distinct preserved prefixes. C++ to DSO cryptographic graph proof remains
false; the observed build/source evidence is explicitly scoped.

Reproduction uses the existing repository drivers. Use the exact command arrays
in each pinned build/run JSON, with `env -u PYTHONPATH`; the native harness checks
the installed package, specialization, DSO digest, doctor and every shipped
Python/header source. Relevant test nodes are:

```text
tests/python/integration/amr/test_accepted_halo_extent_binding.py
tests/python/integration/amr/test_public_tag_selection_native.py::test_public_tag_buffer_preserves_periodic_selection_and_parent_ghosts[linear-0-shape0]
tests/python/integration/runtime/test_public_newton_typed_original.py::test_public_typed_original_policy[relative-uniform]
```

The Tag attempts remain failed. No physical tolerance, equation, Q guard, full
grown-carrier guard or scientific reader criterion was changed to obtain a pass.
Full Tag Serial/MPI, typed Newton variants/rollback, nonconstant N8/N16 AMR,
signed free transport and the remaining 94 obligations are still required.
GPU, ROMEO, other native dimensions and GitHub CI have no new qualification here.
