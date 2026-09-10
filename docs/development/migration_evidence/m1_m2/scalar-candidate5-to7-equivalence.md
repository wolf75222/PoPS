The candidate 5 full scalar result applies to the final scalar path as **proved source-equivalent scientific evidence**. Candidate 7 did **not** execute the full scalar profile in this comparison. No material change to the inspected scalar execution path was found.

Candidate 5 executed the unchanged 128 × 128, three-level AMR profile, including manual continuation, scientific-output reopening, strict bit-identical restart, and an independent full SSPRK2 preset run. It passed in 3937.817645 seconds and reached step 456 at time 0.4. The reported relative L2 error was 0.01700899 at the output sample time 0.193359. The execution belongs to source `76319c0658187d9542c527870854321609a0f891` and native `16342cd011e2dc8d002ee180971060ef5a3925367e4953595fc5ab2f49c8ce02`. [Execution receipt](/Users/romaindespoulain/dev/tmp/PoPS-migration-20260907-evidence/candidate5-examples/scalar/report.json).

The comparison used the immutable candidate 5 and candidate 7 Python environments, checked their changed Python members against their exact source checkouts, and loaded their authenticated Dim2 native extensions. Candidate 7 is source `bd583faf196f3c1faeedec489e04d24e959ed00b`, native `d1eb7da32a5c0a7b2e7433dfdd309d03d4b19a8afaab5f426d03ad5ee875f518`. Both manual and preset cases went through validation, resolution, exact Module/provider lowering, production native-loader emission, Program detachment, and the resolve-authenticated production Program emitter. No compilation, simulation, binding, installation, or package write was performed. A common inert output-root string was used for these pure constructions; the complete numerical profile was retained.

| Compared value | Candidate 5 versus candidate 7 |
|---|---|
| Scientific Module and model identities | Identical |
| Program graph and semantic identity | Identical; also matches the completed scalar run’s reported Program hash |
| Complete production native-loader C++ | Byte-identical for manual and preset; SHA-256 `363a39540a2f841470e00664b900de1566f9f2d4e82aad22aabc9a41354a9243` |
| Complete production Program C++ | Byte-identical for manual and preset; SHA-256 `156fc423a18196e0b93d1e33b45f93809dd0b1e03ab03122eda229b5aba93a58` |
| Bind parameters, selected signed terms, reads, outputs, routes, guards, methods, halos, and AMR plan data | Identical |
| Resolved operation records | Four records gain conservative `opaque`/`fallible` boundary effects; their identities are recomputed |
| Snapshot semantic identity | Identical |
| Snapshot artifact differences | Only source-file provenance and its derived IDs differ between the immutable checkout/environment paths |

The remaining changes are bounded by the actual scalar graph: one state read, two operator calls, two Program values, and one commit. ScalarUpwind uses explicitly authored wave speeds, so the qualified AD/Roe path is absent. There are no auxiliary providers, DerivedAux launchers, local solves, or projection steps. The exact InputAux gate sees empty declared and supplied sets; empty BindInputs identity and auxiliary snapshot evidence remain unchanged. The richer effect metadata changes reporting and identity, while the complete generated C++ remains identical.

The existing independent [AMR auxiliary equivalence proof](/Users/romaindespoulain/dev/tmp/PoPS-migration-20260907-evidence/aux-outcome-amr-equivalence.json) is reused. Its 28 recorded AMR source blobs still match candidate 7, and no native source changed between that review’s endpoint and candidate 7. Its finite-wrapper and preparation analysis was not repeated.

This establishes source equivalence for the inspected scalar path, not native-binary equality, actual candidate 7 scalar execution, cross-candidate bit-identical results, or cross-candidate checkpoint compatibility. Artifact and ABI identities remain distinct. An acceptance rule requiring a run on the exact final binary still requires that run. The scope does not extend to other dimensions, MPI scaling, GPU, or later migration stages.

[Machine-readable proof, exact differences, hashes, and limits](/Users/romaindespoulain/dev/tmp/PoPS-migration-20260907-evidence/scalar-candidate5-to7-equivalence.json).
