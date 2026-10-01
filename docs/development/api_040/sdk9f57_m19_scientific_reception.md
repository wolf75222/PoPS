# SDK9f57 M19: actual saved-state scientific reception

ROOT approves two independent archived receptions from the genuine installed Dim2 PoPS package: six finite-product Reduce/Lift variants in Serial and six in MPI2, each with initial, accepted, restored and replayed phases. The independent equations use exact velocity weights and signed component sums; the largest original-equation error is **2.842170943040401e-14**. Restored and replayed states are checked byte exactly.

The MPI envelope receives the entire actual 13-test XML on each rank, with exactly seven named non-M19 cases permitted and no failure/error/skip filtering. Those additional fixture passes are not assigned the M19 scientific seal.

ROOT verified all 2,892 snapshot entries against 1,406 archive blobs, the 1,128 installed and 1,128 checkout production files captured before upgrade, and all 610 unchanged original campaign files. The portable archive prevents current ENV/source changes from being mistaken for the archived run. Original fixture files are additionally checked against the execution Git commits. The build receipt identifies the actual compiled C++ source as `0abbe25395a44e570d8d5525693b8e2dcbf4d387`; Serial execution uses checkout `b167556663a0736c22699be6976f17c81e26d05b` and MPI2 uses `cfd5b01ed6c409879a823e9f93a9f93b5f5198cb`. The Python installation includes the separate metadata repair.

Each reception requires two external ROOT digests: the exact owner manifest and a separate approval referring to that digest. The pure independent reader is extracted from its SHA-pinned frozen archive and run with isolated Python; it imports no PoPS runtime. Nineteen copied and resealed countermodels are refused, including a 1-ULP restart mutation below the numerical tolerance. These negative-only envelopes do not impersonate authentic ROOT approval.

| Backend | Manifest SHA-256 | Approval SHA-256 | Independent report SHA-256 |
| --- | --- | --- | --- |
| serial | `8fe3cf1aa2befa14c72aaad4dad9379c0799dfcf20ff6689281e78d599199bf5` | `8bea510ef7e41973ecbdbd33b76e2e1d547c70c075dea92277b102e918b3d9ec` | `9281a79a6d2fdf88d9a9239928df9dac3145570fbc4a5adf3e332441727afdb8` |
| mpi2 | `112d0f585aecb2fdb717bf03baea03c8a7c2ee73e7dbe5e2926d2d6be68a6203` | `de0ca3ce79e8123b70b43e7edc60fc50b356e9ec17132da7bcf43e05ce77bd7e` | `521fded2e0b4fc28e3c83f1d950762da04622dbbe9dadb4336889c8763a7f529` |

Reproduce the frozen independent receptions after checking the pinned archives and manifests:

serial:

```sh
env -u PYTHONPATH /Users/romaindespoulain/miniforge3/envs/pops-api040-ir17/bin/python -I /Users/romaindespoulain/dev/tmp/pops-api040-native-reception-evidence-20261001/m19-sdk9f-root-approved-20261001/frozen-independent-reader/tests/review/sol61_m19_saved_reception.py check --pins /Users/romaindespoulain/dev/tmp/pops-api040-native-reception-evidence-20261001/m19-sdk9f-offline-pending-20261001/serial-owner-pins.pending.json --pins-sha256 8fe3cf1aa2befa14c72aaad4dad9379c0799dfcf20ff6689281e78d599199bf5 --approval /Users/romaindespoulain/dev/tmp/pops-api040-native-reception-evidence-20261001/m19-sdk9f-root-approved-20261001/serial-ROOT-approval.json --approval-sha256 8bea510ef7e41973ecbdbd33b76e2e1d547c70c075dea92277b102e918b3d9ec
```

mpi2:

```sh
env -u PYTHONPATH /Users/romaindespoulain/miniforge3/envs/pops-api040-ir17/bin/python -I /Users/romaindespoulain/dev/tmp/pops-api040-native-reception-evidence-20261001/m19-sdk9f-root-approved-20261001/frozen-independent-reader/tests/review/sol61_m19_saved_reception.py check --pins /Users/romaindespoulain/dev/tmp/pops-api040-native-reception-evidence-20261001/m19-sdk9f-offline-pending-20261001/mpi2-owner-pins.pending.json --pins-sha256 112d0f585aecb2fdb717bf03baea03c8a7c2ee73e7dbe5e2926d2d6be68a6203 --approval /Users/romaindespoulain/dev/tmp/pops-api040-native-reception-evidence-20261001/m19-sdk9f-root-approved-20261001/mpi2-ROOT-approval.json --approval-sha256 de0ca3ce79e8123b70b43e7edc60fc50b356e9ec17132da7bcf43e05ce77bd7e
```

The [machine receipt](sdk9f57_m19_scientific_reception.json) retains exact inputs, approvals, commands and limitations. These receipts qualify finite-product transfer and exact saved-state restart on the archived SDK9f57 execution. They do not qualify full Vlasov transport, BGK, a self-consistent field solve, GPU execution, the newly integrated IR19/20/21 source, or the full migration. The original resolved-plan payload/system model CPP were not retained; CompiledPlanRecord/artifact aggregate and build source are ROOT-owner-attested, and an independent CPP-to-DSO compile graph remains unproven.
