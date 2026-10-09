# Uniform Fan–Li15 eight-step storage profile @3

This separate `pops.fan-li15-public-composition-native-fixture@3` fixture preserves @2 byte-for-byte. It uses exactly the same public equations, canonical/reverse basis orders, N16, dt=1e-4, eight SSPRK2 steps and unchanged scientific guards. It requires the actual public `observe_accepted_state_storage()` API; absence is an explicit refusal, never a private capture or AMR substitution.

All ranks participate in each observation. Each of nine phases immediately writes its actual rank-local POPSCAR1 bytes; ROOT writes the complete bytes, valid NPY and exact time/tick JSON before the next run. Shared-directory writes are collectively completed before the receipt hashes files. The independent wire decoder verifies full archives, Uniform dimension/Real/world/block authority, rank-local versus complete payload equality, complete nonoverlapping valid coverage, and exact NPY valid bits. All grown payload bits are preserved, including signed zero; no fill, refresh or reconstructed payload is requested. Ghost correctness, checkpoint/restart and full M17 qualification are not inferred from storage observation.

Before bind, every Model uses C25 require and retained actual source evidence/dumps. Program retained CPP, IR, compiled partition and DSO identities use the same strict guards as @2. Original collective exceptions and capture failures remain separate. The independent math reader retains authority for Gauss4/GL48 saved-face checks and cross-campaign permutation at their original thresholds; this author fixture emits neither ROOT approval nor a scientific seal.

Source preparation uses genuine immutable DTOs plus a distinct synthetic wire encoder; these tests are not Native receipts:

```sh
env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 /Users/romaindespoulain/miniforge3/envs/pops-api040-ir17/bin/python -m pytest --noconftest -p no:cacheprovider -o 'pythonpath=python .' tests/review/test_sol61_m17_eight_storage_preparation.py
```

Future installed SDK19-or-later nodes (ROOT alone runs after actual SDK reception):

```text
tests/python/integration/runtime/test_fan_li15_full_eight_storage_runtime.py::test_installed_original_fan_li_eight_storage[canonical]
tests/python/integration/runtime/test_fan_li15_full_eight_storage_runtime.py::test_installed_original_fan_li_eight_storage[reverse]
```

The Native campaign must authenticate the newly rebuilt artifact and receive Serial/MPI2 independently. The current preparation executes no Native compile/bind/run. SDK19 build identity is not inherited from SDK18.

Raw-evidence ordering followup: capture performs only bulk reads. All rank-local/complete bytes, valid NPY, clocks, storage metadata and SHA receipt are persisted first. Only then does a voted integrity guard run, before any following Native step. A refusal retains the partial images and records capture-failed separately. Genuine immutable DTO/independent-codec Source adversaries mutate a grown-only local bit, a complete valid bit and a world envelope; each retains exact raw files and prevents a future run. No Native storage defect is alleged by these synthetic adversaries.
