# AMR gather-v2 engineering preparation

Base exact SDK19 Source 1b64477d2f1c93e856e996a4c83e506fcc91b386. No Native execution, compilation/JIT, installation, ENV or ROMEO mutation performed. The private sparse worktree contains tests only; setup_env was not run under the explicit Source-only/no-ENV scope.

Three public layouts reuse the existing evolved-stage AMR case unchanged before bind: replicated coarse, distributed coarse_max_grid=4 (multiple coarse owners in MPI2), distributed coarse_max_grid=8 (one coarse box and an empty rank in MPI2). Native geometry/owners must satisfy these conditions or the fixture fails. Fine refinement must be partial with holes. No topology is fabricated after bind.

The existing engineering set_block_level_state writes domain-sized dyadic arrays with real −0 into actual owned valid cells, for every actual block and both levels. No advance, ghost fill, physics guard relaxation or custom runtime is introduced. This is engineering storage/getter qualification; it is not a received physical evolution and makes no claim about the validity/readiness of grown ghosts. Genuine full-grown carrier archives and exact getter arrays are saved before assertions. Uncovered fine cells must remain +0, covered cells must reproduce native carrier bytes, and all getter arrays retain −0. Entire carrier images are preserved for separate offline inspection.

Public C25 require authenticates all three fixture models before bind, compiler-owned Program C++/IR, manifests and DSO hashes through the existing retained-provenance@2 helper. Per-rank receipts pin actual images, topology and unchanged time/macro/temporal relation observations. Actual all-rank launcher/XML/package/Source/SDK19 identities remain required by the external ROOT receiver; this prepared script alone supplies no such authority. Prior 732028 failure is preserved.

Source checks: 7 PASS in 48.95s, comprising three real validate→resolve layouts and four pure bit/topology checks. Three Native nodes collect only; no execution. Source changes after this cohort add only C25 provenance/clock persistence to the Native fixture; collection rechecked. Compiler/JIT/Native runtime unknown until ROOT runs it. A future Native rejection during engineering setter/carrier capture must remain a failure rather than be hidden or converted to a skip.

Commands from this checkout:

```
env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 /Users/romaindespoulain/miniforge3/envs/pops-api040-ir17/bin/python -m pytest --noconftest -p no:cacheprovider -o 'pythonpath=python .' -q --tb=short tests/review/test_sol61_amr_gather_object_bytes_preparation.py
```

Future installed execution uses `env -u PYTHONPATH <SDK19-python> -m pytest -o pythonpath= -q tests/python/integration/runtime/test_amr_gather_object_bytes_runtime.py` under the existing ROOT SLURM driver, world1 and actual world2 separately. No Source import override is permitted in that run.
