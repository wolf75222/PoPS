# Independent C25 bounded review

Reviewer: GPT-6.1 Sol (Hooke), not the implementation author.

Baseline `06bdbe74c013c1dfab11b08a60ca71a8eccf9b2d` author cohort was
independently replayed: 32 passed, one explicitly deselected Native-authority
node, 13.15 seconds. This is Source plus real tiny host compiler coverage,
not PoPS Native, MPI, GPU or scientific qualification.

The unchanged independent probes first produced five failures and one pass
(7.85 seconds). Two failures expose the same migration defect at the facade
and backend explicit-destination routes: `recompile` invokes compilation
before checking an already published destination. Another compiles TU B while
the evidence helper receives TU A; no association refusal occurs. The last
republishes a coherent foreign binary/spec/semantic at the old handle's path;
public source inspection reports that replacement without authenticating the
handle's existing identities. Another probe changes the TU after the final
authenticated read; the old export returns these changed bytes without a new
hash comparison. These are provenance and migration failures,
not evidence of incorrect scientific execution.

The emitter source-authority mutation probe passes. Both facade and backend
specifications include `pops.codegen-source@1`, independently of SDK/header
identity. The authority intentionally hashes codegen Python implementation
files, not generated TU bytes or a compiler dependency graph. Actual compiler
input has its separate retained, committed evidence.

The real native loader is stronger than a plain pathname `dlopen` assumption:
`AuthenticatedNativeFile` creates a private unique image via `mkstemps`, copies
and authenticates its bytes, and `native_loader.hpp` opens its `load_path()`.
This review does not claim a reproduced stale native handle. A conservative
explicit recompile refusal remains an implementation policy.

Independent command (from this checkout):

```sh
env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 /Users/romaindespoulain/miniforge3/envs/pops-api040-ir17/bin/python -m pytest --noconftest -p no:cacheprovider -o 'pythonpath=/Users/romaindespoulain/dev/tmp/PoPS-sol61-m17-composition/python /Users/romaindespoulain/dev/tmp/PoPS-sol61-m17-composition' tests/review/test_sol61_c25_independent.py tests/review/test_sol61_model_source_evidence.py tests/python/unit/codegen/test_compile_provenance.py tests/python/unit/codegen/test_facade_compile_cache.py tests/python/unit/codegen/test_compile_cache_lock.py -k 'not failed_program_compile_leaves' -q --tb=short
```

The facade/backend compiler seams write explicitly identified metadata-only
fixtures. The TU association and replacement tests compile actual tiny C++
libraries with the host compiler, without importing PoPS Native or invoking
its JIT. The old Source counterexamples are preserved in external
`/Users/romaindespoulain/dev/tmp/sol61-c25-independent-20261002/red-six.xml`.

Author delta `565536d0315e575cf45fcda8bbca58188dd39b49`, on baseline
`06bdbe74`, closes these failures. It votes no MPI operations: this is local
compiler/cache authority. Explicit destination guards execute before the
compiler, the retained TU must match the command source and destination,
inspection authenticates the handle's attached identities while holding the
publication lock, and export hashes the actual bytes returned. Existing cache
lock tests cover genuine processes; the new snapshot lock is reentrant.

Independent final replay: **39 passed, one deliberately deselected
Native-authority node, 20.02 seconds**. All six unchanged probes pass, plus a
seventh checks the imported checkout path and absence of `_pops`. The combined
cohort covers fresh/hit/legacy, policy validation, poisoned/partial companions,
foreign semantic authority, symlinks, real tiny host compiler input/output,
thread-reentrant and cross-process publication locking, and sidecar commit-last.
No remaining blocker was identified in this bounded Source review.

This authority does not authenticate a full compiler dependency graph, prove
scientific behavior or qualify same-process Native reload. The broad codegen
source hash specifically invalidates emitter-only changes such as return910;
it is not a proof of every transitive non-codegen implementation dependency.
ROOT owns integration and actual SDK/Native fresh/cache/inspection reception.
