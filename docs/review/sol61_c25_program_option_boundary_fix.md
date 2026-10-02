# C25 model admission / Program compiler option boundary

SDK17's genuine M16 second-State job731993 failed after two Model TU/DSO
compilations. Public compile-once publication reported `_compile_problem_impl()
got an unexpected keyword argument 'model_source_policy'`. The original raw
failure and transfer remain historical evidence; no scientific result is claimed.

The original C25 patch projected options only in the multi-layout branch.
Single-layout compilation reconstructs options inside `_compile_resolved_problem`
and therefore still forwarded the model policy. Both paths now use one
`_program_compile_options(plan)` projection. `model_source_policy` remains in the
immutable authenticated resolved plan and is forwarded unchanged to every model
by `compile_install_models`. It is removed only from the copy passed to Program
compilation. Compiler flags, force/debug/path/library options retain their prior
meaning. Public option spelling, validation, model require/recompile admission,
retained-source @2, artifact export/import and scientific equations are unchanged.
There is no branch by model, State, fixture, number of MPI ranks or formula.

The new witness constructs a genuine public Case, validates and resolves each
policy, then enters the actual public compile phase after its Native selection /
bootstrap cut line. Only model compiler entries and the Program compiler entry
are Source-only sentinels. The strict Program function signature reproduced the
unexpected keyword for all three policies (3 FAIL21.69s); corrected Source entry
preserves the exact sentinel exception and immutable plan verification. No fake
Native module is substituted. Earlier direct public API attempts correctly
refused absent Native authority; they are not the causal regression proof.

Coherent command from this private checkout:

```sh
env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 /Users/romaindespoulain/miniforge3/envs/pops-api040-ir17/bin/python -m pytest --noconftest -p no:cacheprovider -o 'pythonpath=python .' tests/review/test_sol61_model_policy_program_boundary.py tests/review/test_sol61_model_source_evidence.py tests/python/unit/codegen/test_compile_provenance.py tests/python/unit/codegen/test_facade_compile_cache.py tests/python/unit/codegen/test_compile_cache_lock.py -k 'not failed_program_compile_leaves' -q --tb=short
```

36 PASS, one preexisting Native-authority node deselected,33.09s. No Native/JIT,
ENV, Main or ROMEO mutation was performed. ROOT must independently review and
rebuild SDK18 before rerunning authentic public Native fixtures. SDK17's two
failed jobs remain failures rather than being retrospectively qualified.
