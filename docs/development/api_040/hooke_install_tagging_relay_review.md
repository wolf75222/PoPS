# Independent actual InstallPlan tagging relay review

Actual reviewer GPT-6.1 Sol Hooke; frozen authorfaa9486d5cf134babe54b86c8b8ff0b8cee22f27, parent68d3cbbb. Separate docs-only worktree; Native/ENV untouched. No native/JIT/C++ build.

Decision: no concrete Source blocker. InstallPlan._resolved_tagging_for_layout authenticates the requested qualified ID among actual artifact LayoutPlan layouts, obtains exactly that registered ResolvedLayoutAMRAuthorities, checks local.layout_id, then explicitly requires exact ResolvedTaggingAuthority. Singleton resolved_tagging rejects len!=1 rather than selecting the first arbitrary layout. LayoutInstallProjection rechecks its local object against parent's registry and delegates by its selected compiled layout ID. Existing projection construction already authenticates selected CompiledLayoutProgram and local authority identity. No new field/model/physics selection, shared table assembly or fallback tagging.

Actual native failure log installed-sdkbb416-amr12-representative-serial-dim2/pytest.log was read: original[1-8] failed at bind due missing InstallPlan.resolved_tagging. The earlier e0 source review followed LayoutInstallProjection but missed the concrete mono InstallPlan class; this follow-up corrects that scope error. It does not overwrite earlier evidence or pretend the failed native run passed science.

Author tests construct an actual public-resolved AMR phase plan, exact CompiledSimulationArtifact and exact InstallPlan. Positive assertion preserves the exact same local tagging object and a genuine registered child projection. Negative cases cover actual typed2-layout aggregate unqualified lookup refusal, uniform layout absence, foreign ID, removed parent registry and nonexact tagging. Only executable-platform discovery is monkeypatched to metadata; fake plan namespaces are not used for the positive relay. Multi-AMR native execution is not tested by this fixture; the2-layout refusal test uses a typed heterogeneous uniform aggregate.

Independently executed frozen tests:5passed7.43s. Source pytest command disables repository conftest and explicitly chooses checkout/python. No native load/compilation calls are in the tested seam; author reports loaded native extensions0. This receiver run does not requalify installed source, native setter, geometric AMR science, checkpoints or MPI. ROOT must sync/authenticate installed Python and rerun original strict N8 first. The fix changes Python forwarding only, no C++ ABI shape; native DSO reuse remains subject to ROOT identity checks.

Reproduction:

```sh
env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 /Users/romaindespoulain/miniforge3/envs/pops/bin/python -m pytest --noconftest -o pythonpath=python tests/python/unit/codegen/test_install_tagging_authority.py -q --tb=short
```

Frozen SHA256:

```text
fe2e708cc85fd8748bdc594471cdaf29fc5164f0e800e270094b380bf9850555  python/pops/codegen/_plans.py
58db879c1306e39d5b26bfcd3dbda5c43c0b9ab0dd7dc546496e4fbfe8c32a95  python/pops/runtime/_layout_install_projection.py
c2baae74eb00f9b33c6532891464939265f458e14134eed1d8d4c3c2ea67fbaf  tests/python/unit/codegen/test_install_tagging_authority.py
```
