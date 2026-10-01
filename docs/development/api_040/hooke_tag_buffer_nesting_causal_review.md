# Independent TagBuffer versus nesting causal review

Reviewer actual GPT-6.1 Sol. Frozen source893888ce, docs-only review worktree. No production fix, package import, native build/JIT or environment change. ROOT actual trace is external evidence, not this review's runtime execution.

## Decision

The native full-initial topology failure is caused by conflating authored tag dilation with derived nesting requirements. It predates checkpoint codec12. Preserve N8 partial-coverage guard and physics. Do not fix by increasing N, changing threshold, removing guards or retaining derived lookahead as tag expansion.

Source chain: AMRTagging Buffer.cells is explicit authoring data (authoring.py363–372; _resolution.py415–436). _resolution.py617 merges it with nesting.minimum_buffer via max. _resolution.py635 sets lookahead from nesting.minimum_lookahead. hierarchy_native.py202–203 exports both as native transition buffers/lookaheads. Native prepare_cluster_shards amr_system.cpp9697–9739 then uses their SUM as candidate dilation reach. The installed ROOT trace records TagBuffer0, transition buffer(2,2), lookahead(1,1), correct mesh_marker leaf and threshold1.03. Thus derived reach3 expands the central two tagged columns across periodic N8.

Lookahead is not an independently authored future forecast here. transfer.py979 derives minimum_lookahead=max(transfer order-1). hierarchy.py185 labels the sources stencil/transfer/reflux/boundary nesting requirements; hierarchy_resolution.py170–179 requires transition buffer/lookahead to satisfy those nesting minima. Therefore candidate dilation should consume only explicit TagBuffer. Both derived buffer and derived lookahead belong to a separately authenticated parent coverage/stencil requirement. A proposed TagBuffer+existing lookahead candidate reach would still conflate contracts, even if that reduced reach happened to pass N8.

Berger-Rigoutsos nesting_covered at356–402 grows temporary parent-required boxes exclusively for a coverage proof. cluster_rec at405–446 retains scan.bounds as candidate/output region. Keep that semantic separation and existing native stencil calculation nested_cluster_options amr_system9600–9635; review how the authored derived lookahead contributes to coverage rather than silently dropping its contract. The smallest coherent fix needs distinct tag reach and nesting fields through resolution, typed lowering, native config/identity and execution. Do not weaken hierarchy_resolution nesting validation or repurpose transition fields without version/identity updates.

## Independent saved-data read

Read actual initial-level0/1 NPZ through standard-library ZIP/NPY header and float64/byte decoding, without NumPy/PoPS/native imports. Coarse forcing channel0=1.3100000000000023, 64/64 above1.03; channel1 marker range0.9639873473537272..1.0360126526462725, 16/64 above1.03. Coarse active0/64. Fine marker64/256 above1.03, active256/256. These are actual saved observations, not generated simulation output. ROOT native trace distinguishes them from the source-only causal reading.

Source component mapping is name-authenticated (amr_system8315–8331); native ABI exports real component_stride (native_tagger_session555), bytecode evaluates leaf.component (prepared_tagging_execution220). The ROOT trace confirms correct channel/seuil, so no wrong-component workaround is justified.

| Principle | Decision |
|---|---|
| 1.1–1.5 | Preserve authored Buffer0 semantics; keep numerical nesting distinct from tag policy |
| 1.6 | Source cause corroborated by ROOT actual installed trace; corrected execution still required |
| 1.7 | No performance/native backend claim from this review |
| 1.8 | Fix common typed contract seam, no per-model or N8-specific treatment |

Trace SHA256: eaa2e47003df3c984ab45261cf9a9c8de21188f762bd65a2acbc5a34938494b6 .

Inspected frozen source SHA256:

```text
13af8a5baeb71e9a2e79ee82a78149885a6fc219cc0017fba29af7503bb770fe  python/pops/amr/_resolution.py
e4dac9727232daf164e692f204283659a17a36767f1a2d53eb7cdf572e12a4f8  python/pops/mesh/_amr/transfer.py
01ee7ef181302c072ac325dd81e7ca9a46fc53fadffbaea6f62778c935b94df3  python/pops/mesh/_amr/hierarchy.py
eab365516954224f4b583ae073a1e90d7a7ce08070c8610ec5d2dce5419cac78  python/pops/mesh/_amr/hierarchy_resolution.py
5f3b48e8f2112f2a565d52c548bcf1f226e96cdb8d69ef293510eea6cdfe1638  python/pops/mesh/_amr/hierarchy_native.py
fcdb3c26b5277aed0af1d0b2acaf2b78c4ec66d19b54dfc05ea96b7f66ae32ca  src/runtime/amr/amr_system.cpp
a59bacc78229b9140a5206254bf297ebf63b73e88b0297f3d93ee62575462a59  include/pops/amr/tagging/berger_rigoutsos.hpp
```
