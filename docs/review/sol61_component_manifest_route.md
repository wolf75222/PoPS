# Detached compiled component manifest route

The documented CompiledModel.manifest and CompiledProblem.manifest return remains CompiledArtifactManifest. Their existing component_model_metadata route validates exact low-level types and, for CompiledProblem, a unique program-block route. The rich report now uses that route; build_compiled_manifest remains restricted to the whole CompiledSimulationArtifact and still intersects all block capabilities. No first-model fallback was added.

A detached model carries conservative variables, roles, parameter declarations, provider inputs, ABI and capability facts. It does not carry the resolved installation plan needed to derive ghost depth or Field outputs. These values remain None/empty; no reconstruction default or runtime storage authority is manufactured. The original compiler-owned ModuleManifest is retained unchanged. CompiledModel.arguments() still independently requires its existing resolved-plan authority and is not repaired or invoked by this manifest route.

Source checks use a real CompiledModel returned by compile_install_model with an explicitly Source-only compiler fixture, and exact typed two-block CompiledSimulationArtifact records in both declaration orders. Only platform metadata discovery is isolated for the aggregate fixture. No native library is loaded, and no dump_ir or runtime execution is claimed.

Historical parent caller produces the exact TypeError requiring CompiledSimulationArtifact (one RED). The corrected coherent cohort gives 21 PASS, 2 explicitly compilation-dependent metadata nodes deselected, 4.56 seconds. The initial exploratory wider run exposed absent native selection in pre-existing fixtures and the low-level arguments-plan seam; it did not provide native evidence.

Evidence and exact command: /Users/romaindespoulain/dev/tmp/sol61-component-manifest-fix-20261003. Parent freeze 63efb0661a968079a059cbaa83f12319f09fd4ac. Existing report schema is unchanged; this is an additive exact component builder and caller routing correction, not a new scientific or storage contract.
