# V@3 component-manifest consumer correction

Base99c80569. Historical integrated Source receipt d20c1592 retains545PASS/1FAIL. Its failure expected an obsolete TypeError from CompiledModel.manifest(). The public component route now returns CompiledArtifactManifest; this test authenticates the exact class and schema2 JSON roundtrip, variable/role/capability metadata, unknown prepared ghost/dimension and absent Field/native-entrypoint claims. ModuleManifest and complete C25 source provenance are checked unchanged by inspection. No production, physics, tolerance, capture wire, or Native result changed.

Affected coherent four-file cohort:41PASS12.13s, no skip. Source package imported from this private checkout with absolute pythonpaths and PYTHONPATH unset; no Native execution. XML SHA256 bb0106e8b169d704f4fe0a734706720fe9737981aa8e42b193a67ad8612aceb5.
