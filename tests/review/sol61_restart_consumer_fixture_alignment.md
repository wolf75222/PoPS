# Three historical Source consumers aligned — tests only

The unchanged parent reproduces three failures: missing positive full-carrier capacity, and two restore callback signatures missing state_storage. The old 26F broad XML remains preserved; this patch does not requalify its 23 Native import/selection failures or claim inclusion in the earlier Source552 cohort.

The cache-only budget fixture supplies a declared metadata-only full-storage byte authority (4096), independent of the lazy cache inventory it tests. This is not an observed Native capacity or a runtime storage claim. Its artifact is the actual CompiledSimulationArtifact class, with Source typed phase records and an explicit Source PlatformManifest boundary. No fake Native extension is registered; retained diagnostic source remains unavailable.

The two restore fixtures accept and assert state_storage=full, use actual ConsumerCursorAuthority, exact domain-run identities, and publish the restored last-run source in their explicitly metadata-only Native callback boundary. They verify geometry/diagnostic/identity ordering, cursor publication and a distinct authenticated continuation; the regrid case retains the exact transformed-source identity comparison.

Baseline three nodes: 3 FAIL (2.60s). Corrected three plus the previous 20-check epoch/protocol selection: 23 PASS (2.08s), no skips/exclusions. Production diff empty relative f977. Native/MPI restart qualification still requires official rebuild and execution, plus independent review of this tests-only delta.
