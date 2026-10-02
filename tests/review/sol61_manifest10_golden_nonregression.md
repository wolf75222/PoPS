# ModuleManifest 10 golden expectations — Source only

Commit 79db292961b25a0869c917cc92cba79252ad8154 (joint primitive coordinates, 29 September) changed the ordinary ModuleManifest wire from 9 to 10. Physical global quantities separately use 11. CHECKPOINT.md records version 10 and explicit refusal of 9. This patch changes no manifest implementation or identity authority.

The exact pre-M16 parent 3f5a55026f39de3429f3bbb552d043a2a65a9d09 already has SCHEMA_VERSION=10 and both golden assertions incorrectly expecting 9. Thus the failures precede the M16 facade/composition changes. Only those two expectations are corrected. The independent principal primitive test's 9 remains an intentional old-wire rejection input.

A param-manifest test explicitly exercises wire-10 roundtrip, identity hash equality and rejection of wire 9 without mutating the module. Existing deeply frozen/copy-out and principal primitive authority tests remain in the affected cohort. Source results do not qualify a Native artifact or CI.
