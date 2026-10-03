# Restart continuation epoch — Source preparation @1

Actual SDK23 job732282 failed at the first replay run after successful restart: the consumer correctly rejected a closed run identity. This patch preserves that refusal and immutable consumer ledgers.

A successful public restart authenticates a deterministic continuation chain from the archived run (including None), the owner’s prior last run and prior continuation. The exact token is voted before native transaction begin. Source-run publication remains historical; only future RunManifest requests use the new lineage. Recursive owner/child snapshots compensate failures, so a failed publication followed by retry yields the same continuation bytes. Repeated successful restart of the same archive advances the chain even without an intervening run. Regrid restart retains its existing transformed-topology identity and additionally chains the owner continuation.

No checkpoint wire, equations, controls or consumer closed-run guard changes. No random UUID, ledger clearing, private fixture workaround or Native execution. Native restart/replay qualification requires independent review and rebuilt installed Python SDK.

The seven new tests use actual RuntimeInstance.restart/_restore_checkpoint, begin_run and closed-run admission. Their Native restore boundary is explicitly metadata-only. The disagreement test injects peer envelope rows into the real collective orchestrator; it is not an MPI execution.

Baseline12320: 4 FAIL / 3 PASS. Corrected coherent selection: 13 PASS, no skips/exclusions. Six existing Source policy/preflight tests accompany the seven new probes. Broad exploratory runtime_gate selection remains preserved: 26 FAIL /20 PASS /79 deselected; Native import/selection preconditions and old fake restore signatures/resource budget fail in this Source-only environment. These failures are not converted into passes or silently repaired.

Principles→decision→file→oracle: generic continuation ownership (1.3/1.5/1.6), no model branching; runtime_instance supplies owner-history lineage; checkpoint_collective votes before mutation; new tests distinguish replay, repeated rewinds, initial None, rollback/retry, child ownership and closed-run refusal. Source verification does not establish Native/MPI behavior.
