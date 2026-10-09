# Accepted auxiliary storage and lazy invalidation

An accepted external Field publication invalidates downstream providers. The exact registry
contract deliberately defers their evaluation until the declared consumer requests them.
Checkpointing must preserve that state rather than evaluate providers or label old values fresh.

Clean images continue to serialize byte-for-byte as POPSAUX2. POPSAUX3 retains the same body and
adds a lexicographically ordered, unique list of invalidated provider identities. Every listed
provider must belong to the sealed registry, be non-Input, and have a valid last accepted point.
Unpublished Input values, providers without an accepted publication, foreign identities and
malformed extensions remain refused. The native decoder owns structural admission.

Uniform and AMR restore prepare storage, accepted publication metadata and invalidation lists
privately, vote preparation failures, and publish with allocation-free ownership swaps. AMR's
invalidation list is hierarchy-wide and therefore must agree across all saved levels. A rejected
restore publishes none of these candidates. Subsequent consumer refresh uses the restored list
as the existing forced-provider input; capture and restore do not launch providers.

This is auxiliary wire version 3, independently of Uniform CP9 and AMR12. The C++ accepted-state
DTO and headers change and require a rebuilt matching Native package. The coherent release
contract advances Native ABI10 to ABI11: the actual accepted-state DTO grows from 104 to 128
bytes on the measured Host profile, and the physical-time evaluation extension also changes
the point layout. Generated C++/Python release products and the complete header signature
must match the rebuilt module and every generated loader. Existing ABI10 Native/loader
receipts do not qualify ABI11. This ABI change is distinct from the retained CP9/AMR12
envelopes and from the auxiliary decoder's POPSAUX2 compatibility. Provider-empty migration
attestation remains POPSAUX2. Ghost values are restored by the existing declared ghost routes;
this extension does not qualify a ghost formula, scientific execution, or GPU/MPI runtime.

Source validation uses the real ExactAuxiliaryRegistry and checkpoint header in dimensions 1–3,
the actual Uniform Python wire guards, and syntax compilation of both complete runtime units.
Native checkpoint/restart reception remains a separate rebuilt execution owned by ROOT.


Uniform installed Program clock ownership contract @1
---------------------------------------------------

The Uniform future auxiliary reserve is owned by the exact installed Program DSO,
including cold providers that acquire their first accepted point later. It uses the
longest installed logical clock identity and the sealed registry's wire framing;
pending Input publications are not accepted invalidations. Raw registry publication
with an unrelated longer clock remains outside this Program-owned certificate.
Accepted checkpoint restore prepares the Input cache from the accepted image and
publishes it with the existing full transaction. Full pending-attempt rollback
continues to retain its own pre-attempt state.

A Uniform Program DSO must export
`const char* pops_program_checkpoint_clock_manifest_contract()` returning exactly
`pops.program.owned-clock-manifest@1`, together with the mandatory logical-clock
count, indexed identity and primary identity accessors. The count is positive and
bounded; all identities are nonempty and unique and the primary belongs to the
owned set. The exact schema discriminator is checked before invoking any clock
callback, before invoking the install entry and before publishing installation
state. Missing, null, empty, future-version and near-match discriminators fail
closed and require artifact regeneration. The reader records typed version1 in
`ProgramOwnedClockManifest`; version, owner, primary and clock set enter exact
installation authentication and the existing install snapshot/rollback owner.
Clock identities authorize provenance and never select numerical formulas.

This is a deliberate mandatory Uniform Program metadata contract change. Old
Uniform Program DSOs lacking the discriminator are rejected even if their table
exports happen to exist. The matching SDK header signature changes and generated
Program cache inputs change, so regenerate Uniform DSOs against the rebuilt
matching runtime. The Native ABI number, Uniform checkpoint envelope and auxiliary
wire versions are unchanged by this metadata supplement; no old Native receipt
qualifies the modified headers. AMR keeps its existing temporal metadata contract
and does not consume this Uniform-only export. Handwritten Uniform fixtures must
provide the same contract as generated DSOs; intentionally wrong-ABI fixtures
continue to fail at their earlier ABI fence.

Durable tests reside in `test_exact_aux_registry_nd.cpp` (cold reserve, long owned
clocks, accepted cache restore, invalid manifest/overflow and existing install
rollback), `test_program_abi_symbols.cpp` (complete export surface and eight real
isolated DSO negative contracts), and
`test_uniform_clock_manifest_contract.py` (actual pure emitter and installed-owner
capacity helpers). Source-only host checks do not qualify installed Native,
System execution, MPI, GPU or scientific checkpoint/restart behavior.
