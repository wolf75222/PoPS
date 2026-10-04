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
