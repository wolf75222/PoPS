# ADC-667 frozen Uniform-v2 checkpoint

`uniform_v2_ab2_98b7ffe6.npz.b64` is a base64 transport of the byte-exact
checkpoint produced once by the real `System.checkpoint` writer from detached
PoPS revision `98b7ffe6dac02b58e1fe85c653846b57baa27829` (the parent of the
final runtime contract cutover), not by projecting a current checkpoint.

The historical revision was built and installed in an isolated Conda
environment. Its `test_time_history_checkpoint.py` AB2 model ran three
accepted steps on a 4 x 4 Uniform grid before that revision's writer emitted
the archive. Decoding the frozen file must produce SHA-256:

`82490ddc97dbf37e6431c3c0ddb61c30439bdf4df9166f659146634d27766226`

The fixture deliberately preserves the limitations of the real v2 writer: it
is an unsealed NPZ and contains no checkpoint manifest, semantic/artifact/
bind/run identities, temporal restart state, ConsumerGraph cursors, or
qualified field-provider state. Tests decode the base64 without altering the
archive bytes and must never regenerate it from a current payload. A complete
offline migration therefore needs an explicit, version-reviewed mapping for
those absent facts. `test_checkpoint_migration.py` supplies that schema-4 mapping together
with a separately authenticated current-v9 authority and proves the resulting artifact passes
strict restart. The authority natively attests its empty POPSAUX2 image and binary registry
contract; the mapping pins SHA-256 digests of the exact bytes of both. The v2 artifact carries no
auxiliary authority; the v2-to-v9 migration copies only that empty POPSAUX2 image byte-identically
from the current authority, never fabricating it from v2. Its `checkpoint_migration` provenance member is reserved
by the live Uniform budget in a fixed 16 Ki-character envelope. The current runtime must continue
to refuse this v2 file directly.

For the native dimension, a valid POPSAUX2 image is provider-/payload-empty: it has zero persisted
groups, components and providers, but carries a nonempty opaque sealed registry contract and an
`accepted_generation` in `[0, UINT64_MAX)`. It need not equal a freshly sealed bare-registry
contract: real code generation can install zero-valued consumer plans, and this authority's real
contract is 1312 bytes. The attestor alone does not prove compatibility with the target registry;
that exactness comes from the full-image and raw-registry-contract SHA-256 pins, byte-identical
copy, and strict restart against the live target. Generation is preserved provenance rather than
rewritten: this AB2 fixture records generation `0`, while a structurally empty publication may
produce a value greater than zero; `UINT64_MAX` is refused as wrap poison.
