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
those absent facts. `test_checkpoint_migration.py` supplies that mapping together
with a separately authenticated current-v6 authority and proves the resulting
artifact passes strict restart. The current runtime must continue to refuse this
v2 file directly.
