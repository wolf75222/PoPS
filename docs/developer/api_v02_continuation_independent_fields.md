# Independent Field ownership and CONT6

A Field produced by an independent `FieldProblem` has a scientific owner even
when it has no evolved State owner. Its type carries the canonical identity of
the authored problem, the exact unknown or derived projection, component count,
and FieldSpace. This identity does not include a solve invocation or schema path.
Equal names, shapes, component counts, or meshes do not identify the same owner.

The native type has exactly one ownership alternative: a nonnegative State owner
with an empty provider key, or owner `-1` with a nonempty canonical provider key.
The Field value and its producer support must agree with the declared type. The
core treats the provider key as opaque bytes; it does not parse physical models.
The installed catalogue resolves a complete type and producer support to a real
prototype. An owner-only legacy provider cannot restore an independent Field.

Scientific identity describes admissible values, not a live read permission.
Before structural packing allocates or copies, the context authenticates the
actual source allocation and its lifetime, invocation, and scientific owner.
Before unpacking, it authenticates the exact retained value or registered source
and compares its metadata. A provider descriptor and a matching layout alone
cannot create this authority. Copy publication registers the actual destination.
The producer's historical point is preserved rather than replaced by the
consumer's current time.

## Persistent representation obligations

A Field may also carry an immutable `FieldRepresentationObligation`, independently
of its type, producer support, or face-expression provenance. Its two nonempty
strings record a closed transformation contract and its production origin. They
are descriptive evidence, not permission to apply a transformation. A contract
must include all effective captures: the same expression graph with a different
parameter or provider value is a different closed transformation.

The codec, copies, selected Variant arm, and rematerialization preserve these
bytes. A transformation consumer still needs a native proof tied to the actual
live value and captures. Encoding this metadata does not implement or certify
that consumer. The corresponding rejection guards remain until that proof is
available. Both the obligation and producer support have const data members, so
an ordinary mutable shared-pointer alias cannot change retained provenance.

## Wire and rematerialization contract

CONT6 retains the CONT5 tables and optional shared face-expression table. Before
each Field record it adds the provider string, a Boolean representation-presence
word, and, when present, the contract and origin strings. A present producer
descriptor also includes its provider string. The owner word encodes `-1` only
for this independent-provider alternative. Metadata is charged to the declared
codec or archive workspace budget before retained allocation.

CONT6 is canonical only when at least one reachable Field has a provider or a
representation obligation. Existing State-only images continue to use their
previous encoding. Regular decoding and inert archive decoding enforce the same
framing. Migration compares complete providers and obligations across source
shards, preserves ghost values and original points, and obtains target geometry
from the exact target prototype before creating fresh placement authority.

## Evidence and remaining reception

Source tests cover component and gradient identities, same-name foreign problems,
different unknowns, frozen Python schemas, malformed and truncated CONT6 records,
and CONT5 compatibility. C++ syntax checks cover the real codec and migration
tests. These include rejection before prototype allocation, metadata-preserving
structural copies, and different obligations or providers on source shards.

The C++ 1→2→3→1 case simulates ownership partitions through the real codec and
rematerializer in one process. It is not an MPI RuntimeInstance restart campaign.
Native execution of the new CONT6 tests, public Python retention of independent
Fields, and live Uniform/AMR provider authority must be received together before
this extension is reported as qualified. No numerical or scaling guarantee is
implied by successful serialization.
