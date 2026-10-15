# Continuation Field producer support

A retained Field may be carried opaquely by another AMR level or the macro parent.
Its immutable `ContinuationFieldSupport` describes the producer: program identity,
Program owner, producer level, original point contract, topology epoch and
materialization generation. These bytes describe provenance; they do not grant a
live read permission. The AMR context authenticates the actual native Field storage,
its weak lifetime and allocations, and the active materialization separately.

Historical values keep their original point. Reading an old sample on a valid
spatial support does not require its original time to equal the consumer time.
Replacing an allocation or reading on a different support requires a new explicit
transfer; shape equality does not grant authority. Uniform restoration rejects an
AMR support descriptor unless an explicit transfer has established a Uniform value.

`ContinuationFieldMetadata` carries both the producer support and the independent
Uniform face-expression provenance. Structural packing observes the declared
`ContinuationType`; unpacking passes both metadata components together. Copying one
component never silently discards the other.

`POPSCONT5` adds a producer descriptor before each Field record and an explicit flag
for the shared face-expression table. It is emitted only when a reachable Field
has producer support. CONT1–CONT4 remain readable and keep their existing encoding.
The installed prototype provider receives `(owner, support)` and validates the
producer against the installed context. A legacy owner-only provider refuses a
nonempty descriptor. The collective identity commits the descriptor; source payload
bytes, rank-qualified authority and ordinary layout validation remain mandatory.

During rank rematerialization, the source ownership table is selected by the
producer level from the authenticated complete forest. Target geometry comes from
the actual target producer prototype. Values, ghost cells and the descriptor are
preserved exactly; the target context creates fresh live storage authority. The
archive prototype cache distinguishes the complete descriptor, so equal owners at
different levels cannot alias accidentally.

Source reception includes an independent CONT5 reader and C++ cases for historical
points, exact providers, rejection by legacy providers, metadata-preserving
marshalling and a 1→2→3→1 Field conversion. Syntax-only compilation has passed.
These cases require the official native reception before being reported as executed.
The separate complete RuntimeInstance migration remains under investigation after
SDK012 rejected post-requalification validation on MPI1→MPI2; this document does not
claim that end-to-end receiver has passed.

## Invocation level and typed recovery

`p.level()` produces a native Int64 describing the authenticated current
invocation. Uniform execution returns zero. AMR execution returns the active
hierarchy level, with the same value on all MPI ranks executing that level. The
native context refuses evaluation without a prepared invocation. This value is
introspection, not permission to read a Field; a branch selecting the producer
level still passes through the exact Field access check. Branches that request
incompatible collective barriers on different levels remain invalid.

A typed error may retain a Field from its producer and carry it opaquely to other
levels. Packing validates the live source before copying; unpacking requires the
exact retained instance or an authenticated Store leaf and preserves its metadata.
A structural `Variant` uses the invocation point for its envelope while each
contained Field keeps its exact declared historical point. An `Accepted` leaf
keeps its scope authority; this envelope rule cannot make foreign scopes valid.

The source015 receivers `test_api_v02_field_error_capsule_native.py` and
`test_api_v02_independent_field_continuation_native.py` exercise this contract,
including foreign-level reads, complete rollback, and a retained old observation
consumed after restart. Their presence is not a native pass receipt.
