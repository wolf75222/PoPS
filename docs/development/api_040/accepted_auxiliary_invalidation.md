# Accepted auxiliary invalidation transition, revision 2

The installed first-field witness on SDK35 (job 733187, ABI11, Source
6f44559ecaee983211d995fbcd004587e078a462) reaches its first accepted step, then
fails accepted checkpoint capture. An external field publication reaches several
exact instances; a read-only instance's derived provider is never consumed. The
runtime previously put every downstream identity in its dirty list, including
this provider with no accepted publication point. The checkpoint correctly rejects
an invalidation without an accepted image.

Revision 2 of the publication transition records downstream invalidations only
for providers with an existing accepted point. The complete dependency graph and
consumer closure remain intact. A never-published provider is already due on its
first exact read, including after restart. Publication and consumer refresh use
the same registry query; checkpoint capture never evaluates a dormant provider.
Accepted stale images remain forced, and rejection preserves their accepted
provenance and dirty identities.

This change extends the C++ registry with
`accepted_dependent_provider_identities`; it adds no model-specific logic, data
members, physical equation, floating-point operation, Python API, or wire field.
ABI11 and the POPSAUX2/3 encodings remain unchanged. HeaderSignature changes and a
new Native build is required. Unknown, input, duplicate, and unpublished dirty
identities remain rejected by the checkpoint codec. Source/Host checks do not
qualify the new installed runtime; job 733187 remains a preserved negative result.
