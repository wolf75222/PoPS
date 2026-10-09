# Principal numerical policy ownership - implementation clarification 1

The C11 principal group owns the joint physical flux, its authored wave-speed
bound, reconstruction and numerical face policies. Its constituent native blocks
provide state storage, geometry and halo access. A constituent block is not an
independent finite-volume discretization of one row of the coupled system.

Installed reception of source 83b2b12 exposed seven compilation failures: all six
principal AMR witnesses and the Primitive/User widest-halo witness. The loader
passed the row's retained User reconstruction/face policy to the native block
builder even though the resolved model was `program_only_storage`. This selected
a finite-volume builder and attempted to instantiate a wave-speed function for
that isolated row. Such a function was neither declared nor mathematically implied
by the bound on the complete principal system.

The loader now constructs a storage-only block without standalone User policy
arguments when the resolved carrier is Program-owned. The principal helper still
emits and instantiates each authored policy with its exact row, complete sampled
state, parameter context and common halo envelope. The declarations and source
identities remain part of the model/Program identities. Ordinary standalone User
finite-volume blocks keep their existing builders. No row wave speed, zero flux
or numerical fallback is invented.

This is a correction of native realization ownership. It changes neither the
public specification version, native ABI nor the serialized spatial policy
schema. The existing installed runtime tests are the numerical and compilation
regressions; source emission checks alone do not establish acceptance.

Original failed receipt (parent workspace):
`outputs/installed-83b2b12-critical-pde/pytest.xml`. Twelve other runtime tests
passed there. Two additional failures in that receipt came from a C17 test calling
diagnostics on the wrong facade before executing a step; that fixture was corrected
separately in 6ce6782.
