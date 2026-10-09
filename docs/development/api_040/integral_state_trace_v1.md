# Persistent integrals from accepted external traces

`Program.integral_state(name, initial=q0)` declares a Program-owned, globally replicated
binary64 scalar. `Program.accept_external_trace(q, rate=rhs, axis=..., side=...,
component=..., scale=...)` transfers the **accepted** face amount of one resolved
conservative RHS evaluation into that scalar. The public read is
`runtime.integral_state(q)`. This is a persistent state, not a diagnostic overwrite.
The author chooses `initial`, support, component and signed conversion `scale`; PoPS
does not infer a circuit or surface charge from a model name.

The selected `rate` must identify one exact conservative face occurrence: a
finite-volume `-div(flux)` or a prepared constitutive `div(diffusion)` in its
accepted affine quadrature. Its native face carrier supplies every active face
amount with its actual geometric measure, orientation and temporal weight.
Prepared diffusion retains its evaluated flux density; transport retains spatial
face integrals. Diffusion's cell incidence is negative on the lower side and
positive on the upper side, the opposite of negative-divergence transport.
Nonconservative source occurrences never contribute to an external trace.
An inseparable fitted drift/diffusion flux is one joint occurrence; a Rate with
both independent transport and diffusion occurrences is ambiguous to this selector
and is refused rather than choosing a current implicitly.
An external trace is a face on the nonperiodic physical boundary of the domain;
interior and periodic faces are excluded. The selector also carries the exact
source evaluation identity. Thus two SSPRK evaluations of the same physical
occurrence cannot consume each other's faces. On AMR, coarse covered cells are
excluded by the composite active mask, and the scalar transfer runs once after
all levels, subcycles and synchronization have staged their records. No matching
unconsumed external face in the whole selected hierarchy rejects the attempt.
The numerical producer remains responsible for covering every active boundary
face; a nonempty ledger alone is not a geometric completeness proof.
One exact evaluation/support selector has one persistent-state consumer in this
version. A second declaration targeting another scalar is rejected at authoring;
an author can derive additional scalars from a later explicit algebraic extension.

For a right-boundary flux into an external capacitor, accepted cell incidence is
negative. With `C=1`, `q0=0.7`, flux `2`, face measure `1`, and two accepted
weights `0.1` and `0.2`, `scale=-1/C` gives `q=0.7+0.2+0.4=1.3`.
`scale` is an authored unit/sign conversion, not a numerical stability claim.
The accepted scalar, consumed record keys, cell states, histories, clocks and
exchange mailbox are all copied by the same native attempt snapshots. A child
acceptance remains provisional until the parent accepts. Parent rejection restores
the original scalar and permits the same physical trace on a later retry; a
second consumption within one attempt fails before scalar publication.

The declaration is installed before the first step, so `runtime.integral_state(q)`
returns `q0` immediately after bind. The restart image remains `POPSEX01` for
plans without integral states. `POPSEX02` carries exact state identities, initial
and current values, face support, source evaluation identities and consumed keys.
An integral-bearing plan refuses an old image that lacks these declarations.
Restore compares declarations with the installed Program and current scalar
values across ranks before swap; exchange records may legitimately differ by rank.

Version boundaries in this change:

| Contract | Old → new | Reason |
| --- | --- | --- |
| Public API | 2 → 3 | New author/read surfaces for persistent scalar states. |
| Semantic IR | 2 → 3 | Exact declarations and transfer selectors enter Program identity. |
| Program serialized IR | 4 → 5 | Optional integral declarations and transfer selectors. |
| Native release ABI | 3 → 4 | New System/AMR and ProgramContext methods. |
| Generated System package ABI | 6 → 7 | Generated providers call the new native methods. |
| Accepted exchange image | 01 → 02 conditionally | Durable scalar and consumption payload. |

Module manifest schema 10, artifact snapshot schema 6, AND9 and the Python AMR
checkpoint envelope remain separate and unchanged. A package compiled against
the former ABI must be rebuilt before native execution. Source-level tests cover
real `validate → resolve → emit` for two SSPRK stages; the native transaction and
restart tests require the corresponding rebuilt C++ targets and MPI reception.

The prepared-diffusion extension uses **Program serialized IR 6** only when an
external-trace selector names a `diffusive_rhs` evaluation. Transport-only Programs
retain version 5 and their semantic identities. It reuses existing POPSEX02 fields,
native integral/consumption APIs and the exact constitutive occurrence identities;
no restart image or native method signature changes. The changed template header
and generated package still require the matching rebuilt SDK/native installation.
Diffusion's existing geometry, realization and stability refusals remain in force,
including unsupported active embedded geometry and unproved explicit AMR tensor
or coupled-gradient routes. The accepted face ledger intersects EB activity and
finest-owner coverage when those carriers are provided by an admissible realization.
