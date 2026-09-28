# Contract decisions

## Production integration and versions

The reference runtime and its extension ABI are teaching/reference artifacts.
They are replayed in a separate environment. Production continues using PoPS
qualified handles, expression IR, native component interfaces, FieldView,
Kokkos traversal, MPI execution lanes and AMR continuation. No mesh is unrolled
into generated code and no per-cell Python callback is introduced.

Package 1.1.0 adds API surface. Public API and semantic IR revisions advance to
2 for Program expression DAGs/captures and numerical path bodies. Existing
binary layouts and checkpoint payload fields do not change; exact artifact and
semantic identities still have to agree. No release tag or compatibility with
the reference ABI is asserted.

## Expression evaluation

Physics bodies can index declarations and concrete Program coordinates. A
Program expression retains exact input value identity, components, block,
support and stage information. Newton's candidate is distinct from fixed
captures and its initial seed. The compact DAG is lowered through the common
CSE emitter, not evaluated by a host interpreter. Nonfinite leaves or
intermediates produce collective native rejection before publication, even
when a later minimum/maximum could mask the invalid result.

The common expression DAG now retains lazy `where` branch scopes and explicit
binary64 `rounded` barriers. See [the floating profile](floating_profile.md)
for supported consumers, strict compiler flags and the formal derivative
contract. Arbitrary heterogeneous supports, mixed global unknown products and
all implicit library compositions remain implementation obligations, not new
mathematical limitations.

## Field capability and qualified bind inputs

A state-storage package does not imply a default Poisson source. Its absent
RHS callback is represented by an absent capability, not a callback that only
throws. Uniform CFL stepping requests the default field when a block supplies
its RHS or the caller explicitly configured that field. MPI request schema 2
compares these requirements collectively before branching. An explicit field
request without a provider still fails before Program publication. This changes
field scheduling, not the physical equation; it removes the old fabricated
dependency from transport-only programs.

`bind(params=...)` authenticates qualified authoring handles with the compiled
artifact's BindSchema and only then constructs immutable canonical inputs.
Duplicate aliases, unqualified declarations and foreign authoring capabilities
are rejected. An exact canonical identity remains suitable for a restored
artifact. This makes the public convenience path follow the same ownership
rules as explicit `validated.resolve(handle)`; it does not weaken BindInputs.

## Source-authored reconstruction

`reconstruction.User(body, formal_order=..., name=...)` traces a scalar body
over `sample(integer_offset)` and freezes its live stencil and source identity.
Generated C++ specializes the existing native oriented stencil policy, so the
body executes in Kokkos face loops for each component. The halo derives from
the actual stencil, including both branches of a conditional. There is no
16-cell mathematical restriction: integer offsets, halo depth and envelope
width must fit their native integer representation, and actual resource
availability governs allocation. Declared formal order is an author's claim,
not an order certificate or a TVD/positivity guarantee.

The current integration has a scalar componentwise body and one policy per
compiled model package. Runtime-parameter captures, cross-component bodies,
different methods on occurrences within one block and composed higher-order
path policies are still open implementation obligations. The supplied
mini-runtime is not used to realize any of these paths. Source and installed
native evidence are recorded separately while this tranche is under test.

## Authored nonconservative paths

`SymbolicPath(product, frame=..., quadrature=..., speed=..., bend=None)` retains
the exact physical product. With `j=R-L`, its authored curve is
`Psi(s)=L+s*j+s*(1-s)*K(L,R,s,axis)*j`; `bend` supplies the square matrix K.
This preserves endpoints and a constant path on the diagonal for finite K.
PoPS differentiates the expression with respect to s. The numerical integral is

`B(L)*j + sum_q w_q * (B(Psi(s_q))-B(L)) * Psi'(s_q)`.

The exact constant-matrix part guarantees first-order consistency even for a
quadrature that does not integrate the curved tangent exactly. Nodes and
weights are finite, nodes lie in [0,1], and weights sum to one. Quadrature order
and accuracy on the variable part remain the selected method's obligations.
Two rejected drafts demonstrated why endpoint normalization alone was
insufficient: an arbitrary integral could erase B, and an uncorrected midpoint
rule could change a constant B by an O(j) term.

The speed body declares a bound for the complete `DF+B` along this path.
Finitude/nonnegativity are checked natively; PoPS does not claim to prove an
arbitrary authored spectral bound. A method author must supply that proof or
qualification. Endpoint eigenvalues and density positivity are insufficient.
Runtime parameter captures participate in the same native parameter table as
physical expressions and are not baked into generated defaults.

The existing shared conservative flux, separate nonconservative side terms,
stage trace and synchronized hierarchy continuation are reused. The current
realization is first order in conservative variables with Rusanov dissipation;
higher-order interior path contributions, arbitrary reconstructions, auxiliary
traces and general face solvers remain gaps. The new mechanism is not a full
Fan–Li or HyQMOM scientific qualification.

Uniform path evaluation now uses the same conservative face flux and lateral
nonconservative contributions as the native AMR operator. Its current native
acceptance route requires an authored Courant budget and checks actual incident
face speeds before publication. A FixedDt program without that separate budget
is rejected. Adding a public acceptance budget independent of the step selector
remains an extension obligation; a universal hardcoded CFL value is not assumed.

## Resource and floating-point corrections

PreparedResourceCache keeps its old native object while constructing a
candidate. Allocation and constructor failures are reduced collectively before
publication. A resource constructor using MPI collectives must synchronize its
own internal failure paths. The cache still returns a synchronous reference;
this correction is not an asynchronous resource lease or thread-safe cache.

Limiters retain subnormals and finite representable extrema; invalid inputs
remain nonfinite for the native acceptance layer. WENO rescales only when
required and normalizes weights without overflowing them. These changes do not
grant system positivity, realizability, entropy stability or SSP guarantees.
The isolated arithmetic benchmark measured significant extra cost; end-to-end
performance reception remains open.
