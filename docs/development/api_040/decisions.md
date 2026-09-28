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

The currently implemented pointwise subset does not establish general lazy
`where`, arbitrary heterogeneous supports, floating `rounded` barriers, mixed
global unknown products or all implicit library compositions. Those are open
implementation obligations, not new mathematical limitations.

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
