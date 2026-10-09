# M16 explicit public composition — SOURCE / CPU header preparation

Base: `3f5a55026f39de3429f3bbb552d043a2a65a9d09`. Shared basis is gel `78d9624`.
No Native, MPI, device, ROMEO, science or performance receipt is produced here.

`CartesianMonomialBasis(indices)` records a complete total-degree monomial basis
in any supplied order. It has no model names, species, dimension or degree ceiling.
The immutable descriptor is shared with the independently authored M17 path.
For M16 the rotation primitive has a genuine two-velocity mathematical contract;
spatial rank is independent. The public Program call supplies `basis` and
`components={multi_index: State_component_name}`. Both the basis order and complete
bijection are part of Program identity. Compatibility calls build this explicit
pair in the Python library. The compiler refuses an absent descriptor; it never
infers physical names, a species, or a fifteen-slot layout.

The compiler authenticates that binding against the actual model layout, gathers
into the mathematical primitive's internal degree ordering, and scatters back.
Density, solved first moments, common affine map, Cayley overflow avoidance,
compensated summation, exponential phase refusal, and source-first AMR dependency
rules are retained. An incomplete State, wrong-rank basis, foreign/duplicate
binding or unknown descriptor is refused. This does not assert that arbitrary
independent States can be combined into one native layout.

Explicit binding requires conditional Program IR23. Existing affine programs must
be reauthored; their old IR identities are not claimed identical. Canonical default
Cayley arithmetic and emitted gather/scatter operations retain their prior order.
Unrelated programs retain their existing conditional versions. ROOT owns central
release/SDK integration and genuine rebuild qualification.

The former degree-four ceiling was a demonstrator restriction. The same expansion
now accepts positive degree, with a native signed-int component-cardinality guard
and widened constexpr indexing to avoid integer overflow. Storage is O(d²), work
O(d⁵) with this direct polynomial expansion; compilation and per-cell stack costs
grow with degree. No uniform cost, stability or arbitrary-degree success is
promised: coefficient/moment overflow or nonfinite inputs refuse publication.
The B.1 oblique HyQMOM15 hyperbolicity obstruction remains unchanged. This operation
does not repair it, project onto realizability, or qualify the full M16 campaign.

Reproduction from this private checkout (ENV is read-only):

```sh
env -u PYTHONPATH /Users/romaindespoulain/miniforge3/envs/pops-api040-ir17/bin/python -m pytest -q -o pythonpath='python .' tests/python/unit/time/test_affine_moment_explicit_basis.py tests/python/unit/time/test_affine_moment_update.py tests/python/unit/time/test_affine_moment_exponential_policy.py tests/python/unit/moments/test_api040_m15_m16.py
env -u PYTHONPATH /Users/romaindespoulain/miniforge3/envs/pops-api040-ir17/bin/python -m pytest -q tests/review/test_sol61_affine_monomial_public_header.py
```

Author results: 45 Source PASS (67.02 s), then 2 actual public-header CPU PASS
(2.45 s), binary32 and binary64. Source-origin assertion and `_pops` absence are
explicit. The header probes compile the complete actual public header with its
real dependency, not an extracted/mock implementation. Their independent oracle
integrates a three-atom measure after the rational affine map at degrees 2/4/5;
invalid density leaves the output unchanged. These host checks do not exercise
Kokkos or a compiled PoPS package. Independent structural/order-five review and
ROOT Native qualification remain required.
