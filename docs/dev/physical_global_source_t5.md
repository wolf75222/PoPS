# Physical global source inputs (T5)

This tranche separates the physical expression from its temporal capture. A
`Module` or `Model` declares an immutable scalar `global_quantity`, with explicit
`PhysicalDimension` units. It is a declaration input with global scope, not a
mutable RuntimeParam, a cell field, or an initial circuit value. `Case` projects
that declaration through its normal authenticated `block[quantity]` registry.

The physical declarations precede the method:

```python
Q = model.global_quantity("circuit_quantity", units=PhysicalDimension())
S = model.source("reaction", on=U, value=(-gamma * Q * U[0],))
balance = model.rate("balance", equation=ddt(U) == -div(F) + S)
source_rate = balance.select(S)
# Case assembly and source StateStorage/FV providers are declared here.

P = pops.Program("feedback")
Ut = P.state(block[U])
q = P.integral_state("q", initial=q0, units=PhysicalDimension())
captured = P.integral_value(q, at=Ut.n.point, scope="candidate")
source = P.evaluate_source(source_rate, Ut.n,
                          global_inputs={block[Q]: captured})
reacted = P.value("reacted", (Ut.n[0] + P.dt * source[0],), at=Ut.n.point)
```

The Program supplies the point and candidate scope; the source's original
expression owns `-gamma*Q*U`. The exact selected source occurrences and their
coefficients remain in the balance IR. Multiple source occurrences receive only
their declared global reads, after the union of required ports is authenticated.
Source evaluation rejects missing or unused bindings, foreign/unissued ports,
different units, and a capture at another exact point. The captured SSA value is
an explicit input of the source node. Code generation rechecks canonical SSA
metadata, including replaced values, before producing a kernel.

The existing native `PreparedIntegralCapture` authorizes the value immediately
before the genuine source Kokkos kernel. It checks the complete evaluation point,
exact execution/attempt lease, owner, canonical units, value bits, and the
rank-local ledger provenance. Collective exception votes and exact MPI agreement
precede consumption. Its integral identity includes the physical body and port
binding through the Program semantic identity. Only the authorized `Real` POD is
captured by the cell kernel; no circuit field is allocated. Accepted state,
integral ledger, rollback, retry and checkpoint remain owned by ProgramRuntime.

The feature extends ModuleManifest to version 11 only when global ports exist,
with a strict `global_quantities` table (declaration identity, version, scope and
canonical units). Unextended manifests remain version 10 with the same JSON
bytes and hashes. No native ABI, persistent carrier, or POPSEX wire change is
introduced. The established typed integral identity remains v2.

The implemented realization is a named source-only Equation consumer on the
existing StateStorage source route; it supports generic pointwise source Expr
bodies and component orderings. A physical global leaf has no unbound native
fallback. Default native sources, implicit source solves, FieldProblem and flux
bodies do not yet have global binding providers and refuse before an unbound
kernel can run. Those missing providers are extensions of the input contract,
not mathematical limits on global quantities or temporal schemes.

The closed normalized fixture uses the declared reaction `S=-gamma*q*U` and
transport `F=U`. One actual accepted exterior FV trace updates the persistent
integral once. Two resolutions compare saved state with an independent reaction
and transport balance, check real ledger occurrences/consumed keys, reject an
unsafe attempt and retry safely, and compare fresh-runtime checkpoint replay
byte exactly. It is a mechanism witness: the Cagas/sheath/paroi M14 physical
data and circuit closure are not supplied by this tranche.

Author evidence (private source checkout, Python source path printed): 26 source
tests passed, including the operator-first Module, two distinct physical sources,
and legacy Module manifest/hash tests. The emitted Uniform and AMR
C++ translation units, dimension 2 with three permuted components and Outflow
borders, passed `clang++ -fsyntax-only`. Three legacy manifests (unknown units,
dimensionless units, one/two components) were compared byte-for-byte with the
version-10 class and hash function extracted from base `8eabb8e2`; all agreed.
The public runtime fixture is provided for central native reception; no native
execution, SDK build, installation, or JIT was performed by this author.

The follow-up authentication patch repeats registry issuance at emission, after
SSA resealing: a same-owner metadata-equal clone is rejected before conversion
to the kernel POD. Global declaration version 1 requires an exact integer;
float, boolean and string substitutes are rejected. Its 21 physical-source
source tests include a separate valid emission before each mutated binding.
The units image also requires exact integer numerators and denominators before
decoding: JSON booleans and floats cannot impersonate canonical exponents. The
follow-up 29-test source/manifest/hash suite passes with six such injections.
