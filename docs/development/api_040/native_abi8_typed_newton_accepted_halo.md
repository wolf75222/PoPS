# Native ABI8: typed Newton convergence and accepted halo preparation

ABI8 integrates the changed `FieldNewtonOptions` layout and native typed-policy
setters with the accepted-halo execution and Field RHS read-effect contracts.
Python constants and C++ headers are regenerated from
`schemas/release_contract.v2.json`; a fresh native rebuild and authenticated
installed-header reception are required. The earlier ABI6/7 artifacts are not
qualified by source tests of ABI8.

An explicitly typed `Newton` criterion is authenticated independently from its
seven numerical controls. `Relative(r)` selects `r * norm(F_initial)`;
`Relative(r, floor=AbsoluteFloor(a))` selects
`max(a, r * norm(F_initial))`; `Absolute(a)` selects `a`. The residual and
reference are the full Original equation, with the existing Uniform/AMR norms.
The computed typed cutoff must be finite representable `Real`. Failure is voted
before admission or the next Krylov collective. Every execution-lane rank first
compares the relevant criterion bytes exactly, including the legacy tolerance,
then participates in the cutoff vote. This adds communication to the legacy
engine; its tolerance arithmetic, controls, Program IR and emitted Uniform C++
remain unchanged. Extreme legacy cutoff overflow is an existing compatibility
limit, distinct from the corrected typed path.

Typed prepared nonlinear plans use schema2 and the authenticated prepared
spatial Newton identity v4. Program IR22 is selected only when typed spatial
convergence is present. Legacy prepared plans and Program version selection
retain their existing contracts. Typed implicit accumulation stages still lack
an authenticated policy relay and explicitly refuse this unsupported route.

`AcceptedHaloPreparation(cells=...)` is an explicit execution effect. Ordinary
AMR execution keeps schema2 and accepted-state8. The opt-in execution uses
schema3, accepted-halo request1 and accepted-state9. It stages the complete
candidate hierarchy, obtains exact engine clock windows, prepares native
transfers and boundaries, and restores canonical accepted storage before any Q
publication. Topology preparation uses the compiled primary-clock identity and
the actual declared clock partitions, including final remainder intervals.

Field boundary dependencies require genuine temporary solved producers and a
compiler-certified `StateReadExtent@1` for direct RHS cell reads. The required
Field DAG is resolved before staging. Unknown effects, positive-radius state
reads, uncertified InputAux and dependency cycles require a stronger declared
method and are refused on this opt-in route. Classification uses exact generic
AST node types and transitively expanded primitives, without physical model
names or formula recognition. It does not infer effects from opaque callbacks.

Initial periodic preparation supports the actual zero-duration bootstrap point.
Initial typed GhostBC ABI1 requires a positive authenticated interval; an
initial-interval authority is still missing and is refused explicitly. No
synthetic timestep or cached Field setter substitutes for that authority.
These remaining routes are contract-extension obligations, not model/grid caps.

Uniform checkpoint payload8, AMR payload12 and the POPSCAR1 carrier codec remain
unchanged. Existing scientific receipts retain their original SDK/ABI and scope.
Independent Source and host reviews are recorded in the integrated Newton and
accepted-halo review notes. This ABI document does not assert a successful
native build, MPI run, GPU run or scientific reception.
