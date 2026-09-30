# Typed global IntegralState capture: bounded T5

The runtime ledger already owns each global IntegralState. The new connection
reads that scalar into a declared Program pointwise expression, then supplies a
POD value to the actual Kokkos cell kernel. It allocates no cell field for the
integral and does not mutate RuntimeParams.

Declare physical dimensions explicitly:

```python
q = P.integral_state("q", initial=0.7, units=PhysicalDimension())
candidate_q = P.integral_value(q, at=U.n.point, scope="candidate")
```

Candidate scope is provisional. It can read the accepted value at the beginning
of a step, or a declared current candidate, but is never labelled a durable
accepted-state observation. The enclosing native transaction owns acceptance.
The common expression has an explicit cell template and the capture must share
its evaluation coordinate. Its output endpoint may be the next state endpoint.
The model's source body remains an exact retained physical occurrence. The
Program visibly authors any composition with the global scalar.

PreparedIntegralCapture has no public numerical constructor or value extractor.
Uniform and AMR contexts authenticate the context owner, exact cache-owned
attempt, complete evaluation point, typed integral identity, canonical physical
dimension bytes and current scalar bits before authorizing the native POD value.
The capture owns a detached POPSEX02 ledger image; changed records, quantities or
consumption keys revoke its provenance even if the scalar is unchanged.
Ledger images are rank-local; ownership can differ. Scalar values, identity,
units and points agree exactly on the prepared MPI lane. Local preparation errors
reach a collective failure vote before the agreement collective or the kernel.
Rollback permanently revokes the attempt; an allocation surviving rollback does
not authorize a new attempt.

Typed declarations use pops.integral.v2 with the canonical dimension SHA-256
embedded in the key and integral_units_v2 in Program semantic data. Unknown units
remain valid for legacy declarations, but cannot be captured by this API.
Legacy declarations retain v1 identities. Default POPSEX01/02 serializers and
their record formats are unchanged. The new class is in the SDK manifest.
No member is added to ProgramRuntimeState, System or either Program context.
The units authenticate the declared scalar type; this tranche does not add
automatic dimensional inference to the existing common expression algebra.

Direct global arguments in Equation operator signatures, characteristic flux
bodies and FieldProblem boundaries are **not realized** by this tranche.
ProgramGlobal.to_cpp refuses those routes explicitly. Candidate scope and
pointwise composition are realization choices, not mathematical limits on a
circuit or on future global input ports.

## Closed normalized witness

The public fixture authors a transport flux F(U)=U and a named physical source
S(U)=-gamma U in one retained balance with separate exact partitions. It
declares the composed feedback reaction q S(U), followed by first-order transport
on [0,1] with Outflow on x and one periodic y cell. All quantities are normalized;
q is explicitly dimensionless. gamma=0.3, q0=0.7, dt=0.01 and U0(x)=1+0.2x are
fixture data, not recognized names or runtime special cases.

The Program uses explicit reaction then transport:

```text
R_i = U_i (1 - gamma q dt)
U_i(next) = R_i - (dt/dx)(R_i - R_(i-1)), R_(-1)=R_0
q(next) = q + dt R_(N-1)
```

The q update consumes the genuine outward FV face integral at x=1 exactly once,
with declared scale -1 reversing the ledger's -div incidence. The User Riemann
provider declares the centered Rusanov expression and its numerical speed
envelope. The original named source is evaluated once; Python computes an
independent recurrence from saved pre-step U and q only for assertions.
Two configurable resolutions, actual selected exterior occurrence/key counts,
reaction inventory, global integral, checkpoint and byte-exact fresh-instance
continuation are checked. An unsafe numerical speed envelope rejects the same
public artifact, restores state/ledger/time, then admits an endpoint-clipped
safe retry.

The C++ fixture checks capture revocation, rank-local point/unit/value/provenance
mutations, foreign context ownership, an actual Kokkos scalar consumer and parent
rollback. A separate C++ fixture retains faces from the real native scalar FV
operator, consumes its nonperiodic right exterior trace once, rejects the parent
after that delivery, then retries and roundtrips the actual ledger checkpoint.
Neither fixture manufactures an exterior numerical value or qualifies a wall
experiment. Root must receive both real runtime paths after rebuilding.

M14's corpus specifies an absorbing wall and a proposed external capacitance,
and W11 requires rollback after wall-current evaluation. It does not supply
enough circuit/sheath equations and data to identify this witness with the full
Cagas experiment. Direct boundary feedback and a coupled capacitance/sheath
acceptance solve remain separate work. Rejection after actual native exterior
consumption is a runtime fixture here; it is not yet an executed scientific receipt.

## Author checks and native reception

Author checks are source Python, generated C++ syntax and the actual native test
translation unit syntax. Root owns installation, MPI runtime execution and saved
scientific receipts. No installed extension result qualifies these new headers
until it is rebuilt against this freeze.

The source-only author batch passed 39 cases: seven new capture/preparation cases,
twelve legacy IntegralState cases and twenty existing common-expression cases.
The nonperiodic public fixture prepared and emitted successfully. Generated
Uniform and AMR translation units passed Dim2 syntax; test_program_runtime.cpp
with both native fixtures passed Dim1 syntax (one external GTest character
conversion warning). No SDK build, installation, JIT or native simulation ran in
the author checkout.
