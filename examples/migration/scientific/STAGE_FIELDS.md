# Three stages and two screened fields

The scientific library `api040_stage_fields_library.py` declares
d_t=0, (-Delta+3)phi=q+d, (-Delta+5)psi=q-3d/4 and
q_t=.15Delta q-.8q+.2phi+.4psi. The periodic rectangle is2x3, the grid16x12,
and dt=1e-4. Initial arrays are exact cell means of a product cosine with distinct
nonuniform receiver and donor amplitudes.

`api040_three_stage_screened_fields.py` follows physics, spatial resolution,
the public SSPRK3 composition and the normal compile/bind/run pipeline in order.
The reusable library is the single authoring graph. There is no second technical
assembly or model-specific compiler implementation.

Each RHS solves and consumes one joint two-unknown FieldProblem, publishes both
declared components and reads the publication at its actual receiver stage. The
foreign donor is committed unchanged. Stage coordinates are0,1,1/2: the resident
last writer is the final execution stage at1/2, not the maximum-time stage or a
fresh solve at the accepted endpoint. Three State histories and six observed
Field histories expose the actual stages.

The runtime test has representative, identity-rename and changed-decay cases.
Its scalar/FFT oracle is a byte-exact copy authored separately from this physics
and from the compiler. The original pre-Native budget JSON is unchanged. The
changed-decay budget uses the same independent formula before execution; it is
not fitted to observed errors. Native test timing600s is a conservative catalog
estimate for the whole three-case file, not a measured CI duration. Dimension2
is declared explicitly and follows the actual geometry.

This package is prepared only. Root must review the generic stage-field support,
complete its official build and receive one representative before variants or
separate failure injection. Author-written tests using a non-author oracle do
not themselves constitute independent reception. Histories and checkpoints
retain real arrays; they are never replaced by calculated outputs.

Successful CG iteration reports and a persistent publication-stage-clock getter
are not currently provided by this fixture. Empty solver diagnostics are not
CG evidence. Saved-field residuals are checked independently; history sample
windows/dt are not presented as measured publication coordinates. This is a
world1, one-step Uniform witness, without AMR, MPI2, GPU or convergence claims.
