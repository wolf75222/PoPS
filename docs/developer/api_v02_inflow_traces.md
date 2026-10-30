# Prescribed inflow traces (SDK010 candidate)

The SDK008 multiblock receipt showed an extra mass loss of 0.00017440000000010797. The first-order Riemann solver consumed the centered ghost extension `2*g-q` as an exterior trace. For g=0, a=.002, q=.1, h=1, dx=1/64, SSPRK2 independently predicts the spurious loss `a*q*(1-a*h/dx)=.0001744`, and first fine cell `.0776768`, matching the native observation.

The corrected contract uses the authored Inflow value as the exterior Riemann trace at the actual face position and stage time. It preserves the ghost extension needed by other stencils and diffusion. The selected model, providers, reconstruction of the interior trace, numerical flux, admissibility, face measure and stability validation remain authoritative. Failed face evaluation still cannot publish a candidate. The resulting flux is shared by divergence and AMR reflux.

M14 formerly adapted its oracle to the same ghost-as-trace defect (historical sections3–7 audit). Its new oracle uses the actual reservoir state and explicitly distinguishes the former mirrored behavior. The historical receipt is preserved as evidence of the former behavior, not reused to qualify this change. M14 native full-state/circuit/rejection, multiblock affine/reflux/restart, analytic-stage and boundary C++ receipts must be rerun on the corrected SDK.

Source work is prepared in an overlay. No native qualification is claimed until the SDK rebuild and receipts complete.

## Chosen by name (MG8, ROM-101)

This prescribed trace is a realization of its own: `Inflow(state=..., value=g, trace="prescribed")`. Its face keeps the fixed-state law; the lowering sets a per-face flag, `prescribed_trace`, which the native boundary reads (`PreparedHyperbolicBoundary::with_prescribed_trace_faces`) without naming the boundary family (CF2). `Inflow(state=..., value=g)` is master's realization again, the mirrored ghost `2*g-u` of D9 (`dirichlet_ghost`), whose reconstruction is also the exterior trace of the numerical flux: master's tutorials keep their bits only with it (spec 0.3, MG8; adjudication, section 19). M14 and the two-block receipt name the prescribed trace, so their oracles and assertions are unchanged. The spelling is provisional and goes to N15 (ROM-98).
