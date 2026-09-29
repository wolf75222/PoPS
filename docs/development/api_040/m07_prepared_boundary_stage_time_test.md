# Prepared physical-boundary stage-time witness

`SystemInterfaceCoreSession.prepared_analytic_inflow_uses_the_stage_physical_time`
tests the generated legacy spatial and generated coordinated-face Path routes.
Each route installs a model-qualified analytic Dirichlet inflow at the left
face, with authored value `1 + t`, and evaluates its actual prepared RHS at
physical times 0 and 0.25. The valid state is identically 1. The observed
left ghost must therefore be `2(1 + t) - 1 = 1 + 2t`: 1, then 1.5.
The observer runs after the real generated closure and only reads its prepared
state; it does not supply a replacement boundary fill or residual.

The test prepares a boundary transport session for each point and uses the
System's exact block, lane, topology and point. It checks that at least one
rank actually observed a left ghost for each evaluation. It qualifies the
staged physical-time argument separately from the M07 static-boundary lake
case. A passing native run is still required before claiming the witness is
executed; source review alone does not qualify the numerical result.
