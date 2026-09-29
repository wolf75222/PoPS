# M03 conservation measurement, revision 2

The original M03 stiffened-EOS run is retained as a failed reception:
`m03-stiffened-renamed-openmp1`, native SHA
`e1070cb63a8e733b0a6032ac4d2b8ad003756668c100e057e5f8edd5b9c01280`.
On N=100, t=.15, density L1 was .03114694, but the comparison against unchanged
far-field fluxes gave mass/energy/momentum defects of 3.75e-6/4.20e-5/7.94e-6.
The finite-volume tails reach the Outflow boundaries before the exact Riemann
fronts. Assuming that the numerical boundary states remain at their initial
values is therefore an invalid discrete conservation oracle.

This revision changes the measurement, not the equation, EOS, domain, boundary
condition, mesh, final time, CFL or numerical method. All tolerances remain
unchanged. The earlier failed criterion is not retrospectively labelled passed.
The ideal-EOS run happened to keep boundary disturbances below those tolerances;
that does not establish the old assumption for a second EOS.

For SSPRK2 with the existing first-order Rusanov faces,

`U1 = Un + dt L(Un)` and `Unext = (Un + U1 + dt L(U1))/2`.

Summing the conservative residual cancels interior faces. The change in the
domain integral must equal `dt/2` times the sum of the incoming boundary fluxes
at Un and U1. At an Outflow face the ghost state equals its adjacent cell, so
the numerical face flux there equals that cell's physical flux. Computing U1
on the two boundary columns requires their adjacent columns and the periodic
transverse neighbors; it does not require a native residual or a storage delta.

The revised example uses public `ScientificOutput`, `NPZ(ROOT)` and the
accepted-step schedule to save every actual state. It runs the ordinary adaptive
controller to the prescribed final time without inventing a smaller step or
using private step APIs. After the run it authenticates and reopens the series,
checks consecutive macro-steps and clocks, saves the boundary bands and actual
step durations, then recomputes the boundary budget from those saved bytes with
an independent NumPy implementation. Its first-stage prediction includes the
transverse Rusanov divergence even when the exact solution is y-invariant.

The same conservation tolerances apply both cumulatively and to every accepted
step. The difference from the exact far-field flux remains a separate recorded
boundary-discretization error. Density error, positivity and convergence still
use the independent Riemann reference. There is no state clipping and no
comparison of two quantities inferred from the same final storage difference.

The receipt schema advances to 2 and names this boundary-budget revision. The
seven pure oracle tests include a full two-dimensional finite-volume assembly
with transverse variations and both EOS choices; they discriminate the former
constant-flux assumption. Astra independently reviewed the quadrature argument.
Actual native acceptance is recorded separately by `run_scientific_checks.py`.
The extra accepted-state output is a diagnostic campaign configuration, not a
performance comparison with the earlier output-free run.
