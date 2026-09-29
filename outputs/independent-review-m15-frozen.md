# Independent review: M15 axial B.1 native preparation

Reviewed author commits `610c39c` and `ae3e24e` in `PoPS-principal-group`, against the
independent pre-native oracle fixed at `30e7d47`. This is the five-moment,
order-four axial Appendix B.1 variant in genuine one-dimensional space. It is
not qualification of the requested Fox–Laurent multi-order recurrence.

The scientific body retains the full five-component B.1 closure, signed
Jacobian speeds, FirstOrder/HLL and ForwardEuler. The native schedule is fixed
at `dt=1/(100N)`, `t=.02`, `N=32,64,128`, with canonical and genuinely permuted
components. Initial cells are analytic averages of a positive, spatially
varying two-Gaussian mixture. The B.1 fifth flux differs from a Gaussian
substitution by about `0.234` in max norm on these states. The literal
`rho>0`, `H2>0`, `H3>=0` tests and no-projection rule remain explicit.

Two separately authored NumPy oracles were compared before native execution:

| N | Initial max difference | Final FE/HLL max difference |
|---:|---:|---:|
| 32 | 0 | 1.11e-16 |
| 64 | 0 | 2.22e-16 |
| 128 | 0 | 3.33e-16 |

The independent oracle's three pure tests passed. I also reran the author's
seven source/math reception tests against the source checkout; all passed in
3.61 s. Neither result is an installed-native qualification. In particular,
the Dim=1 native artifact, MPI, state NPZ files and end-to-end receipt remain
for root's central run.

Two issues were corrected during review of the preparation: the negative
density fixture originally also violated `H3`; it now isolates density with
`(-1,0,-1,0,0)`. The negative runtime test now checks that the same artifact
advances valid data and accepts only the actual native recovery/face-admission
diagnostic, with collective rank agreement before inspecting a returned state.

At `610c39c`, writing the failure receipt inside rank zero's exception handler
could itself raise and skip the ensuing broadcast, leaving other MPI ranks
waiting. Follow-up `ae3e24e` catches that secondary write failure, preserves
the original error, and broadcasts it. It also wraps final receipt construction,
hash reading and writing before a final status broadcast. This repairs the
identified source-level collective failure seam. It is **not** a substitute for
an actual timeout-bounded two-rank run; MPI behavior and all native numerical
criteria remain unverified here.
