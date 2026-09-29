# M05: closed periodic viscous shear reduction

The corpus closes only the scalar subproblem
\(u_t=\nu u_{xx}\), \(\nu=0.03\), \(u(x,0)=\sin(2\pi x)\), periodic
\(x\in[0,1)\), N=32,64,128, and T=0.1. The preserved handoff's
`reference/PoPS_API_v0.4.0/legacy/v0.2.0/tests/run_revision.py`, H02,
tests the semidiscrete sign on a random periodic vector. The current public
case uses `Model.diffusive_flux` and `Diffusion` with \(+\operatorname{div}
(\nu\nabla u)\); it uses neither a Hall/coupled-gradient provider nor a
model-specific C++ operation.

`api040_m05_shear_oracle.py` is independent of PoPS. It computes exact
cell means both by sinc and by integrating sine, the continuous amplitude
\(e^{-\nu(2\pi)^2t}\), the finite-volume symbol
\(\lambda_h=4N^2\sin^2(\pi/N)\), its semidiscrete amplitude
\(e^{-\nu\lambda_ht}\), and the Forward Euler amplitude
\((1-\nu\lambda_h\Delta t)^N\). The fixed schedule is N steps of
\(\Delta t=0.1/N\). Its largest authored incident-face frequency fraction is
0.768 at N=128, below the predeclared 0.9 bound. This time choice does not
make the continuous error monotone between every adjacent grid: spatial and
temporal errors can cancel, so no order is inferred from three errors.

For *any* periodic cell vector, the independent stencil checks
\[
  h\sum_i u_i\nu\Delta_hu_i
     =-\frac{\nu}{h}\sum_i(u_{i+1}-u_i)^2\le0.
\]
Forward Euler has an additional nonnegative time-discretization term:
\[
 E(u^{n+1})-E(u^n)
 =\Delta t\,h\sum_i u_i\nu\Delta_hu_i
  +\tfrac12\Delta t^2h\sum_i(\nu\Delta_hu_i)^2,
 \qquad E=\tfrac12h\sum_i u_i^2.
\]
The witness tests both identities separately and requires decreasing native
energy; it does **not** call a Forward Euler energy drop equal to the
semidiscrete work.

The installed reception saves and reopens `initial`, `before_last`, and
`final` native states for every grid. It splits the run into N−1 and one
accepted step so the runtime's actual last-step accepted-exchange ledger can
be compared with the native state increment. The ledger is currently read
through the diagnostic `_program_exchange_records()` API: two owned-cell
incidences per cell, one physical occurrence and one evaluation context,
with each oriented face flux independently checked against
\(\nu N(u_{i+1}-u_i)\). The sum of integrated face amounts must equal each
saved cell change times its width, and its global sum must vanish. All MPI
ranks participate in bind/run/gather/status; `BindArray` is supplied through
`initial_values` keyed by the exact block-qualified subject (no parallel
legacy `initial_state` authority). Only rank zero writes NPZ and
receipt, then broadcasts the verdict. Package prefix, native file hash,
artifact identity/ABI, execution context and each NPZ hash enter the receipt.
All Python/NumPy authoring, oracle and installed-package preflight decisions
are agreed by the bootstrap world before compilation or bind; rank-local
errors cannot let a peer enter the next native collective alone. The runtime
test must be run through `docs/development/api_040/run_installed_checks.py`
with its explicit `--test` path and `env -u PYTHONPATH`, after the matching
Dim1 wheel is installed. Source-only pytest results do not prove bind or run.

Criteria were set before native execution: initial means 2e−14; state versus
the independent Forward Euler oracle 5e−12; state versus continuous cell
means 2e−4; mass 5e−13; ledger flux 2e−12; cell change 5e−13;
semidiscrete work and Forward Euler energy formula 5e−13; time 3e−14.
Source/math tests and a collectable native test are supplied. Neither source
emission nor the inherited H02 unit calculation is a native M05 receipt.

This reduction does not qualify compressible Navier–Stokes, variable stress,
thermal conduction, thermal-energy balance, or Couette walls. Those require
their own constitutive inputs, wall data and joint stress/heat-flux accounting;
no thermal flux is silently substituted into this scalar equation. A negative
viscosity is refused by the selected diffusion method, while signed values
of the shear component are physically valid.
