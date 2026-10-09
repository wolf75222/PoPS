# HYQMOM15 and isotropic BGK collision checks

These linear, module-scope examples exercise the RIEMOM2D collision law in the
fifteen raw moments through degree four. They are separate from the collisionless
[Hoffart conducting-disk case](../diocotron/README.md).

With density `rho`, mean velocity `u` and central covariance `C`, the isotropic
temperature is `Theta=(C20+C02)/2`. The collision rate is
`nu=2*rho*sqrt(Theta)/Kn`, and the source is `nu*(M_Maxwellian-M)`.
Mass and momentum are conserved; the two diagonal stress sources are exact
opposites, preserving their trace. No covariance-matching replacement, variance
floor, projection or complex-eigenvalue truncation is used.

| Script | Physical and numerical configuration |
|---|---|
| `01_homogeneous_bgk15.py` | Positive two-Gaussian mixture; density1.5, mean(1/4,-1/8), covariance[[5,.75],[.75,3]]. Uniform4×4 storage, source-only SSPRK2, Kn=.01/.1/1, 40/80/160 steps to `t=Kn/3`. The mixture is constructed explicitly in the script; `analytic-inputs.json` is an independent exact-rational check. |
| `02_x_shock_bgk15.py` | Exact HYQMOM15 Cartesian transport, full15 Jacobian speeds, HLL with piecewise-constant reconstruction. Uniform64×4 cells on[-.5,.5]×[0,1], outflow x/periodic y. Left/right densities1/.1, zero mean and unit temperature; Kn=.1. Forward Euler to t=.01, CFL.1, maxdt1e-4, strict source-frequency bound.05;201 actual saved states. |

Run these bounded diagnostics with **one MPI rank** and two Kokkos/OpenMP threads
in an installed Dim=2 MPI PoPS environment. Their output directories must be new:

```bash
env -u PYTHONPATH OMP_NUM_THREADS=2 POPS_THREADS=2 mpiexec -n 1 \
  python docs/tutorials/hyqmom_bgk/01_homogeneous_bgk15.py \
  --kn 0.1 --steps 80 --output output/bgk-homogeneous-kn01-n80

env -u PYTHONPATH OMP_NUM_THREADS=2 POPS_THREADS=2 mpiexec -n 1 \
  python docs/tutorials/hyqmom_bgk/02_x_shock_bgk15.py \
  --output output/bgk-x-shock-kn01
```

The homogeneous example stores initial/final moments and checks the exact
continuum solution and the discrete SSPRK2 amplification. Repeating it at40,
80 and160 steps measures temporal convergence; endpoints do not constitute a
saved time series. This source-only test does not evaluate transport eigenvalues.

The spatial example retains every actual state and RunReport. It requires one
accepted step with zero rejections per requested interval before evaluating its
explicit-Euler boundary-balance check. Failures and partial states are retained.
The boundary flux is an independently evaluated physical outflow expression;
the public interface used here does not expose native face/speed buffers.
Positive density/covariance checks do not certify the complete degree-four
moment cone or hyperbolicity in arbitrary oblique directions. See the
[Maxwellian and correlated-state analysis](../diocotron/HYQMOM15_LIMITATION.md).
