# RB-04 — Shu–Osher

Mach-3 shock interacting with a sinusoidal density wave. This increment
is the FLASH/literature IC plus a leftover post-shock density probe.
There is no closed-form evolved state and no uniform-fine reference run.

| Field | Content |
|---|---|
| Identifier | `RB-04` |
| `verification_kind` | `robustness` |
| `evidence_status` | `required` |
| Equations | 1-d gamma-law Euler, primitives \(W=(\rho,u,p)\). Conserved \((\rho,\rho u,E)\). No sources. \(\gamma=1.4\). |
| Oracle | IC only (this increment). Shu & Osher, *J. Comput. Phys.* 83, 32–78 (1989), Example 8; FLASH Hydro `ShuOsher`. Left \((\rho,u,p)=(3.857143,2.629369,10.33333)\), right \(\rho=1+0.2\sin(5x)\), \(u=0\), \(p=1\). Diaphragm \(x_0=-4\) on \([-5,5]\) (the usual literature cut). The image \(x=-0.8\) on \([-1,1]\) is a linear map of \(x_0\) and is not stored, so \(\sin(5x)\) is unmodified. A unit map of the same IC is \(x_{\mathrm{unit}}=(x+5)/10\), shock at \(0.1\). Plan oracle (uniform-fine reference) is not in this increment. `exact.py` does not read PoPS output. |
| Domain and boundaries | 1-d interval \([-5,5]\). Physical tube is transmissive; the authored in-memory layout uses the periodic helper and is not run. |
| Parameters | \(\gamma=1.4\). FLASH/literature left state as printed (not recomputed from Rankine–Hugoniot). Amplitude \(0.2\), wavenumber \(5\). Usual final time \(t=1.8\) is documented only; this increment does not evolve. Dimensionless. |
| Native dimensions | `POPS_NATIVE_DIM=1`. In-memory path does not load a native artifact. |
| Required capabilities | Cartesian 1-d, uniform, KokkosSerial, MPI off. HLLC/Rusanov + MUSCL when a native series exists. |
| Configurations | Single resolution \(n=32\) for the in-memory report. Authored scheme: Rusanov, MUSCL/VanLeer, SSPRK2, AdaptiveCFL. No reference-fine series. |
| Diagnostics | Left/right ICs. Right-state sine amplitude \(0.2\). Positivity of \(\rho,p\). `post_shock_density_probe` leftover observation (density for \(x<x_0\); not a gate). |
| Thresholds | Empty `orders` with reason containing `reference-fine not in this increment`. No spatial-order gate. |
| Proves | Documented FLASH/literature Shu–Osher IC (left/right states; right sine amplitude \(0.2\); positivity); leftover probe hook; report renderer accepts empty orders justified by a missing reference-fine. |
| Does not prove | Observed spatial/temporal order, shock–entropy-wave interaction quality, limiter/WENO behaviour, AMR tagging, Poisson, coupling, MPI, HLLC vs Rusanov parity, transmissive native BCs, agreement with a uniform-fine or external-code reference. |
| Resources | `pr.nodes = 1`. `two_node.nodes = [1, 2]`. No ranks, GPUs, or wall-time claim. |
| Provenance | Campaign `repository_sha` is `git rev-parse HEAD`. Native compiler/Kokkos/MPI/Slurm not recorded on the in-memory path. |
