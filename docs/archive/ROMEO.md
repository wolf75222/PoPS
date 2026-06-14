# Runs ROMEO (GH200 + EPYC)

Reproduction of the diocotron growth rate (Hoffart, arXiv:2510.11808, mode 4, analytic
target 0.911) on the ROMEO supercomputer (URCA). SLURM job log:
[`romeo/HERO_RESULTS.md`](../romeo/HERO_RESULTS.md). Full account:
[`tutorials/10_diocotron_reproduction.md`](../tutorials/10_diocotron_reproduction.md).

## Rate convergence (WENO5-Z + SSPRK3, EPYC)

| Figure | Source | Conclusion |
|---|---|---|
| ![convergence WENO5](romeo_highorder_convergence.png) | job 613961, WENO5-Z + SSPRK3, modes 3/4/5 x eff 256/512/1024 | overshoot ~+8 % **flat in resolution**: geometric limit (cartesian boundary), not numerical |
| ![mode 4 growth](romeo_growth_mode4.png) | real `ring_amp.csv`, mode 4 | eff 512 and 1024 **overlapping**: rate converged in effective resolution |
| ![AMR efficiency](romeo_amr_efficiency.png) | job 613945, AMR vs uniform column | same rate for **~40 % of the cells** (multi-level VanLeer AMR) |

The AMR column with multi-level Poisson equals the uniform one at equal effective resolution
for ~41-44 % of the cells (normalized mode 4 rate `0.42`/`0.526`/`0.563`/`0.592` at eff
192/256/320/448). The analytic target `0.911` is not reached: the overshoot is flat in
resolution, which points to the cartesian staircase geometry (circular ring and wall on a
square grid), not numerical diffusion. Next step: cut-cell or polar coordinates.

## GPU: bit-identical CPU/GPU

On 1 GH200 (`armgpu`, Kokkos/CUDA 12.6), the full coupled step + the multi-patch AMR
run on GPU:

- checksum `diocotron_amr_kokkos` = **4394594.404318** exactly equal to the CPU -> bit-identical;
- AMR mass drift on GPU: `2.2e-16`;
- safeguard `romeo/sanitizer.sbatch` (compute-sanitizer memcheck/initcheck/synccheck): 0 errors.
