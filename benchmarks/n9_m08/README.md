# M08 gap: the Program CG against the native CG (N9, SV-T1)

Spec 0.3 SV-T1 measures, on ROMEO, the gap between the reference CG and the native CG on M08
Poisson with the same generated operator and rtol = 1e-10, from 1 to 64 CPU ranks and 1 to 8 GPUs,
256² to 4096² cells per rank, with the A B B A harness. The threshold is 0.9 of the native
throughput (OD9). `m08_gap.py` runs that comparison on the PoPS build of the current
`PYTHONPATH`; `benchmarks/romeo/n9_m08_gap.sbatch` runs it on ROMEO.

- `native`: the M08 field problem -Δφ = q, periodic, mean-zero gauge, solved by
  `CG(max_iter, rel_tol=rtol, abs_tol=0)` on the field route (the prepared affine problem over
  the generated field operator, then the SV8 gate).
- `program`: the same equation solved by the CG that `examples/api_v02/method_krylov.py` writes
  as a Program (`iterate` and `dot_all`), stopped at ||r|| <= rtol ||b|| on its recursive
  residual, with x0 = 0.

Each accepted step solves once from the unchanged state q. Both routes are compiled, bound and
warmed up, then advanced in interleaved A B B A blocks of fixed accepted steps.

## Run

From the repository root, with a native MPI build of `pops` on `PYTHONPATH` and the native test
environment of `tests/python` (`POPS_INCLUDE`, `POPS_KOKKOS_ROOT`, `POPS_CACHE_DIR`):

```sh
python3 benchmarks/n9_m08/m08_gap.py 256x256 --out m08_256.json
```

Options: `--rtol` (1e-10), `--max-iterations` (20000), `--steps` per block (2), `--blocks` of
A B B A (3), `--warmup` steps per route (1). On ROMEO, from a checkout on the login node:

```sh
benchmarks/romeo/submit_n9_m08_gap.sh                                  # x64cpu, one rank
benchmarks/romeo/submit_n9_m08_gap.sh --constraint=armgpu --gpus-per-task=1
```

The job is charged to the project account `r260105` (the project's ROMEO account; the other
scripts of `benchmarks/romeo/` still name `r250127`), and `--account=...` at submission overrides
it, since the submit script passes its arguments to `sbatch`. The job builds the wheel of the
checked-out revision in `/scratch_p/$USER/$SLURM_JOB_ID`, runs the
sizes of `POPS_N9_SIZES` (256 to 4096) and writes one JSON per size to
`~/pops-benchmark-results/n9-m08/`.

## Output

One JSON document (`pops.n9.m08_gap.v1`): the cells, rtol, the A B B A order, ranks, threads,
host, Slurm job and source revision, the compile time of each route, `timing` (median, minimum,
maximum and samples of the time per solve, per route), `throughput_program_vs_native` (native
median over program median), `od9_threshold` (0.9), `parity` (the largest difference of the two
mean-zero potentials, the largest native value and the native mean) and `limits`. No time
threshold is asserted; the expected parity is at the solve tolerance.

Local smoke runs (cloud container, 4 cores, Kokkos Serial, one rank, with a test suite running at
the same time, so the times are noisy and say nothing about ROMEO):

| cells | native s/solve | program s/solve | program/native throughput | parity linf |
|-------|----------------|-----------------|---------------------------|-------------|
| 64²   | 0.0127         | 0.0115          | 1.10                      | 1.5e-14     |
| 256²  | 0.0663         | 0.0760          | 0.87                      | 3.9e-14     |

## Limits (SV-T1 stays open)

- The Program CG applies the Program's Cartesian Laplacian (`program.laplacian` on cell
  midpoints), not the generated field operator of the native route. The harness checks that both
  solve the same discrete equation (the potentials agree to the solve tolerance), not that they run
  the same kernel. The reference CG of `pops.lib.solve` on the generated operator (G10) replaces
  this route when it exists.
- A `Uniform` Cartesian layout lowers to one box (`single_box`): under MPI one rank owns every
  cell, so only one rank measures a per-rank size, and the job runs one rank. The multi-rank rows
  of SV-T1 (up to 64 ranks) need a decomposed layout.
- The GPU rows need the armgpu build of `build_python.sh`; this thread could not reach ROMEO, so
  the job script and these expected outputs are the deliverable (OD10).
