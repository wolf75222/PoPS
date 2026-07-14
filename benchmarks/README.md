# PoPS performance harness

This directory is a standalone CMake consumer of the real PoPS targets. It records JSONL
measurements and numerical checks; it does not impose machine-specific millisecond thresholds.

## Covered cases

- `arith_halo`: real `MultiFab` arithmetic followed by periodic `fill_boundary`. It compares
  `saxpy` with alias-safe `lincomb` using measured `A B B A` blocks and reports the paired
  geometric time ratio.
- `tensor_krylov`: real `TensorKrylovSolver`/BiCGStab with a constant nonsymmetric coefficient
  tensor and a manufactured solution, using one diagonal-tensor `GeometricMG` V-cycle as
  preconditioner.

No other PoPS kernel or solver is represented by this harness.

Each measured region is bracketed by a Kokkos device fence and an MPI barrier. The recorded sample
is the maximum rank time. Warmups are discarded, robust statistics include median, MAD, p10/p90,
and a trimmed mean, and validation runs outside the timed interval. The JSONL metadata includes the
source revision/dirty flag, compiler, build type, Kokkos execution space, concurrency, MPI rank
count, host, and SLURM job id.

## Local build and run

From the PoPS repository root:

```sh
cmake -S benchmarks -B build/benchmarks -DCMAKE_BUILD_TYPE=Release
cmake --build build/benchmarks --target pops_benchmark -j
build/benchmarks/bin/pops_benchmark --case=all --output=benchmarks.jsonl
```

Enable MPI explicitly with `-DPOPS_BENCH_ENABLE_MPI=ON`; the harness otherwise uses PoPS's serial
communication seam. See `./pops_benchmark --help` for case sizes and solver controls.

## ROMEO Arm GPU

Submit from any checkout with:

```sh
benchmarks/romeo/submit_armgpu.sh
```

The batch job loads `romeo_load_armgpu_env`, configures and builds inside the allocation, and uses
fail-fast `srun` with one GPU per rank. Work and build files default to
`/scratch_p/$USER/$SLURM_JOB_ID`; JSONL results remain under
`$HOME/pops-benchmark-results`. Override paths with `POPS_BENCH_SOURCE_DIR`,
`POPS_BENCH_WORK_DIR`, `POPS_BENCH_BUILD_DIR`, `POPS_BENCH_RESULTS_DIR`, or `POPS_KOKKOS_ROOT`
when needed.
