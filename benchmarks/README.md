# PoPS performance harness

This directory is a standalone CMake consumer of the real PoPS targets. It records JSONL
measurements and numerical checks; it does not impose machine-specific millisecond thresholds.

## Covered cases

- `arith_halo`: real `MultiFab` arithmetic followed by periodic `fill_boundary`. It compares
  `saxpy` with alias-safe `lincomb` using measured `A B B A` blocks and reports the paired
  geometric time ratio.
- `tensor_krylov`: real prepared generic BiCGStab with a constant nonsymmetric coefficient tensor
  and a manufactured solution, using one persistent diagonal `GeometricMG` V-cycle as the explicit
  prepared preconditioner.

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

## ADC-700 Program cutover campaign

`benchmarks/adc700/` is a separate, non-routine campaign for the two hardware-dependent ADC-700
claims. It compiles one AMR Forward-Euler workload twice: the pinned pre-cutover revision uses its
native temporal route, while the candidate installs the equivalent Program-only route. The ROMEO
job runs both binaries in paired `A B B A` order on real GPU-backed MPI ranks.

```sh
benchmarks/romeo/submit_adc700_program_cutover.sh
```

The default comparison is
`db3d390f43dfb14f12e88db31a9b3e631ff50488` (the parent of the first ADC-700 cutover commit)
against `bdb169b96f71f0f809501b9b2f38b44797749212`. Override
`POPS_ADC700_BASELINE_REF` or `POPS_ADC700_CANDIDATE_REF` only to run a deliberately different
campaign.

The job fails closed unless:

- Kokkos reports a real device execution space (`Cuda`, `HIP`, or `SYCL`), every MPI rank sees
  exactly one scheduler-assigned device, and all recorded device UUIDs are distinct;
- every baseline/candidate run validates a genuinely refined hierarchy and conserved state;
- the final numerical signatures agree within the declared tolerance;
- the median paired candidate/pre-cutover throughput ratio is at least `0.98`.

Raw JSONL, device inventory, and the machine-readable
`pops.adc700.program_cutover.report.v1` report are retained under
`$HOME/pops-benchmark-results/adc700`. This campaign is intentionally outside routine CI: a CPU
run, missing device inventory, malformed ABBA ordering, or absent hardware produces no device or
performance proof.
