# ADC-757 heterogeneous numerics campaign

This is a non-routine hardware qualification campaign. It is deliberately absent from ordinary
CI because a valid result requires exact retained-wheel installations of the same scientific
scenario in Serial, threaded, accelerator and accelerator+MPI modes, plus at least two MPI ranks,
one distinct accelerator per rank, and two native Kokkos streams per accelerator.

The ABBA executable is only a microbenchmark. It cannot close ADC-757 by itself. Before compiling
that executable, the ROMEO driver now:

1. builds and installs one retained candidate wheel through `scripts/build_python.sh --mpi`;
2. authenticates every installed wheel member with `scripts/prove_installed_wheel.py`, then binds the
   native harness to that wheel's signed header tree (wheels intentionally omit `popsConfig.cmake`);
3. runs `pops.runtime.doctor.doctor()` against that installation and records its exact result;
4. requires an `installed-runtime-matrix.v1` receipt for one identical AMR-advection scenario under
   Serial, threaded, GPU and GPU+MPI execution, including artifact identity, module/artifact ABI and
   a common solution digest;
5. requires native receipts proving that the GPU+MPI artifact actually consumed cell-local time and
   accepted-boundary AMR ownership migration, with no fallback and at least one post-migration step.

Header presence and vector updates are never promoted to runtime evidence. On the current bounded
base, `runtime_probe.py` therefore writes an explicit refusal and exits before ABBA: ADC-757C's live
`decide_rebalance/apply_rebalance_decision` route and ADC-757G's accepted local-time publication must
first be integrated into the exact candidate, then exercised by a receipt-producing runtime driver.

The native harness exercises two routes:

- `prepared_local_time`: the baseline advances every cell at the smallest step; the candidate
  advances the slow partition only when due and submits the two partitions to prepared streams;
- `cost_aware_load_balance`: the baseline uses round-robin ownership; the candidate uses prepared
  task costs, migrates ownership with a timed `MPI_Alltoallv`, and executes the two local work
  partitions concurrently.

Both microbenchmark routes retain the same numerical result and publish mass, restart, rollback, and
ledger errors. The stream probe runs five paired ABBA blocks and reports overlap only when the
concurrent pair is measurably faster. The outer SLURM driver runs at least five ABBA blocks for each
scenario. `assemble.py` rejects incomplete or reordered measurements and binds every row to the
runtime-matrix digest, retained wheel and module ABI. `verify.py` independently checks the final
report. Neither program substitutes CPU measurements, header detection or inferred overlap for
installed PoPS runtime evidence.
When Kokkos provides `Experimental::partition_space`, PoPS consumes that API directly. The ROMEO
CUDA installation currently uses Kokkos 4.4.1, so the compatibility route creates non-blocking CUDA
streams explicitly, wraps them in Kokkos execution-space instances, and retains RAII ownership until
all lane workspaces and instances have been destroyed.

On ROMEO, after the candidate revision is available in the checkout configured by
`POPS_ADC757_REPO_ROOT` and after a complete runtime matrix has been produced, set
`POPS_ADC757_RUNTIME_EVIDENCE` to that JSON file and submit with:

```bash
benchmarks/romeo/submit_adc757_heterogeneous_numerics.sh
```

The job uses account `r250127`, the `armgpu` constraint, two MPI ranks and two GH200 GPUs. It
archives the exact revision into `/scratch_p`, compiles the aarch64/CUDA executable inside the
allocation, verifies the rank-local GPU UUIDs, runs the campaign with `srun`, and copies the small
report artifacts to `~/pops-benchmark-results/adc757`.
