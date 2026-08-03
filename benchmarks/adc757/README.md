# ADC-757 heterogeneous numerics campaign

This is a non-routine hardware qualification campaign. It is deliberately absent from ordinary
CI because a valid result requires at least two MPI ranks, one distinct accelerator per rank, and
two native Kokkos streams per accelerator.

The native harness exercises two routes:

- `prepared_local_time`: the baseline advances every cell at the smallest step; the candidate
  advances the slow partition only when due and submits the two partitions to prepared streams;
- `cost_aware_load_balance`: the baseline uses round-robin ownership; the candidate uses prepared
  task costs, migrates ownership with a timed `MPI_Alltoallv`, and executes the two local work
  partitions concurrently.

Both routes retain the same numerical result and publish mass, restart, rollback, and ledger
errors. The stream probe runs five paired ABBA blocks and reports overlap only when the concurrent
pair is measurably faster. The outer SLURM driver runs at least five ABBA blocks for each scenario.
`assemble.py` rejects incomplete or reordered measurements, and `verify.py` independently checks
the final report. Neither program substitutes CPU measurements or inferred overlap for GPU data.
When Kokkos provides `Experimental::partition_space`, PoPS consumes that API directly. The ROMEO
CUDA installation currently uses Kokkos 4.4.1, so the compatibility route creates non-blocking CUDA
streams explicitly, wraps them in Kokkos execution-space instances, and retains RAII ownership until
all lane workspaces and instances have been destroyed.

On ROMEO, after the candidate revision is available in the checkout configured by
`POPS_ADC757_REPO_ROOT`, submit with:

```bash
benchmarks/romeo/submit_adc757_heterogeneous_numerics.sh
```

The job uses account `r250127`, the `armgpu` constraint, two MPI ranks and two GH200 GPUs. It
archives the exact revision into `/scratch_p`, compiles the aarch64/CUDA executable inside the
allocation, verifies the rank-local GPU UUIDs, runs the campaign with `srun`, and copies the small
report artifacts to `~/pops-benchmark-results/adc757`.
