# Collectives per CG iteration (N9, SV20)

Spec 0.3 §7.6 asks the lowering of the reference CG to make no more collectives per iteration
than the native CG (SV20). This directory counts them, for the CG that
`examples/api_v02/method_krylov.py` writes as a `Program` (T04 global) and for the native
`CG(max_iter=2000, rel_tol=1e-12, abs_tol=1e-13)` of M08, and attributes each collective of an
iteration to the PoPS function that issued it.

- `mpi_collective_trace.c` is a PMPI library loaded with `LD_PRELOAD`. It records each MPI
  collective call (operation, element count, element size, reduction operator) and counts the
  point-to-point calls; every wrapper forwards to its `PMPI_` entry point unchanged. With
  `POPS_N9_TRACE_FRAMES=<depth>`, it also keeps the return addresses of each call.
- `measure_cg_collectives.py` runs both cases to their end time on the PoPS build of the current
  `PYTHONPATH`, then reads the trace. An iteration is the period of the longest periodic stretch
  of (collective, call site) pairs, and each collective of that period is attributed through the
  symbol tables of the loaded objects (`nm`).

## Run

From the repository root, with a native MPI build of `pops` on `PYTHONPATH` and the native test
environment of `tests/python` (for example `POPS_INCLUDE`, `POPS_KOKKOS_ROOT`, `POPS_CACHE_DIR`):

```sh
mpicc -O2 -fPIC -shared -o libpops_n9_mpi_trace.so benchmarks/n9_collectives/mpi_collective_trace.c -ldl
mpiexec -n 2 -x LD_PRELOAD=$PWD/libpops_n9_mpi_trace.so -x POPS_N9_TRACE_FRAMES=16 \
    python3 benchmarks/n9_collectives/measure_cg_collectives.py --case both --output cg.json
```

`--case program` or `--case native` runs one case. Rank 0 prints and writes one JSON document
(`pops.n9.cg_collectives.v1`) with, for each case, whether its own checks passed, the
collectives of the run, `collectives_per_iteration`, the iteration's collectives by kind, and
`iteration_sites`, the collectives of the iteration grouped by the function that issued them.

A single-rank lane decides the fused checks locally and makes no collective for them, so the
counts that matter for SV20 are those at two ranks or more. Uniform Cartesian layouts lower to
one box: at two ranks, rank 0 owns every cell, rank 1 none, and the trace of either case sees no
point-to-point call. Every rank still makes every collective of an iteration, so the count per
iteration does not depend on that ownership.

## Measurements

Collectives per CG iteration, measured on 30/10/2026 in a 4-core container (OpenMPI 4.1.6,
Kokkos 4.4.01 Serial, Release build, `--case both`, `POPS_N9_TRACE_FRAMES=16`). Both cases
passed their own checks in every run.

| Tree | Ranks | Program CG (T04 global) | Native CG (M08) |
| --- | --- | --- | --- |
| base 7477934 (b6 with #688 and #693) | 1 | 345 | not measured |
| base 7477934 | 2 | 357 | 3 |
| N9 lot 1 | 1 | 39 | 3 |
| N9 lot 1 | 2 | 57 | 3 |

At two ranks after lot 1, an iteration of the Program CG makes 4 `dot_all` reductions (one
compensated `Allreduce` each), 9 active-mask queries and 5 status maxima (one fused `Allreduce`
each), 20 collectives in the generated Program body, 11 in `prepare_generated_state`, 3 in
`attach_affine_flux_expression` and 5 in the mesh boundary session. The native CG makes 2 sums
in `solve_cg` and 1 maximum in `apply_general_field`.
