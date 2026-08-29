# PoPS performance harness

This directory is a standalone CMake consumer of the real PoPS targets. It records JSONL
measurements and numerical checks; it does not impose machine-specific millisecond thresholds.

## Covered cases

- `arith_halo`: real `MultiFab` arithmetic followed by periodic `fill_boundary`. It compares
  `saxpy` with alias-safe `lincomb` using measured `A B B A` blocks and reports the paired
  geometric time ratio.
- `scalar_mg`: the production `GeometricMG<Dim>` algorithm with a manufactured Dirichlet mode. The
  same source is compiled and exercised in 1D, 2D, or 3D; the resolved PoPS native specialization
  owns the rank.

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
claims. The pinned pre-cutover revision is compiled as a native C++ oracle. The candidate is a
Python frontend campaign: it authors a `Program`, resolves and compiles a MODULE with the exact
NVCC/MPI/Kokkos toolchain, then binds it through `AmrSystem.install_program`. The ROMEO job runs
the native oracle and this Python candidate in paired `A B B A` order on four MPI ranks and four
distinct GPUs.

```sh
benchmarks/romeo/submit_adc700_program_cutover.sh
```

The default comparison pins the baseline to
`db3d390f43dfb14f12e88db31a9b3e631ff50488`; the candidate defaults to the submitted checkout
(`HEAD`). Before archiving, the batch script requires every campaign file (including the Python
driver, verifier, wheel helpers, and GPU wrapper) to be present in that commit and rejects dirty or
untracked campaign paths. Compilation and execution then use only the extracted immutable baseline
and candidate trees. `POPS_ADC700_BASELINE_REF` may name an alias only when it resolves to the pinned
SHA; a different baseline is refused. `POPS_ADC700_CANDIDATE_REF` can select a deliberate candidate
commit, subject to the same archive preflight.

After the wheel is proven, the job freezes both extracted trees and writes immutable archive
receipts containing every file mode/size/SHA-256 plus a canonical tree digest. Each ABBA invocation
revalidates both receipts; the verifier requires the candidate receipt, its linked commit SHA/tree,
and read-only tree state. The receipt helper is copied from the committed candidate tree to one
immutable work-root path outside both archives, because the pinned baseline predates ADC-700. A
four-rank candidate toolchain receipt is shared with the pinned C++
oracle, whose JSONL row emits the complete object and receipt metadata; comparison requires exact
compiler/NVCC-wrapper/MPI/Kokkos/flags/libs equality. The main AMR `PatchLayout` is explicitly
`distribute_coarse=true`, `coarse_max_grid=n/2`; authenticated fine boxes and coarse local/total
box counts are compared in every raw row.

The baseline C++ configure/build consumes that receipt before compiling: CMake 20 is mandatory and
the real `MPI::MPI_CXX`/`Kokkos::kokkos` compiler, include, option, definition, and library
properties must match the receipt. A probe-only configure records those target properties, including
the two defining Kokkos header hashes and the MPI file hashes, before the receipt is frozen; the
oracle emits a `toolchain_build_attested` witness only when this preflight has succeeded, and the
candidate receives the same gate from the batch harness.

The job fails closed unless:

- Kokkos reports a real device execution space (`Cuda`, `HIP`, or `SYCL`), exactly four MPI ranks
  each see one scheduler-assigned device through `MPI_COMM_WORLD`; every run records the complete
  four-UUID map and rank-local UUID, and all recorded device UUIDs are distinct;
- every baseline/candidate run validates the same 2-D conserved state and topology;
- every candidate record carries absolute-path/SHA-256 provenance for the installed extension and
  generated MODULE, including dimension, MPI, Kokkos space, ABI and artifact key, plus the exact
  NVCC-wrapper version/hash, MPI/Kokkos file hashes, compile/link flags, and campaign harness hashes;
- the retained wheel proof authenticates one Dim=2 MPI/Kokkos wheel, its installed extension and
  ABI, and the archived proof script;
- candidate Uniform and planned refined-AMR probes report zero scratch allocations after
  bind+warmup preparation (counters are reset at that boundary) and dispatches bounded by
  `O(operations*levels), never O(cells)` at two resolutions with fixed operation/level counts;
- at least five complete ABBA blocks are present, and `verify.py` recomputes all medians, ratios,
  tolerances, and parity fields directly from raw JSONL before authenticating the report;
- the final numerical signatures agree within the declared tolerance;
- the median paired candidate/pre-cutover throughput ratio is at least `0.98`.

Raw JSONL, GPU inventory, wheel proof, comparator report, and the authenticated machine-readable
`pops.adc700.program_cutover.report.v1` report are retained under
`$HOME/pops-benchmark-results/adc700`. This campaign is intentionally outside routine CI: a CPU
run, missing device/module provenance, malformed ABBA ordering, failed probes, or absent hardware
produces no device or performance proof.
