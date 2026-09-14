# PoPS - Plasma-Oriented PDE Solver

PoPS is a C++20 finite-volume solver with a Python interface for coupled transport,
diffusion, source and field problems on Cartesian and adaptive meshes. Python describes
physics and a time program; generated C++ and Kokkos execute the numerical kernels.

```text
Case -> validate -> resolve -> compile -> bind -> run
```

## Install

You need Git, a C++20 compiler, and conda (for example Miniforge). From a clone:

```bash
git clone https://github.com/wolf75222/PoPS.git
cd PoPS
bash scripts/setup_env.sh
conda activate pops
bash scripts/build_python.sh --dim 2
```

The setup installs Python 3.12, CMake, Ninja and the local CPU dependencies. The build
installs `pops` and checks the native artifact. Choose `--dim 1`, `--dim 2` or `--dim 3`
for the spatial dimension of your problem; each Python process uses one native dimension.
Rerun the build script after native changes; it reuses the build cache and retains other
installed dimensions.

Kokkos is required. MPI and parallel HDF5 are enabled together by
`bash scripts/build_python.sh --dim 2 --mpi`. For a local OpenMP Kokkos installation,
run `bash scripts/kokkos_openmp_conda.sh` before rebuilding. Backend availability is a
property of the installed artifact.

## Run a tutorial

With the environment active, run this periodic scalar-advection problem from the repository root:

```bash
env -u PYTHONPATH python docs/tutorials/scalar_advection/01_openmp_preset_ssprk2.py
```

The default uses one thread; set `POPS_THREADS` before launch to request more on an
OpenMP build. [Tutorials](docs/tutorials/README.md) introduce the physical model first,
then the mesh, numerical method, time program and simulation. They also cover explicit
program authoring, AMR, IMEX, fields, output and restart.

[Examples](examples/README.md) contain complete acceptance cases and focused API workflows.
Named application campaigns live in [adc_cases](https://github.com/wolf75222/adc_cases).

## Build and test the C++ core

```bash
cmake --preset serial
cmake --build --preset serial --target test_prepared_cartesian_nd
ctest --preset serial -L '^cpp-target:test_prepared_cartesian_nd$' --output-on-failure
```

The presets select Dim=2. This builds and runs one numerical target; use
`cmake --build --preset serial` and `ctest --preset serial` for the complete Serial suite.
[Contributing](CONTRIBUTING.md) explains test selection and the other presets.

Source/package and documentation checks:

```bash
python scripts/check_packaging_manifest.py
bash scripts/build_docs.sh
```

Local build and test results cover their selected configuration. Release qualification
has separate [acceptance requirements](docs/development/migration_verification_scope.md);
a CPU run does not establish GPU or cluster behavior.

## Project map

- [Architecture](docs/ARCHITECTURE.md): authoring, compilation, runtime and source layout.
- [Algorithms](docs/ALGORITHMS.md): implemented methods, equations and limitations.
- [Tutorials](docs/tutorials/README.md), [examples](examples/README.md) and
  [benchmarks](benchmarks/README.md): learning, executable references and performance protocols.
- [Contributing](CONTRIBUTING.md): development, focused tests and review.
- [Versioning](docs/VERSIONING.md) and [changelog](CHANGELOG.md): public API and releases.

Licensed under [BSD-3-Clause](LICENSE). See [SECURITY.md](SECURITY.md) for vulnerability reports.
