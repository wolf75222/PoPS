# Testing

adc_cpp is tested with CTest, driven by the build presets. A behavior change must add or update a
test.

## Run the tests

Build and test with a preset, then run the matching CTest preset:

```bash
cmake --preset serial
cmake --build --preset serial
ctest --preset serial
```

Use the `mpi` and `parallel` presets for the MPI and Kokkos OpenMP test sets. The Python module is
built with the `python` preset; the relevant Python tests run from `build-py`.

## What CI runs

`ci.yml` mirrors the presets: the required gate builds and tests in Kokkos Serial (C++ and the Python
module), and a fuller mode adds MPI and Kokkos OpenMP. The CTest labels (`core`, `mpi`) select the
relevant sets.

## What CI cannot cover

The GPU and GH200 paths are validated by hand on ROMEO: the CI runners have no GPU. State explicitly
in a pull request what was validated locally and what could not be. The backend and test matrix is in
[BACKEND_COVERAGE](https://github.com/wolf75222/adc_cpp/blob/master/docs/BACKEND_COVERAGE.md).
