# Validation (core)

Validation status of the `adc_cpp` core. The per-backend coverage matrix is in
[BACKEND_COVERAGE.md](BACKEND_COVERAGE.md) ; the device port details in
[GPU_RUNTIME_PORT.md](GPU_RUNTIME_PORT.md) ; the application validation (named models, diocotron,
ROMEO runs) in the [`adc_cases`](https://github.com/wolf75222/adc_cases) repository.

## CI

- core ctests in Release and in Kokkos (Serial).
- MPI np=1/2/4, bit-identical outputs.
- Python module : additional suite (bindings and DSL).

## AMR

- conservative multi-patch reflux to machine roundoff (mass drift ~ 1e-15).
- the Poisson is solved at the coarse level then injected toward the fine : the AMR refines the transport, not
  the elliptic solve (no multi-level composite solve, no global Schur on AMR).

## GPU GH200 (outside CI)

- System production np=1 validated (#97).
- geometric multigrid device-MPI np=1/2/4 validated (#93).
- AmrSystem + MPI + GPU validated, bit-identical (phase 10, dmax=0, #105).
- Schur and polar device : 7/7 device-clean in Kokkos Cuda single-GPU, and MPI+Kokkos Cuda multi-GPU
  rank-invariant (10 tests, #157), plus Kokkos OpenMP in CI (#155). Covers condensed_schur,
  polar_transport, lorentz, full_tensor, polar_poisson, krylov, schur_condensation (all device-clean
  GH200, compute-sanitizer 0 error). The 4 initial failures came from the tests (host functors or pointers
  called in device kernels, or host read of an async output without fence), fixed
  #150/#152/#158 ; the elliptic / Schur / polar library is device-correct.

## FFT under MPI

`System` in MPI np>1 refuses the FFT cleanly (#106, no more segfault). `DistributedFFTSolver` exists
and is tested separately, but is not routed in `System`.
