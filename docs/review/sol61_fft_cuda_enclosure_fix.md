The real CUDA SDK29 build 732954 failed on `system_aux.cpp.o` after clone/setup,
using source b376 and NVCC 12.6.85 on GH200. Its closed `build.log` SHA256 is
`7d91c3a835f3cc97d15032e04ab183f7264993100d252e014c530aaef75b726e`.
Eleven FFT launches at lines 341, 370, 389, 412, 452, 474, 484, 536, 551, 589 and
672 share the same compiler refusal: their extended Host/device lambdas have
private enclosing member functions. This matches the address-accessibility rule
in [NVIDIA's language support documentation](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-support.html).

`PoissonFFT` keeps its private orchestration and public API. Only the kernel
launches move into named public static functions of the namespace-detail
`PoissonFFTDeviceKernels<Dim, MemorySpace>` helper. Each receives the same view and
scalar snapshots already taken by the plan. The eleven launch expressions, labels,
range policies and arithmetic bodies remain unchanged apart from whitespace and
the explicit `local_count` parameter name. The same reverse-bit device routine
moves with the symbol kernel. No method receives or captures the FFT plan object.

Allocation, diagnostics, FFTW selection, copies, swaps, fences, votes, exchange
order, symbol threshold and normalization stay at their original plan sites.
There is no model recognition, numerical/FP/tolerance change, public PoissonFFT
signature or instance-layout change, or external ABI version change. The updated
headers require a new authenticated build/header signature before Native claims.

The Source review checks the exact negative header (SHA256
`6594028d0c5eb8e7203d8935a16326aedf20b24da96a062aeb9e04308804c92a`), all eleven
launch expressions, and reverses the extraction to prove the remaining plan bytes
are unchanged. The bounded probe explicitly instantiates Dim1/2/3 and checks six
Cartesian shapes, radix and non-power-of-two paths, real/imaginary independent
discrete Fourier modes, repeated solves, null symbol, extent rejection and injected
allocation refusal. Host baseline/candidate outputs must match every published
double byte, separately with and without FFTW; the existing 1e-11 oracle stays.

Root's prepared CUDA command adds a static assertion that the actual default
execution type is `Kokkos::Cuda`. It must preserve the original private-function
compilation refusal and then compile/run the corrected probe on the actual GPU.
No CUDA execution is anticipated here. The bounded engine probe does not qualify
installed Native, MPI, model composition or broader scientific results.
