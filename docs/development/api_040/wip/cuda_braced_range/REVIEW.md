# CUDA737206 braced-range source candidate v1 - isolated one CPP

Exact Source6d/SDKaf8 failure: recorded node receipt e0536c5f and complete native
build log526738 bytes SHA5eb1166d. The real outer Ninja command is preserved at
line2613 in recorded-cuda-command-and-errors.json. It invokes strict_nvcc_wrapper
with --fmad=false, -std=c++20, -extended-lambda, -arch=sm_80, Kokkos/MPI/HDF flags.
Actual node reports NVCC12.6.85, GNU13.4.0 and Kokkos5.2.1 CUDA/OpenMP/Serial.
These are observed toolchain facts, not a Clang CUDA simulation.

## Cause supported and missing evidence

At232/992/1361, the actual host compilation diagnoses expected semicolon before
closing brace of a bare braced range-initializer. Subsequent begin/end deduction
receives const MultiFab<2,Kokkos::CudaSpace>*&, rather than an initializer-list
container. The diagnostics' cudafe-renamed _GLOBAL__N__ anonymous namespace and
NVIDIA annotation diagnostics identify an actual NVCC-to-host translation path.
The Source constructs are valid ordinary C++20; current CPU/Clang compilation
passes. amr_layout_transfer.hpp already includes array and vector, and CPP already
uses std::array elsewhere. The recorded symptom is not the ordinary missing
initializer_list-header deduction diagnostic; adding a guessed include alone
would not explain pointer begin/end and all three closing-brace failures.

A coherent inference is that dependent bare braced ranges are mishandled during
that translation and presented as a brace/statement-expression/pointer range to
GNU. Actual inner NVCC argv and generated host .cudafe1.cpp/.ii bytes are NOT in
this receipt. The precise emitted syntax/pass causing this remains unproven.
No vendor-bug ID or universal CUDA compiler defect is claimed. The proposed
bounded --keep probe must settle that distinction under a separate Root GO.

## Bounded candidate

Only src/runtime/amr/amr_layout_transfer.cpp changes: explicit private <array>
include plus the three failed range initializers become named, typed std::array
containers. Pointer groups retain const MultiFab<Dim>* types and exact member
order; budget values retain std::size_t and exact five-value order. All ranges
snapshot the same values once before traversing. Arrays use automatic storage,
no heap allocation/new kernel. Loop bodies, null guards, validation/overflow
checks, ghost maxima, serialization integer order, formulas/constants/contracts
are byte-identical after reversing these range-header substitutions. No model,
HeaderSignature/ABI/version or physical implementation branch changes.
Other braced lists are untouched. No Main patch, Native build/import, remote
write/job or failed evidence rewrite occurred. Source6d remains frozen for live
CPP and original Python scientific receptions.

## Actual validation and pending NVCC plan

Read-only git apply --check succeeds. Whole actual CPP Host syntax-only passed
using the current local producer's amr_layout_transfer row/headers/flags: exit0,
2.862 seconds, no warnings. It creates no object/link and is not CUDA acceptance.
source-equivalence-review.json proves inverse substitution restores exact base.
Actual wrapper command bytes, parsed outer argv/argc and errors are pinned.

bounded-NVCC-probe-plan.json is NOT executed: after explicit Root GO in an
admitted existing allocation, preserve original Source/log/dependency identities,
obtain the strict wrapper inner argv and keep actual cudafe host intermediates.
Compile only original and candidate CPP with exact otherwise unchanged flags and
new temp output/dependency/keep paths. Compare the emitted loops. Do not replay a
campaign, change floating-point flags, patch frozen Source or submit a new job.
The typed-array candidate remains pending independent review and actual NVCC proof.
