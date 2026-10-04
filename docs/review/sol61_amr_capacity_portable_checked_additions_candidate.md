# AMR checkpoint interface capacity: portable checked additions candidate

This candidate replaces the braced two-element range-for with two explicit additions, preserving order, size_t overflow checks, exception type/message, interface-fragment multiplication and every capacity bound. It changes no equations, serialized schema (AMR12), ABI10, parser or storage bits. Parent574209e0 has original header SHA780287346eb2c300d8de7fd206ecafa4fb64c1a453996e94dae86fbfa8e4138b; candidate75767bytes SHA b5f4a7810e08850342a203be82d485c4ca63380ee954a3e364f200e4d121e9af.

Actual full GPU build733078 failed at this range-for in the system.cpp TU. The first isolated CUDA job733095 refused allocation before compilation and produced no variant exits; neither establishes that this candidate fixes NVCC. ROOT integration waits for causal compile evidence. Original jobs and packets remain immutable.

The actual complete-header Host probe instantiates dimensions1/2/3, checks exact +1 interface byte accounting and both first/second overflow exceptions. Original and candidate return identical 576/584/592. Fresh candidate Host compile/execution passed using clang++/OpenMP and the existing ir17 dependency headers; this is not a full Native build or CUDA execution. The probe and logs are retained in the independent negative preparation directory, header-candidate-574.

The existing pure AMR engineering reader cohort passed18 tests (0.45s), including typed storage and resealed negative cases. The two attempted runtime identity unit nodes require a real _pops extension and could not collect under the Source-only interpreter; their collection error is preserved. No Native parser/runtime acceptance is inferred or reported.
