# Native numeric constants reference (ADC-618)

Every user-visible native numeric constant of the runtime carries an explicit CLASSIFICATION in the
effective-options report (`numerical_defaults_report()["classification"]`). The classes are:

- `public_knob` -- configurable end to end: a typed descriptor or a setter reaches the native use.
- `internal_default` -- a fixed default, not (yet) user-configurable, but inspectable in the report.
- `diagnostic_only` -- a counter or an instrumented fact, not a tuning knob.
- `hard_limit` -- a fixed cap enforced fail-fast (changing it requires a header rebuild).

The classification is FENCED: `tests/python/architecture/test_numeric_constant_fence.py` scans the
headers below and fails if any `inline constexpr` numeric constant is missing from the map, so a new
user-visible constant cannot ship unclassified. The source of truth is
`pops.runtime.defaults._CONSTANT_CLASSIFICATION` (Python) mirrored by
`numerical_defaults_report_to_dict` (C++ `python/bindings/core/bindings_detail.hpp`).

Scanned headers: `include/pops/runtime/numerical_defaults.hpp`,
`include/pops/core/foundation/types.hpp`, `include/pops/runtime/config/runtime_params.hpp`.

| Constant                              | Value   | Class            | Where configured |
|---------------------------------------|---------|------------------|------------------|
| kNewtonFailNone / Warn / Throw        | 0/1/2   | internal_default | fail-policy enum |
| kNewtonDefaultMaxIters                | 2       | public_knob      | IMEX(newton_max_iters=) |
| kNewtonDefaultRelTol / AbsTol         | 0       | public_knob      | IMEX(newton_rel_tol=, newton_abs_tol=) |
| kNewtonDefaultFdEps                   | 1e-7    | public_knob      | IMEX(newton_fd_eps=) |
| kNewtonDefaultDamping                 | 1       | public_knob      | IMEX(newton_damping=) |
| kNewtonDefaultFailPolicy              | none    | public_knob      | IMEX(newton_fail_policy=) |
| kNewtonFiniteAbsLimit                 | 1e300   | internal_default | fixed overflow guard |
| kKrylovDefaultRelTol                  | 1e-10   | public_knob      | GMRES/BiCGStab(rel_tol=) |
| kTensorKrylovDefaultMaxIters          | 200     | internal_default | fixed |
| kSchurKrylovCartesianMaxIters         | 400     | public_knob      | Program solver max_iter= (Cartesian default report) |
| kSchurKrylovPolarMaxIters             | 600     | public_knob      | Program solver max_iter= (polar default report) |
| kKrylovBreakdownTiny                  | 1e-300  | internal_default | fixed breakdown guard |
| kMGDefaultRelTol                      | 1e-8    | public_knob      | GeometricMG(tolerance=) |
| kMGDefaultMaxCycles                   | 50      | public_knob      | GeometricMG(max_cycles=) |
| kMGDefaultAbsTol                      | 0       | public_knob      | set_poisson(abs_tol=) |
| kMGDefaultMinCoarse                   | 2       | public_knob      | GeometricMG(min_coarse=) |
| kMGDefaultPreSmooth / PostSmooth      | 2 / 2   | public_knob      | GeometricMG(pre_sweeps=, post_sweeps=) |
| kMGDefaultBottomSweeps                | 50      | public_knob      | GeometricMG(bottom_sweeps=) |
| kFACDefaultMaxIters                   | 30      | public_knob      | Program solver max_iter= on CompositeTensorFAC |
| kFACDefaultFineSweeps                 | 400     | public_knob      | CompositeTensorFAC(fine_sweeps=) |
| kFACDefaultTol                        | 1e-9    | public_knob      | Program solver rel_tol= on CompositeTensorFAC |
| kFACInitialCoarseRelTol               | 1e-12   | public_knob      | CompositeTensorFAC(coarse_rel_tol=) |
| kFACInitialCoarseMaxCycles            | 100     | public_knob      | CompositeTensorFAC(coarse_cycles=) |
| kFFTDefaultSpectral                   | false   | public_knob      | FFT(spectral=) |
| kFFTZeroMeanGauge                     | true    | internal_default | fixed gauge |
| kFFTDirectDftFallback                 | true    | diagnostic_only  | counter in report |
| kEbCutFractionFloor                   | 1e-3    | public_knob      | CutCell(cut_theta_min=) |
| kWenoEpsilon                          | 1e-40   | internal_default | fixed WENO floor |
| kEbFaceOpenEps                        | 1e-6    | public_knob      | CutCell(face_open_eps=) |
| kEbKappaMin                           | 1e-2    | public_knob      | CutCell(kappa_min=) |
| kAmrDefaultMaxLevels                  | 2       | internal_default | native AMR envelope |
| kAmrRefinementDisabledThreshold       | 1e30    | internal_default | disabled-refinement sentinel |
| kAmrPhiRefinementDisabledThreshold    | 0       | internal_default | disabled-phi-refinement sentinel |
| kAdaptiveNoEvolvingBlockSentinel      | 1e30    | diagnostic_only  | degenerate-dt sentinel |
| kPhysicalDefault* (B0, gamma, ...)    | various | public_knob      | model / brick physical knobs |
| kPhysicalDefaultNativeIsothermalCs2   | 1       | internal_default | raw brick default |
| kCflSpeedFloor                        | 1e-30   | internal_default | CFL speed floor |
| kMaxRuntimeParams                     | 32      | hard_limit       | fixed device carrier cap |
