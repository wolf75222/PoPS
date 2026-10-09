# Structural Maxwell–Cattaneo prototype (C11 + T3)

Status: source authoring and independent oracle checked; native **not_executed**. Files are new in isolated PoPS-resource-lifetime. Public validation, resolution and graph/source emission use MAIN9e9a0e5 Python sources read-only, not the older isolated package. No core changes, build, installation or native run.

The model is closed by constant finite positive runtime parameters tau and kappa, periodic unit-square boundaries and prescribed Fourier initial data. Unit volumetric heat capacity is used, exactly giving T_t+div(q)=0 and tau*q_t+q=-kappa*grad(T). T is a scalar state; q is a separate two-component state. It is absent from the original M01–M28 table: M11 Maxwell–Stefan and M12 electromagnetic Maxwell are different equations. This is an additional structural test, not an extension of their qualification.

Both principal flux rows are declared with cross-state sampling and a complete characteristic bound sqrt(kappa/tau). The principal symbol has eigenvalues 0 and ±sqrt(kappa/tau)*|normal|; it is symmetrizable, not incorrectly assumed Euclidean-symmetric. C11 produces one native PreparedPrincipalFlux of width3. The q relaxation uses a generic two-component LocalResidual with named physical source -q/tau. The immutable capture is qstar from the explicit principal step; the initial Newton guess is deliberately .9*qstar. Confusing the guess with the captured right-hand side changes the equation and is detectable.

The temporal method is first-order IMEX Euler: Ustar=(I+dt*Lh)Un, then (I-dt*S)Unext=Ustar. Only q enters the implicit local solve; Tstar is committed unchanged by relaxation. No special Cattaneo opcode, closed-form division shortcut in the Program, or renamed single-vector storage is used. Public Positive parameter domains refuse tau<=0; native bind refusal on every rank is prepared in the runner. This is not an asymptotic-preserving diffusion-limit claim.

The independent oracle has no PoPS import. It builds the complete complex 3×3 Fourier generator and uses SciPy expm for the continuum and semidiscrete solutions. Initial data have mode(1,2), so both x/y derivatives are active, with nonzero longitudinal/transverse flux and mean flux. Cell means include sinc(1/N)*sinc(2/N); a separate 12×12 Gaussian quadrature verifies them. The mean T is conserved; mean q decays. The energy pairing diag(1,tau/kappa,tau/kappa) verifies the physical dissipation identity.

N=16, final_time=.04, dt=.004/.002/.001; rebind parameter pairs (.1,.025) and (.12,.048). Criteria were fixed before native: exact discrete-scheme max error2e-10, T inventory2e-12, temporal orders between .8 and1.2, permutation difference2e-10, bound initial-state error2e-14, rebind difference>1e-3. Time convergence compares against exp((Lh+S)t), preventing the first-order spatial error from hiding it. Continuum spatial discrepancy is reported separately, never called time error. Independent predicted errors are [3.6270e-4,1.8042e-4,8.9977e-5] and [4.3289e-4,2.1392e-4,1.0633e-4], with orders[1.0074,1.0037] and[1.0169,1.0085]. Spatial differences are .0126165/.0151285 for this coarse first-order grid.

Both canonical and renamed/permuted variants passed validate/resolve/ProgramModelGraph/emission against MAIN9e9a0e5 without patching production. Each emits exactly one complete width3 principal resource and one generic prepared two-component nonlinear solve. The second variant reverses q component order and block insertion order, and changes state/model/parameter/block names.

Tests: **19 passed in 6.75s** (pure oracle plus public source authoring). Command from this checkout:

```sh
env -u PYTHONPATH /Users/romaindespoulain/miniforge3/envs/pops-api040/bin/python -m pytest -q -o pythonpath=../PoPS/python tests/python/unit/structural_cattaneo_test_oracle.py tests/python/unit/structural_cattaneo_authoring.py
```

After integration, use `-o pythonpath=python`. The requested structural_cattaneo prefix means pytest needs explicit file selection. The first test run revealed a test-module filename collision with the oracle import; renaming the test file resolved that fixture issue.

Central native command after package installation/authentication:

```sh
env -u PYTHONPATH POPS_NATIVE_DIM=2 python examples/migration/scientific/structural_cattaneo.py --output outputs/structural_cattaneo
```

The prepared runner compiles one artifact per structural variant, rebinds each to both physical parameter pairs and three external time grids, checks invalid tau binding, saves and reopens actual initial/final states, compares against independent matrix results, and compares authenticated canonical/permuted saved states. It records source/native hashes, execution context and run reports. The runner is syntax/source-reviewed but its native flow is unexecuted. No native success, MPI/GPU, AMR, heterogeneous layout, stiff diffusion limit or full C11/T3 acceptance is claimed.
