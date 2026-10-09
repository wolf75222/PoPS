# Original AMR field Newton: independent sign diagnosis - 30 September 2026

Source reviewed in an exclusive checkout:
`29daef24292f17c3f11b0b6c8c9357523e951600`.
No MAIN, environment, native library or production header changes were made.
No compilation, build, JIT or native/MPI test was launched by this review.

## Authentic failure and inventory

Root's actual receipt is
`outputs/native-original-amr-canonical-ale-cpp-wave6-dim1-20260930`.
The CTest log SHA256 is
`e5108926f24636d83d17ce669976ae7bf39e2a14d19e59d3eec8df5911975681`.
Each of the two rank XML files contains ten executed tests, zero skips and the
same two failures: `OriginalAmrNonlinearResidualTwoResolutionsAndPermutation`
and `OriginalFieldOutcomeStagesAllLevelsAndRevalidatesBeforeAccept`.
Rank0 XML SHA256 is
`8b1c075eaf12a356eaf6d351ee01eff2774d78e0c4d3dc312dcb3d86bce8a9af`;
rank1 is `fa9de948ee543196055539d6102eb4e488858d8be23cb83af42d36a2c9900b0c`.
The first failure reports `amr_field_newton_line_search_failed` at fixture
line154; the publication case has no solved value at line315.

The selected Serial inventory contains only the four old CompositeGeneralField
tests. The six new declarations reside in an included `.inc` file, whereas
configure-time `gtest_add_tests(SOURCES ...)` scans the primary `.cpp` source.
The MPI wrapper executes the binary's full ten-case inventory. Therefore this
receipt contains **no Serial PASS evidence for these two new cases**. Root
confirmed and corrected the initial inference of a Serial/MPI differential.
Serial registration of all six included cases is required in the repair.

## Exact localization

The generic AMR workspace uses the defect convention `J delta = b-A(q)` and
then `q_trial = q + alpha delta`. Its linear solve copies the supplied RHS
unchanged, normalizes that vector, and uses a positive initial rotated RHS.
`PreparedAmrFieldResidual` supplies **+F(q)** as that RHS, while its central
finite-difference JVP is **+dF(q)v**. The adapted Newton direction therefore
solves `J delta = F(q)` instead of `J delta = -F(q)`.

The Uniform `PreparedSpatialResidual` already keeps the required distinction:
it evaluates the public physical F, negates only the workspace defect callback,
and retains the positive derivative of F. The original AMR final recheck also
evaluates physical F directly; that recheck should remain unchanged.

The scalar counterexample F(q)=q-target with seed zero is conclusive:
delta=-target, so F(q+alpha delta)=-(1+alpha)target. Its norm grows for every
positive alpha and cannot satisfy the existing Armijo condition. The actual
positive central finite-difference orientation preserves this conclusion.
Changing a tolerance, decreasing minimum_step or changing the target cannot
repair the sign convention.

An independent three-component counterexample uses the actual publication
fixture's original equation, reaction matrix and target:

```
F(q) = R q + 0.2 q^3 - forcing
R = [[2, .2, -.1], [.2, 1.8, .15], [-.1, .15, 2.2]]
target = [.15, .25, .18]
forcing = R target + 0.2 target^3
```

The field is spatially constant with homogeneous Neumann boundaries. Same-level
ghost copying, coarse restriction, quadratic coarse/fine interpolation and
reflux preserve constants, so every diffusion contribution is exactly zero
mathematically. This removes transport and ghost hypotheses while retaining
the genuine nonlinear coupled equation. Exact rational Gaussian elimination
gives a componentwise negative wrong-sign delta and positive forcing. Since
R delta = -forcing,

```
F(alpha delta) = -(1+alpha) forcing + 0.2 alpha^3 delta^3
```

every component becomes strictly more negative for every positive alpha.
Thus **any positive physical cell-measure norm** grows. The sign failure is
independent of rank ownership, cell resolution or component permutation.

The independent exact-Jacobian Newton replay fails all twenty allowed
backtracking steps at iteration1 under +F. With the corrected defect convention
-F, it converges under the same tolerance2e-9, twelve-iteration budget,
Armijo1e-4 and minimum_step1e-6. These are mathematical computations, not a
replacement native method or a claim that the FD/GMRES implementation is
received. Both component orders are exercised.

## Reduction, ownership and ghost review

The failed fixture has two genuine partitioned patches per level and partial
refinement at x=1/4 and x=3/4. Covered coarse cells are projected out of defects
and JVP images. The workspace and provider dots use their active masks and
level measures, with one physical contributor for a replicated level before
the final collective SUM. Local validation/launch/fence errors are converged
before that reduction in this original-field route.

The independent partition calculation uses the fixture's actual coarse/fine
cell counts at N16 and N32. Each MPI2 rank represents physical measure1/2;
the full weighted square norm equals the vector's square norm for Serial,
partitioned MPI2 and replicated storage with its noncontributor excluded.
None of these positive weights can change the uphill-direction certificate.

The provider prepares coefficients with peer halos winning over local extrusion;
each operator application copies the iterate into scalar storage, restricts
fine data, fills same-level/coarse-fine/physical ghosts, computes diffusion and
reflux, and excludes covered output. No separate transport or scalar-product
defect is demonstrated by this bounded source review. Their actual MPI/Kokkos
behavior still requires the repaired native rerun.

The appropriate repair is scoped to the physical-residual/workspace adapter:
negate the workspace defect after full physical evaluation, on every hierarchy
level and component, preserve +F JVP and preserve the independent +F final
recheck. Generic Krylov, reductions, equation bodies, capture ownership and
staged publication should keep their existing contracts. Euler owns that
production repair; this commit contains only probes and documentation.

## Reproducible checks

```sh
rtk proxy env -u PYTHONPATH /Users/romaindespoulain/miniforge3/envs/pops-api040/bin/python -m pytest -q tests/review/test_sol61_amr_original_newton_sign_probe.py
rtk proxy env -u PYTHONPATH /Users/romaindespoulain/miniforge3/envs/pops-api040/bin/python tests/review/sol61_amr_original_newton_sign_probe.py --checkout . --output docs/development/api_040/sol61_amr_original_newton_source_math_20260930.json
```

**17 source/math probes pass**. The standalone receipt reports
`sign_defect_detected`, hashes the actual source seams, records the twenty
uphill trials and the independent corrected-convention solution, and explicitly
sets native/MPI execution and saved-state reception to false. Ruff and
diff-check pass. It can be replayed against the author's frozen correction;
source sign compatibility alone will not qualify the actual nonlinear native
solve, CTest inventory, full AMR PDE, AMR convergence or GPU execution.
