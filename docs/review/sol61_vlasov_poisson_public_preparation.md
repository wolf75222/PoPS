# Finite-velocity Vlasov–Poisson public preparation @1

This is a new M19 representative, not a thermal equation renamed. It declares
`∂t f + ∂x(v f) + ∂v(E f) = 0`, `-∂xx φ = ∫f dv - 1`, and `E=-∂xφ`.
Position is periodic on [0,1]; velocity is truncated to [-1,1] with zero boundary
flux. The Poisson gauge is mean zero. This does not qualify infinite-velocity
Vlasov, Landau damping, BGK, convergence, or a universal PDE implementation.

The public script declares independent phase, reduced-number, and spectator
owners. Native axes are velocity then position; physical storage has position on
axis zero. The phase State is not the first block. PhysicalSupportMap performs the
velocity quadrature and the gradient lift. A genuine FieldProblem solves Poisson
at both stages of SSPRK2. The reduced-number accepted State retains the last
predictor moment; it is not advertised as the accepted endpoint Field cache.

The independent NumPy reference uses the discrete periodic Poisson matrix plus
an explicit gauge constraint, central gradient, first-order upwind flux with the
specified velocity boundary, and two SSPRK2 steps of 1/128. It refuses incompatible
mean sources rather than neutralizing them. Four velocity cells give exact dyadic
cell averages of 3/4*(1-v²) with quadrature mass one. The initial spatial profile is
nonconstant, positive and exactly neutral. The reference checks positivity and
mass, and discriminates reversed force and a stale predictor Field. Three pure
Source tests passed; they are neither a substitute runtime nor Native evidence.

The prospective Native fixture requires the installed package, generated current
ABI and mapped-output capability. C25 ALL Model/Program evidence is retained
before bind. Preparation is durable before bind. Every public valid getter is
saved immediately, followed by a public composite checkpoint and typed clocks,
owner boxes and consumer cursors. Failure ledgers preserve the original exception.
One Native node was collected only. Full-grown authority requires later independent
inspection of the actual composite checkpoint. No ghost formula is received here.

A full Field potential image cannot currently be requested through the composite
public executor: RuntimeInstance delegates field_potential_global to an executor
method absent on MultiLayoutExecutor. This fixture does not invent a private
owner lookup or refresh the solver through another accessor. Its future state
comparison discriminates the consumed force; the oracle's Poisson residual does
not certify a saved Native Field residual.

The separately owned Module.frame and source-only compiler admission changes are
consumed coherently. Earlier frozen-API, competing-publication and frozen-claim
REDs remain retained separately. Publisher commits 582a4285 and 851a4b90 preserve
one physical producer while authenticating each stage occurrence. The unchanged
VP resolve→slice→detach→Program emission now passes. Seven additional independent
checks derive claims from this actual resolved VP, refuse foreign producer,
unknown and ports or competing same-point invocation, and emit all three Model
bricks. The final affected cohort passed 37 tests in 13.07 seconds; the emitted
C++ is Source evidence, not a successful C++ compiler or Native run.

Flux() selects the uniquely authored default physical flux for the configured
HLL finite-volume realization. Flux(handle) is a distinct named centered-divergence
route and correctly refuses that FV realization; the prior refusal is preserved.
The velocity-coordinate auxiliary is explicitly listed in the grid operator's
public requirements, in addition to the FieldSpace signature. This uses the
existing exact provider-pack contract, without changing the physical formula.

The SSA fix a8d478f1 separately preserves captured equations when a mapped RHS is
later named at another point, with 45 affected Source tests passing and genuine
drift rejected. Its original equation-identity guard remains unchanged.

Source command (env -u PYTHONPATH, ir17 Python, --noconftest -p no:cacheprovider,
absolute own-checkout pythonpaths): test_sol61_m19_vlasov_poisson_preparation.py,
test_sol61_m19_vlasov_poisson_oracle.py,
test_sol61_value_materialization_immutable.py,
test_sol61_module_physical_frame.py, test_sol61_source_only_module_frame.py,
test_sol61_mapped_publication_occurrences.py and
test_sol61_vp_publication_nonauthor.py, all under tests/review.
