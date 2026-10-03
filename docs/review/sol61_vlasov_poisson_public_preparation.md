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

The full public Source preparation depends on the separately owned coherent
Module.frame and source-only compiler admission changes. The frozen 6a77 API has
no Module.frame and refuses the explicit transport-boundary frame. That failure
is retained; no false positive Source/Native result is reported. The separate
SSA fix a8d478f1 preserves captured equations when a mapped RHS is later named at
another point, with 45 affected Source tests passing and genuine drift rejected.
