# Independent review: CoordinatedFace implementation

Scope: author checkout `PoPS-principal-group`, feature commit `e0823a5`,
generic AMR source-route commit `5cf2788` and typed coordinated AMR commit
`f371e2e`; independent tests and oracle in `PoPS-degenerate-diffusion`
through `df85e2f`. No installed/native PoPS execution is inferred from
source or host probes.

## Contract and source findings

The positive coordinate orientation matches the existing path residual:
`-(G_upper-G_lower)/dx + (right_lower+left_upper)/dx`. M07 maps the
hydrostatic corrections to `left=-C_L`, `right=+C_R`. The new face returns one
`PathInterfaceResult` containing the shared flux, both already signed sides,
one speed and status. `MaterializePathFace` writes all four to scratch and
copies none to the output until every face status succeeds. The residual and
AMR path workspace retain the original side fields separately from the
conservative reflux flux. Conservative components require literal zero sides.

The exact physical rate must contain one retained flux and one retained
nonconservative product. The numerical method identity includes the face
body, law, captures, signs and interface version. The Uniform route compares
`coordinated_face:v1:<model operator identity>` before constructing the distinct
`CoordinatedFaceFlux`. It does not call the legacy symmetric
`PathRusanovFlux`. The typed AMR installer compares the same full token and
dispatches `ModelPathFlux`; legacy paths continue through `kRusanov`. Its
syntax preflight checks the versioned digest shape, then the typed installer
compares that digest with the compiled model. The author's three extracted
AMR C++ host tests passed 3/3, including foreign digest, malformed version,
reconstruction and positivity-floor negatives plus existing User routes. A
generated System/AMR M07 preparer TU passed syntax-only with project flags;
neither check is an installed AMR run.

One concrete compatibility defect was found during review: a new composite
model version member returned zero for legacy paths, while the first trait
asserted one whenever the member existed. The author corrected the trait to
interpret version zero as legacy, version one as coordinated and other values
as unsupported. The author reports a real generated-composite C++ probe that
compiles and executes both branches (8/8 source/host tests at that point).

The compiler authenticates references and evaluates active intermediate
expressions through the common guarded CSE path. RuntimeParam owner resolution
was corrected after the author's positive/foreign capture test exposed an
authored-versus-resolved registry mismatch. I confirmed the corrected source
test checks an emitted `params.get` for the local parameter and rejection of
the foreign owner; native rebind is still an unexecuted independent test.

## Structurally different scientific witness

The independent system is `U=(c,p,r)`, `F=(c,p/2,-r)`,
`B[p,c]=0.7r`, `B[r,p]=-0.4c`. It is genuinely nonintegrable in general:
the closed `(c,r)` rectangle has a nonzero `p` path integral. Its exact
straight-path product integral is
`(0, 0.7*(rL+rR)*(cR-cL)/2,
-0.4*(cL+cR)*(pR-pL)/2)`. The numerical face splits it into signed
30/70 incident-cell sources; a legacy 50/50 path dispatch cannot match the
oracle. The third system differs in state roles, closure and side split from
the Saint-Venant lake witness.

Pure/oracle plus public-source tests: 12/12 passed in 5.34 s against the
author checkout. They compare exact IR values to a separate NumPy oracle,
check two component permutations, actual `validate -> resolve -> emit` in 1D
and 2D, reject nonzero conservative sides and a same-name foreign `Var`, and
prove a face-only RuntimeParam survives owner resolution. Three native tests
were collected, not executed: two permutations through eight FE steps on a
2D y-invariant grid and one same-artifact two-bind test (`margin=.5,1.5`).
They compare complete states and transverse invariance to the independent FV
oracle. The bound used by this witness is conservative for the chosen small
coefficients, but no general stability theorem is claimed.

An independent scalar counterexample shows why a bound based only on the
physical Jacobian is insufficient: for `F(u)=-u`, `B=+1`, the physical
`DF+B` vanishes, yet a 30/70 split with zero numerical speed amplifies a
checkerboard by 8% in one step at `dt=.1 dx`. The public contract correctly
requires the author's bound to cover the complete numerical update.

## Acceptance boundary

Source/body/identity review is favorable on `e0823a5` and `f371e2e` after
the author's C++ host and syntax checks. The independent three-state native tests, M07
installed Dim1 campaign, AMR reflux, MPI failure convergence, GPU and artifact
authentication remain open until rebuilt and run by the central integrator.
The author has not claimed these results from the source delivery.
