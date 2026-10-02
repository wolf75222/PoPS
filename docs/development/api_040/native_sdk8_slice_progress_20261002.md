# SDK8 source snapshot before native construction — 2026-10-02

This historical Source snapshot and its JSON remain unchanged in scope. The
later [actual SDK8 builds and bind results](sdk8_native_build_binding_progress_20261002.md)
record completed construction, one installed typed Newton pass, five C++ MPI
collective passes and two distinct failed public Tag binds. Full scientific
Halo reception remains pending.

The integrated source contains the conservative cell-average projection correction,
the public signed free-transport fixture and the independently reviewed spatial
reader @3. The next native package has **not** been built or received. The
machine-readable [progress receipt](native_sdk8_slice_progress_20261002.json)
records actual checks and preserves the earlier failed native attempt.

The projection uses the existing Gauss nodes and weights, with a progressively
weighted convex mean. Constant samples retain their exact value, including signed
zero, and opposite finite large samples do not overflow a subtraction. The real
repository C++ target has 12 passing tests; the integrated independent host
counter-tests have 3 passes. This closes the source defect that changed constant
one by one ULP. The separate accepted-halo preparation is still under review;
full-grown native tag-selection reception is required after rebuilding.

The integrated M19 source selection has 47 passes. Its future native fixture
uses `ddt(f) = -div((0, v*f))`, a genuine analytic velocity coordinate, HLL,
SSPRK2, conservative projection, independent continuum/discrete Fourier checks
and exact checkpoint restart. It does not receive full BGK or Vlasov kinetics.

The first actual nonconstant AMR N8 SDKbb416 run remains failed: one failure in
222.1134348330088 seconds, at the old raw-relative guard. Its saved native
original norm is `4.2112344066743426e-11`, reference norm
`0.3630691185175441`, raw ratio `1.1598988159250038e-10`. The source-selected
rule was already `norm <= tolerance * max(1, reference)`, with tolerance
`1e-10`. Reader profile
`nonconstant-original-composite-Q-tag-selection-periodic-strip@3` corrects
that normalization mismatch; it preserves all equations, seven Newton controls,
FD step, timestep and existing Q/scientific guards. It also requires an
independently reconstructed original weighted norm at most `1e-10`, exact
diagnostic ratio consistency, authenticated zero seed, and agreement of the
saved final norm with a bound derived from the real IR and geometry. Artificial
zero and in-range false norms are refused. The integrated pure offline suite
has 90 passes, with no PoPS import. No old failed attempt is resealed as received.

The fresh environment `pops-api040-halo9` was prepared by the repository clone
driver: 64 dependency packages match, and all 1,132 source SDK files plus the
native artifact in `pops-api040-ir17` remain unchanged. The new prefix initially
contains the copied historical SDK; it is not the rebuilt SDK8. Source tests
and offline archive readers remain distinct from installed native execution.

The typed Newton cutoff overflow and accepted-halo Field producer effects are
still completing independent reviews. ROOT owns their integration, the central
ABI8 generation, native compilation and subsequent Serial/MPI campaigns. The
unchanged AMR12 carrier codec and the legacy accepted-state contract remain
separately versioned. GPU, ROMEO, other dimensions and GitHub CI have no new
qualification from this source slice.
