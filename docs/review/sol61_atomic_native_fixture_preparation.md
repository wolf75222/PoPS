# Cubature Raw M17 - installed Native fixture preparation @1

This is Source-only preparation, not a runtime receipt. ROOT owns compilation,
Native Serial/MPI execution and external identity/qualification.

The fixture is tests/python/integration/runtime/test_atomic_cubature_public_path_runtime.py,
with two independent physical variants: conservative six-atom transport B=0 and
Raw Bx=rho I, By=-rho I/2. The latter uses no Fourier/free-streaming oracle. The
same public NormalizedPolynomialPath analytic_endpoint port composes physical
flux, matrix, path integral and endpoint speed. No Core opcode recognizes atoms,
component names, density slot1 or these equations.

Cell grid8x4 spans2x1, periodic, single-level synchronous AMR. ForwardEuler has
fixed dt1e-4 and two accepted steps. Every atom's initial weight is
.6+.05k+.03*sinc(1/8)*sin(2pi*x/Lx+k/7)+.02*sinc(1/4)*cos(2pi*y/Ly-k/5).
These are true exact cell averages of positive sinusoidal weights, not point
samples; the Source adversary proves a difference above1e-3. The physical moment
components are the degree-two monomials in the explicit permuted order. Positive
independent x/y atoms guarantee SPD in the initial measure. No repair is added.

The independent offline oracle uses direct fixed six-atom moment identities,
separate from AtomicCubature inversion and the arithmetic DAG. For each positive
coordinate face, it calculates the Rusanov flux and exact Raw path integral;
incident-cell residuals are -I/2 on both sides. A cell receives negative flux
difference plus both incident residuals divided by spacing. These signs agree
with the common generic API contract path_flux.hpp:63 and Cartesian path residual
assembly:448; the oracle does not import or execute either implementation.
The max endpoint speed is1+abs(directionfactor)*max(rhoL,rhoR), as proved in the
public case. This matches the authored bound; no sharper speed is silently used.
Periodic B=0 preserves each component sum; B active is independently evolved by
this actual nonconservative FV law. Source tests distinguish B active from B=0.
The comparison allowance rtol2e-12/atol2e-13 is a floating-point cross-language
FV comparison, not a changed physics or nonlinear solver tolerance.

The genuine fixture requires installed package under sys.prefix, uses existing
MPI compile-once/collective helpers, and runs compile->bind->two steps. It retains
initial/accepted/continuous native arrays, full grown POPSCAR1 bytes and clocks
before numerical assertions. Saved arrays are reopened; exact valid bits are
cross-checked against fullcarrier coverage/geometry/components; retained ghosts
are not independently scientifically qualified. Program C++ must already exist
(no dump fallback), Program IR and compiled artifact manifest are retained and
hashed, along with compiled DSO identity and actual native extension. ROOT must
supply immutable before/after installation/source/header/MPI identities externally.
No replacement receipt/seal is authored by this preparation.

Scope excludes multilevel/reflux/regrid/subcycle/restart, independent whole-grown
Ghost formula, Field/aux dependencies, GPU, full Fan–Li, performance and mission94.
No invalid-SPD rollback case is claimed; negative density refusal is covered only
by the earlier actual-header witness. No Native calls or JIT ran during preparation.

Source preparation command: read-only ir17 Python with env -u PYTHONPATH,
PYTHONDONTWRITEBYTECODE=1, --noconftest -p no:cacheprovider -o 'pythonpath=python .',
 tests/review/test_sol61_atomic_native_preparation.py plus prior case/reference tests.
Native nodes were collected only:2 nodes (B0/Bactive), zero runtime executions.
Future ROOT command uses the installed rebuilt SDK with genuine repository native
fixtures and the same two nodes, then MPI2 as a separate actual campaign.

Final coherent Source/actual-header results:12PASS31.97s, zero skips. The synthetic
codec test is explicitly synthetic and checks saved-gather/carrier bit linkage,
valid-cell poisoning and truncation refusal; it creates no Native receipt.
Native collection:2 nodes0.94s, no test execution.

## Corrective capture delta (before any Native execution)

The public compile result is the actual CompiledSimulationArtifact wrapper, not
CompiledProblem. The fixture now authenticates artifact.verify(), the exact
population LayoutPlan assignment, singleton layout coverage, CompiledLayoutProgram
identity/target/partition, and its program pointer. Retained C++/IR/DSO details
come from that exact selected program; no first-entry or regeneration fallback.
The wrapper remains the authority for the aggregate artifact manifest.

Program provenance is saved before bind. Each real initial/accepted/continuous
capture is saved immediately, before another run or science assertion. The receipt
records partial-captures, native-run-failed, captures-complete or fixture-guards-passed;
capture-time phase records explicitly say science assertions have not yet run.
On a run error, collective failure diagnostics are written before another native
operation, rollback capture is attempted consistently, and the original local
exception object/type is re-raised. Peers without a local exception report the
actual collective diagnostics. Capture/I/O failures cannot replace the original
run exception. This helper cannot repair a deadlock inside a native collective.

New tests exercise the genuine Source wrapper and actual layout-program classes
with repository metadata-only components. Only platform metadata lookup is isolated
(no selected SDK); no fake runtime or Native receipt results. A synthetic Source
callback failure exercises the actual capture orchestration helper and proves
accepted files survive, failure is written before rollback capture, and the original
exception object survives. It does not qualify Native rollback.

Final affected cohort:15PASS30.38s, zero skips. Native collect-only:2 nodes0.89s.
No Native/JIT/ENV execution or mutation occurred. Earlier preparation bd1c3a59
remains historical; these tests are a correction required before Root execution.
