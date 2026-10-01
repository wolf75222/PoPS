# Native public captured-diffusion fixtures

The two new installed-native tests are Uniform Dim2 mechanisms: one scalar field
and a permuted three-field product. They exercise the original physical problem

`-div(D0 * (1 + alpha) * grad(q)) + R*q + 0.2*q**3 = forcing`.

The Model declares these equations and the genuine field-to-source consumer
before the Program selects the frozen capture point and method. `Arithmetic@1`
is selected explicitly. D0 has negative off-diagonal entries and is nonsymmetric
and singular in the coupled case; the local reaction closes the system. The
test makes no SPD certificate or assertion that arbitrary finite D is solvable.
The scalar uses positive spatially varying diffusion. Both q and alpha vary in
space and the actual diffusion contribution is nonzero.

The load is manufactured by a test-only independent periodic Cartesian FV
stencil over an explicit nonconstant target. It does not call a PoPS operator,
the emitter, a native apply, or a solver. Each face uses the arithmetic mean of
the neighboring cell matrices. Source checks also compare the constant-D action
with the independent discrete Fourier symbol, verify telescoping conservative
flux, distinguish the variable coefficient from a uniform replacement by more
than 10,000 times the declared residual tolerance, and refuse a missing exact
material capture before emission. They cover N8/N16 math and actual public
Model/Case/Program resolution/emission for scalar/coupled3, including the signed
singular coefficient's finite-general admissibility. The native fixtures each
use N16; they are not a mesh-convergence campaign.

The seven Newton controls match the existing public original-field witness:
tolerance 1e-10, 20 Newton iterations, linear tolerance 1e-8, 240 linear
iterations, restart 60, Armijo 1e-4 and minimum step 1/1024. The central finite
difference step is 1e-6. These tests select no Uniform preconditioner. The new
MMS residual and solution guards are 3e-8, fixed before native execution to
permit double stencil accumulation and at most 300 times the relative Newton
tolerance. They are not an adjustment to a received result or an older guard.

The installed route is validate → resolve → collective compile-once → bind with
real component-first arrays → run. The native solution is published through an
auxiliary field into the physical source and integrated by the Program. Every
rank gathers accepted States, observed histories, carrier metadata, history
initialization/depth/fill/duration, diagnostics and lifecycle. Forcing/material
remain bit-exact throughout solve and replay. Each local assertion or file phase
converges before another native collective.

Fixture contract `pops.captured-diffusion-native-fixture@2` corrects the historical
fixture's history interpretation. `store_history(depth=1)` declares a maximum
lag of one and allocates two physical slots. End-of-step rotation places the
latest accepted store in slot 1; slot 0 contains the first accepted store in this
two-step witness. The Uniform sealed POPSAUX2 image records accepted auxiliary
owners, geometry, payloads and generations. Both history slots, their exact publication identities, outgoing
durations and fill counts (1 then 2) are gathered and checked. Each publication
has ordinal 1 within its own time window. The native checkpoint retains both
slots; saved solution arrays come from slot 1. The historical native failure
at the fixture's one-slot assertion remains a failed reception. No equation,
Newton control, finite-difference step or scientific tolerance is changed.
The first @2 reception passed the two-slot guards and then exposed another
fixture error: its carrier accessor belonged to AMR. The fixture now captures
the real Uniform auxiliary image and compares its bytes through restart/replay.
That failed reception is retained separately.

Accepted, continuous, restored and replayed arrays are saved to NPZ. Scientific
checks reload those actual saved arrays and independently recompute the original
residual; the accumulated source response is checked against the native observed
field. Checkpoint restoration and the next accepted replay must preserve all
gathered images exactly. The receipt records the actual artifact/platform,
native extension and every compiled block/Program DSO, their SHA256 values,
compile commands when retained (cache hits honestly report absence), exact
compiler-owned C++ sources, checkpoint hashes and phase NPZ hashes. JUnit records
artifact, dimension, rank, size and receipt paths. File reads use one fd's bounded
fstat extent and recheck its identity before hashing. No donor evidence is edited.

ROOT can receive these two installed cases after rebuilding the captured-D
production gel in all native dimensions:

```sh
env -u PYTHONPATH /Users/romaindespoulain/miniforge3/envs/pops-api040/bin/python -m pytest \
  tests/python/integration/runtime/test_public_captured_diffusion.py -m compiler \
  --junitxml=/ABS/FRESH/OUTPUT/junit.xml -v
```

For MPI2, launch the same command through ROOT's authenticated native launcher,
with one JUnit output per rank and external source/SDK/native owner pins. These
fixtures use the existing collective-directory and compile-publication helpers.
No native run, JIT, install, build or MPI execution was performed by the author;
only source/math/emission checks and Ruff/diff checks are received here. The
historical SDK375 AMR masks, budget and public-step failures remain separate.

Author reception on parent `3ceab3b719ad3ea316ddcf6b34b73396fce0e6d1`: the two
source/math/emission cases passed in 11.66 seconds with explicit source package
insertion; the two installed-native cases were deselected. Ruff and diff checks
passed. No native positivity or scientific output is invented in this receipt.
