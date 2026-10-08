# Physical globals across the resolved compiler boundary - 30 September 2026

Base: `368055dbe1f4c1f5fad4a11791508ae2b0520f03`.
Private checkout: `PoPS-sol61-global-ownership`.

The installed SDK20d reception reports eight passing cases and three physical-global
failures before compilation in
`outputs/installed-original-field-physical-global-feedback-dim2-sdk20d-20260930/pytest.log`.
`bind_source_globals` consulted the live Case registry after the compiler had
deliberately detached that registry. The source tests previously emitted the live
Program directly, so they did not exercise this cut.

The new public `pops.compile(resolved_plan)` regression reproduces the exact
`MissingOwnershipError` against a Python source archive of the parent. Its legacy
q*S case reaches the problem compiler, while its physical-global case fails.
Only native selection/bootstrap and compiler execution are replaced in this source
test; real resolution, per-block lowering, detachment and complete Program emission
execute. The test stops at the problem compiler seam; its temporary placeholder
model binaries provide no native execution or artifact qualification.

The resolved plan now prepares a private immutable proof while the real registry
can authenticate each exact issued port. It records the existing IR hash, exact
node/input/capture identities, canonical port/unit data and authoritative Module
hashes. Plan verification repeats live port authentication and checks that proof,
including after a public plan identity has been recomputed. The proof is excluded
from the serialized plan payload because it authenticates existing semantic data;
the serialized contracts and identities are unchanged.

Detachment transports this proof into an immutable binding to the exact clone-owned
source nodes and ports. It does not restore a registry, infer authority from equal
metadata, or keep the Case/source Program alive. A detached clone, missing binding,
changed capture/input/point/scope/unit/version or changed Module body is refused.
The emitter also compares the actual lowered source AST against the state-bound
registered Module body, before primitive expansion and global substitution, so
altering the emitted expression alone cannot retain an unchanged Module identity.
Ordinary live source emission keeps its registry-issued-port check.

The public Model GC regression releases both the source Case and source Program
while retaining a usable detached proof. Re-detachment preserves the proof and
the exact IR identity. Arbitrarily setting the detached marker or deleting its
binding cannot mint authority from metadata.

Validation uses the `pops-api040` Python executable with `PYTHONPATH=python` for
source identity; the shared environment is not installed into or modified.
The affected source selection passes 69 tests, including 19 new probes. It covers public compilation, clone/reseal attacks, original
captures, physical source manifests, lowering, GC, and exact parent parity.
The frozen parent receipt `sol61_global_authority_parent_parity.json` pins four
IR/plan/Module-manifest tuples (legacy manifest 10 and physical-global manifest 11)
and eight complete Uniform/AMR C++ sources. Candidate detached emission preserves
those hashes. Both full three-component generated Programs pass Dim2 C++ syntax
against the real Kokkos/MPI headers with `/usr/bin/clang++ -fsyntax-only`.
Ruff passes for new files, and for all touched files with only the existing
`program_global_sources.py` compact-statement rules E701/E702 excluded. Diff-check passes.

The additional architecture/sparse/nested/graph selection passes 22 tests. Five
tests in `test_compiled_program_detach.py` fail identically on the archived parent:
their historical `_Model` test double lacks `primitive_recipes`, and fails during
source authoring before reaching detachment. This correction does not change those
fixtures or weaken the physical-source authoring protocol.

The new private proof is a compiler capability, not a persisted checkpoint codec
or a new public physical-global protocol. No C++ header, SDK, installed package,
main checkout, numerical tolerance, original source equation, or native fixture
is changed. Installed serial/MPI reception remains root-owned and pending for
this corrected Python implementation.
