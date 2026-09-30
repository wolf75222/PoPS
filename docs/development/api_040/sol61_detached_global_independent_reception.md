# Independent detached physical Global-source reception

The immutable author candidate reviewed first was `2fbf709a57e8df1192279bf6dab669636b048edc`, with parent `368055dbe1f4c1f5fad4a11791508ae2b0520f03`. Review work occurs in the exclusive `PoPS-sol61-global-ownership-review` checkout. Production, MAIN, installed package, SDK, shared cache and native environment are unchanged by this review.

## Counterexamples and bounds

The independent model declares three physical components `(z,a,k)`, the dimensionless Global port `charge`, and the physical source before any temporal Program. Its source is `S_c=-0.47 Q U_c`, either directly or through the public primitive recipe `loss=0.47 Q; S_c=-loss U_c`. A five-by-three Cartesian periodic layout and a public explicit Program exercise the actual resolution/lowering/emission pipeline.

On the parent, the actual public `pops.compile(resolve(validate(case)))` path reaches the substituted external compiler for the legacy model. For the physical Global model, it fails with `MissingOwnershipError`, at `BlockHandle.__getitem__`, because `bind_source_globals` tries to qualify a port through the detached registry. On `2fbf709a`, both models reach the actual generated `problem.cpp` compiler seam. Only bootstrap/native selector/dependency checks/toolchain access are substituted, and no emitted artifact is loaded. Placeholder bytes from that seam are explicitly not native results.

The new independent counterexample on `2fbf709a` changes only the lowered `prim_defs['loss']` to twice its original recipe, after successful baseline detachment/emission. The source body and physical Module hash remain unchanged. Emission succeeds and its C++ differs, violating the original physical source. The guard authenticates `_source_terms` but expansion reads the unauthenticated lowered primitive map. The regression test requires refusal and is red on that candidate. This is a genuine physical-body mutation; it neither replaces a saved native result nor substitutes a scientific oracle.

The live plan deliberately retains its Program and Model while it exists; its Case can be collected. Once the detached Program is the sole retained result, true weak references to the original Case, Program and Model are all cleared after `gc.collect()`. Its block registry is `None`, and its prepared authority remains valid. This demonstrates detached ownership, not early disposal of the live resolved plan.

The immutable correction received is `f7378be89e10de170b37f7158bb91031c754829b`. Its authority helper reconstructs the expected lowering from the authenticated physical Module and exact StateSpace, rather than minting a new proof from the mutated primitive map. The same primitive counterexample now refuses before kernel emission. Independent mutations of a runtime parameter default, an equal cloned parameter declaration handle, an extra parameter that shifts native slots, and component ordering also refuse. An unread finite constant recipe and an unrelated lowering cache preserve emitted bytes exactly.

## Independent checks

Final source reception on `f7378be8`: **87 passed**, comprising **25 independent probes** and the affected author source tests; four nonportable static author provenance goldens were deselected after the five-failure diagnostic run explained above. Those tests and their goldens are unchanged. Ruff and whitespace checks pass. No new unresolved physical-source contradiction remains in this bounded review.

`tests/review/test_sol61_detached_global_authority.py` uses its own model/Program fixture and toolchain sentinel; it imports no author test helper. It covers public compilation with and without Global, actual Model/Case/Program collection, a mutated primitive recipe, six detached row/input/point/unit/version/clone mutations, replay of a proof into an equal foreign Program, original Module and lowered source body mutation, five live-plan mutations followed by re-sealing, and parent serialization parity. A port issued by another independently built Case with equal canonical metadata is also refused, even when the Program IR remains unchanged. Semantic failures are checked as `ValueError`/`TypeError` with relevant diagnostics. Unexpected `RuntimeError`/`OverflowError` are not masked.

`tests/review/sol61_detached_global_parity.py` computes receipts from the real Python package in the selected checkout. The same independent fixtures on parent and candidate produce equal Program IR, resolved-plan identity, physical Module hash, ModuleManifest serialization digest and generated C++ digest for all three cases (15 equal entries). `sol61_detached_global_independent_parity.json` preserves the parent values and exact pinning. A legacy Manifest 10 case and physical Global Manifest 11 cases participate, including a primitive-backed physical source.

The four author static plan/Manifest goldens are not portable between checkout paths. In the actual parent/candidate payload comparison, only `snapshot_artifact_hash` differs; the captured physical manifests differ only in operator provenance absolute file paths and their provenance digests. The native catalog is identical (`8797c3721e51b181711c0d4d832688b5f32af7d184ce28b81422815010b465c3`). Adding lines before this independent fixture likewise changes its recorded source provenance. The first immutable parity receipt is preserved, not silently updated. The independent test compares every one of the five receipt fields against the exact parent in a pristine process using the same fixture path/lines, and separately verifies the initial physical IR/Module/C++ fingerprints. No provenance field is normalized or omitted from the parent/candidate comparison.

The parent interpreter is authenticated at exact SHA `368055dbe1f4c1f5fad4a11791508ae2b0520f03`. Set `SOL61_GLOBAL_PARENT_CHECKOUT` if its private worktree is elsewhere; the default is sibling `PoPS-sol61-global-ownership-parent-review`. This comparison requires the private source checkout and fails if the parent pin is wrong.

## Reproduction

Use the source interpreter without native installation:

```sh
env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 /Users/romaindespoulain/miniforge3/envs/pops-api040/bin/python -c 'import sys; sys.path.insert(0,"python"); import pops,pytest; print(pops.__file__); raise SystemExit(pytest.main(["tests/review/test_sol61_detached_global_authority.py","-q","-p","no:cacheprovider"]))'
```

Run from the review checkout. The receipt runner accepts an absolute checkout path; its source package path is explicitly chosen before importing PoPS. The public compiler probe stops at its first actual problem compilation, so no native problem, Kokkos kernel, MPI consumer or physical evolution is executed by this review.

Native serial/MPI/GPU execution, numerical evolution, exterior integral feedback, restart rollback and M14 physical qualification remain outside this source-only reception. Existing unsupported direct Equation/FieldProblem/implicit/flux routes are not promoted to supported by this ownership patch.

The coherent affected suite adds the four unit files `test_resolved_physical_global_authority.py`, `test_physical_global_primitive_authority.py`, `test_physical_global_source.py`, and `test_integral_candidate_capture.py` to the independent pytest command and selects `-k 'not test_detached_emission_preserves_exact_parent_ir_manifest_and_cpp'`. The independent strict fresh parent/candidate comparison covers the omitted static provenance comparison under one exact shared call site. The four author CPP/IR/Manifest/plan receipts were not rewritten or bypassed in production.
