# M19 free streaming: source receipt, native reception pending

This tranche declares the original linear kinetic transport

\[
\partial_t f+v\partial_x f=0,\qquad x\in[0,1]\text{ periodic},\quad v\in[-1,1],
\qquad f_0(x,v)=(1+0.1\cos(2\pi x))(1+0.25v).
\]

It extends the previous finite ProductDomain Reduce/Lift witnesses with actual nonzero transport. It does not receive Vlasov–Poisson, BGK, self-consistent acceleration, or full kinetic coupling. Those equations need their own mechanisms and native scientific receptions.

## Exact implementation and bounded correction

Private base: `b782cfebd844b24fca4c6e49b07bbdd547e04ff1`. Production fix: `ce6ef7b24b0586ef148e4a0b955ae0e93326c6d0`, one file `python/pops/model/provider_pack.py`, eight added lines. No header, serialization, ABI, legacy IR schema, or native provider was changed.

The public call `RATE(f.n, P.input_fields(f.n, for_rate=RATE))` issues a real empty `FieldSpace` token at the same SSA point. The declared empty space owns no storage components. Before the fix, `build_operator_provider_pack` tried to select an absent `('field','fields')` storage provider and refused the complete public Case. The fix skips that storage lookup only when the declaration exists, has zero components, and the exact concrete type and `to_data()` image agree. Missing, nonempty, foreign-frame and foreign-layout declarations continue through strict selection and refuse. The physical flux still requires and prepares its exact `velocity_coordinate` auxiliary provider. No Module bookkeeping, hidden field, state read, or fallback slot is introduced.

`tests/python/support/m19_freestreaming.py` declares physics at module scope. The existing Rectangle Cartesian chart realizes the named product support with native axis 0 = velocity and native axis 1 = position (the Cartesian direction labels are x/y respectively). A typed `AnalyticAux` evaluates the actual physical velocity coordinate; no velocity table or per-nv model is generated. The flux is `(0,v*f)` and signed HLL wave bounds are `(v,v)` in the position direction, `(0,0)` in the velocity direction. Zero velocity flux and periodic position are explicit boundaries. The velocity diagnostic `AxisQuadrature` is a separate declared cell measure, not the transport coefficient.

Discretization: conservative cell averages, FirstOrder reconstruction, HLL with explicit signed bounds. Time: SSPRK2 with true RHS evaluations at c=0 and c=1, fixed exact binary `dt=1/(4 nx)`. Realization: the existing C++/Kokkos/MPI Uniform transport and auxiliary providers. Acceptance is in the installed-native fixture, not in the physical model.

## Source and mathematical evidence

Command (source imports only; no selected Native module):

```sh
rtk proxy env -u PYTHONPATH -u POPS_NATIVE_DIM PYTHONPATH=python /Users/romaindespoulain/miniforge3/envs/pops-api040-ir17/bin/python -m pytest -q tests/review/test_sol61_m19_freestreaming_source.py tests/python/unit/codegen/test_component_provider_pack.py tests/python/unit/codegen/test_program_input_fields.py tests/python/unit/codegen/test_analytic_aux.py tests/python/unit/codegen/test_auxiliary_producers.py tests/python/unit/codegen/test_projection_auxiliary_preparation.py tests/python/unit/codegen/test_wave_speed_provider_manifest.py tests/python/unit/codegen/test_emitter_provider_pack_bind.py tests/python/unit/codegen/test_projection_provider_signature.py
```

Result: **89 PASS, 11.86s**. Both public cases `(nx,nv)=(32,8),(64,12)` undergo actual validate/resolve/Program emission. The generated Program prepares the exact auxiliary before both negative-divergence calls. The real model emitter contains the coordinate producer, velocity-dependent physical flux and signed characteristic speeds. Source negatives retain missing/changed field and auxiliary refusals. The integration fixture collects exactly two native cases; it was not executed by this agent.

A fresh Python-source archive of parent b782, with the same public physical helper, reproduces `MissingInputProvider: [('field','fields')]`. The fixed source emits 13,688 bytes and two provider preparations for 32×8. A genuine nonempty public Module/FieldSpace/provider pack has identical JSON SHA256 on both source versions: `5a242920f1be3528f70a1a60c0f202109f85d9a382f1810867c69d5c64ae7e89`. This is a bounded pack compatibility check, not all-program byte parity.

`tests/review/sol61_m19_freestreaming_oracle.py` imports only NumPy/stdlib. It independently computes the finite-volume Fourier eigenvalue

\[
\lambda_j=-|v_j|n_x(1-e^{-i\,\operatorname{sgn}(v_j)2\pi/n_x}),
\quad G_j=1+\Delta t\lambda_j+\tfrac12(\Delta t\lambda_j)^2,
\]

and the original continuum cell integral of `f0(x-vt,v)` in both coordinates. Differentiating sinc gives the velocity-weighted cosine integral; an independent 32-point Gauss tensor integral receives that expression. Native initialization uses the existing conservative analytic quadrature; no initial array is bound beside the Case InitialConditionPlan.

The oracle checks maximum discrete error ≤3e−12, mean continuum absolute error ≤`0.65*t/nx`, actual evolution ≥0.005, and mass plus the first two velocity moments conserved within 3e−12. These are new declared witness guards, not lowered historical limits. Reference-only continuum L1 at t=1/8 is 0.0022893721705324927 for 32×8 and 0.0011663089621362962 for 64×12; the refinement order is approximately 0.973. A frozen population, wrong velocity sign and ForwardEuler amplification each fail the discrete-equation guard. Mathematical reference arrays are explicitly synthetic source checks, never native artifacts.

## Obligations for ROOT native reception

Run `tests/python/integration/runtime/test_m19_freestreaming_runtime.py` against the rebuilt installed package, Serial and MPI2. The fixture requires an installed import, genuine compile/bind/run, the sole declared InitialConditionPlan, actual auxiliary velocity values after evolution, and unchanged original equations. The current reception profile uses binary64 and Uniform dimension 2; this is not a production precision/geometry restriction.

Each case saves five actual NPZ phases: initial, accepted at 1/8, continuous at 1/4, restored at 1/8, replay at 1/4. Each phase has a separate native checkpoint and phase receipt. Checkpoint hashes are taken immediately and checked again after all observations. State/checkpoint filenames cannot overwrite each other. Initial is verified against the conservative analytic seed. Accepted/continuous/replay are verified against both independent equations. Restore and replay require exact state bytes, clocks and macro steps. The auxiliary is observed only after actual materialization; no initial auxiliary value is invented.

Root-written receipt `pops.m19-freestreaming-native-fixture@1` captures the actual native DSO, package entry, exact compiled Program CPP and IR from the same handle, actual compile-frozen model manifest, both component binaries and sidecars, frame and explicit quadrature, artifact/bind identity, original source hashes, rank-owned boxes from every rank and every phase. Source/result dtype and all physical constants are saved. Program CPP is never re-emitted to replace missing compiler provenance. The existing model driver may retain no translation unit; `model_sources.cpp=null` explicitly records that provenance gap, while an available actual retained model CPP is saved. No model CPP is synthesized from the manifest, helper, or receipt. Coordinates and cell measures used by the mathematical oracle come from the declared authenticated Cartesian grid; the actual velocity provider is independently observed. This fixture does not claim a raw per-cell geometry snapshot or recomposition of the aggregate compiled graph payload.

All rank operations and local checks converge through the existing collective helpers; the shared directory is elected before writes. Only rank0 writes observations/provenance. Physical ownership is recorded from `local_boxes` for all ranks and is not inferred from the proof writer. Retained binary origins are recorded for ROOT's immutable portable snapshot; no external approval is minted here. A complete scientific offline reception must receive exact raw JUnit, native/package/source identities and immutable external ROOT pins/approval, then re-evaluate the real saved arrays. Old finite-product M19 seals cannot qualify this PDE tranche.

No native execution, JIT, MPI run, installed-environment mutation, or heavy compile occurred in this author worktree. ROOT owns those future receipts. No claim is made for AMR transport, GPU, full Vlasov/BGK, or CI.


## Current integration on 05159edd

This integration reuses ce6ef7b24, de0b6a590 and de481435; the original author
worktree remains clean and unchanged at de0b6a590. No physics or acceptance guard
was reimplemented. The only production delta is eight additions in
model/provider_pack.py, requiring exact type and to_data equality with the
registered empty FieldSpace. Missing, stale, nonempty and subclass alias inputs
still fail strict lookup, and actual missing auxiliary requirements remain errors.

Current public validate/resolve and C++ source emission pass both mathematical
cases 32×8 and64×12 (native axes v,x). Program IR is5 for both; no contract version
change is necessary. The Native fixture schema remains@1 because its durable
receipt semantics are unchanged. The new machine-readable
m19_current_source_preparation_sol61.json is explicitly SOURCE_ONLY and records
actual current emitted C++ hashes and reference-only mathematical metrics.
No compile, bind, run, JIT or native build was performed for this integration.
The intended execution is the existing Host/Kokkos Uniform realization; this
preparation establishes neither BGK/Vlasov collisions nor GPU execution.

Source reproduction uses the existing pops environment without setup or ENV
mutation, and an explicit private WT python path (env -u PYTHONPATH):

```sh
rtk proxy env -u PYTHONPATH /Users/romaindespoulain/miniforge3/envs/pops/bin/python - <<'PY'
import sys
from pathlib import Path
sys.path.insert(0,str(Path('python').resolve()))
import pops,pytest
assert Path(pops.__file__).resolve()==Path('python/pops/__init__.py').resolve()
assert not list(Path('python/pops').rglob('_pops*.so'))
raise SystemExit(pytest.main(['-q','tests/review/test_sol61_m19_freestreaming_source.py','tests/python/unit/codegen/test_component_provider_pack.py','tests/python/unit/codegen/test_program_input_fields.py']))
PY
```

ROOT's future actual Native nodes are
`tests/python/integration/runtime/test_m19_freestreaming_runtime.py::test_native_m19_signed_freestreaming_exact_restart` (two cases).
They require the installed rebuilt package and select Dim2; Source collection
alone does not satisfy their state, auxiliary, clocks, exact full-payload restart
or continuum/discrete transport assertions. Model translation units missing from
actual components are recorded as missing, never fabricated by re-emission.


The three origin gels above were applied together with `cherry-pick --no-commit`
in this private worktree; they are already present in the index, and are not
cherry-picked again. The integration commit therefore carries their combined
eight-line production delta, fixture, checkpoint counterreview and current Source
preparation together. The machine receipt records the original base and origin SHAs.

Its IR5/C++ hashes are a historical Source measurement. After Hooke primary export,
Newton typed convergence/ABI8 and the next SDK integration, ROOT must remeasure
actual emission and package identity before a new Native campaign. The tests retain
the mathematical/public-route guards but deliberately do not require future emitter
bytes to equal this old receipt. No prior SDK7 reception elsewhere in the repository
qualifies this new source integration. Full BGK/Vlasov and GPU qualification remain
outside this free-streaming fixture.
