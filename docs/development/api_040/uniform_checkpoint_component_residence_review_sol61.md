# Uniform checkpoint: component residence and bind identity

Source baseline: `bfae73f34174079bc6f2d8f4d50aa11d08eca5b5`. Private branch
`codex/api040-sol61-uniform-checkpoint-plan`, outside Documents. This change is
Python only; no native module, SDK, environment, header, installation or JIT was
changed by this reviewer.

## Actual failure and causal evidence

Root's actual M19 MPI2 reception failed all six cases per rank at initial child
checkpoint capture. The strict preflight refused differing capture-plan identities;
this is retained as a historical native failure, not received as a positive.
The authentic evidence is under
`/Users/romaindespoulain/dev/tmp/pops-api040-native-reception-evidence-20261001/`:

- `m19-product-mpi2-dim2/rank0.log`: the six original failures.
- `m19-checkpoint-plan-payloads/rank-0-plan-0.json`, SHA256
  `15ba6c4073868d48072c6aed703d878b910f6c0f73d6a8c39dfa63d6deefd9ce`.
- `m19-checkpoint-plan-payloads/rank-1-plan-0.json`, SHA256
  `7bdc5acc6a329a3b9005877001da19eb176d021de1b6923bd2e49c058a6839e8`.

Root's diagnostic intercepted identity construction without changing its result.
The exact payload difference is only `runtime_identities[2]`, domain `bind@1`:
rank0 `f338d8da3b182b6796cb736b473344a26dcc389bab6bff8b281490508a88b1d3`,
rank1 `601d032ab9c28754200b980470da46ae87a437f50fd13387a62837e798e058f5`.
Target, clock, state schema, geometry, EB, cadence, histories, cache, program hash,
semantic/artifact identities and lifecycle agree exactly. The diagnostic wrapper
itself remained red because its expected exception surface was wrong; the
unchanged captured plans are the evidence used here.

The source chain explains this difference: `compile_resolved_plan_once` restores
the peer's local cache after authenticating the common compiled artifact;
`bind` then calls `component_store_dir()` and installs the same authenticated
component in that local cache. `InstalledComponent.to_data()` includes `path`, and
`InstallPlan._payload()` used that entire projection for `components`. A host
counter-probe with the exact InstalledComponent class, public manifest/interface
contracts and real `_plans._evidence` reproduces the identity difference by
changing only residence. No executable component is manufactured by that probe.

## Corrected contract

`InstalledComponent.bind_identity_data()` uses an explicit schema-v2 projection.
It retains all content, interface, runtime, platform, entry, loaded-state and
provenance facts from `to_data()`, excluding only the local residence `path`.
`to_data()` retains that path unchanged for inspection and realization provenance.
`_plans._evidence` selects this projection only for the exact InstalledComponent
type. File bytes, exports and loaded native identity are still authenticated by
the unchanged installation/`verify` guards. A forged binary identity, artifact,
entry, origin, loaded state or runtime contract still changes the bind identity.

There is no checkpoint fallback or rank-zero identity substitution:
`_system_io.py` and `_checkpoint_collective.py` are unchanged. Plan disagreement
still refuses before native capture; sealed-state disagreement still refuses
before publication. Every rank participates, and only rank0 publishes.

Component-free bind payloads and digests remain byte-identical to the frozen parent
through an independently extracted historical InstallPlan payload and recursive
evidence function. Binds containing InstalledComponent intentionally acquire the
new projection. Their old bind digest is not silently admitted on restart. Fresh
capture/restart under the new Python source is a Root native obligation; this is
not a migration or a historical checkpoint compatibility claim.

## Source/host reception

`tests/review/test_sol61_uniform_checkpoint_plan.py` executes real Uniform prepare,
collective consensus, root publication and manifest sealing over a bounded
two-thread transport seam. ABI metadata and transport are explicit host seams;
no native catalogue or MPI implementation is replaced. Tests receive:

- distinct local shards and an empty peer, one/three components, common plans,
  all-rank capture and exclusive rank0 publication;
- relocated exact component metadata through both strict checkpoint votes;
- ncomp, spatial bounds, cadence and bind mismatch before capture;
- different sealed physical state before publication;
- six component content/execution-metadata mismatches;
- actual file digest authentication at two different resident paths and changed
  bytes at one path refused by the unchanged verifier (export inspection is an
  explicit host seam; these are opaque input files, not native binaries);
- component-free full bind payload/digest equality against the pinned parent.

Exact source-only command, run from the private checkout:

```sh
rtk proxy env -u PYTHONPATH -u POPS_NATIVE_DIM PYTHONPATH=python PYTHONDONTWRITEBYTECODE=1 /Users/romaindespoulain/miniforge3/envs/pops-api040/bin/python -B -m pytest -q tests/review/test_sol61_uniform_checkpoint_plan.py tests/python/unit/codegen/test_component_packages.py tests/python/unit/codegen/test_component_cache_authority.py tests/python/unit/codegen/test_bind_parameter_evidence.py
```

Final result: **37 PASS in 1.32s**, comprising 19 independent checks and 18
existing package/cache/bind regressions. Ruff and `git diff --check` pass. Frozen
checkpoint guard blobs remain `b993a0e28b7ea2dd6421c63ac0f4d940ab45342c`
(`_system_io.py`) and `fe9e1df7d49345976ba1e4195fa55e112f8fe29f`
(`_checkpoint_collective.py`).

An initial broader command included `test_typed_phase_records.py`; collection
stopped because it imports the unselected native `_pops`. It executed no tests and
is not a pass or a production defect. This reviewer did not select/load a native
module to circumvent that source-only boundary. An initial file-verifier probe
also had a tuple instead of the export inspector's set result; that host seam was
corrected before the final reception.

Native/MPI2 success, common restart, actual loaded component verification and all
six M19 cases remain pending Root's rebuilt Python reception. No AMR, GPU or new
scientific physical qualification follows from these source/host tests.
