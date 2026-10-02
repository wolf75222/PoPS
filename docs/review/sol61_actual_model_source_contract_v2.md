# C25 actual model source inspection and migration

Fresh `compile_native` retains the exact UTF-8 bytes read from its actual
compiler input before invoking the unchanged compiler. Successful output
seals `pops.model.actual-compile@1`: TU hash, command, compiler executable
hash observed before/after compilation, SDK header signature and output
binary hash. Paths are report data; timestamps are absent. A recorded
compiler hash is not re-authenticated against a live compiler on inspection.
No compiler-to-DSO cryptographic graph proof is claimed.

The TU and record are same-directory atomic companions of the model DSO.
The existing identity-specific process lock now covers explicit destinations
as well as cache destinations; nested facades on the same thread borrow that
lock. Different threads/processes still acquire OS locks. Staged binary and
companions publish before the final identity sidecar commit. Model sidecar
v2 pins the provenance record hash; legacy sidecar v1 remains readable.
Inspection rejects partial companions, source/binary/hash changes, duplicate
JSON keys, malformed schemas and companion/identity symlinks. TU export
retains exact newline bytes. A crash may leave uncommitted companions; it
cannot legitimately certify them as complete inspection evidence.

Public migration policy applies to every compiled Case block through:

```python
resolved = pops.resolve(validated, layout=layout,
    compile_options={"model_source_policy": "require"})
artifact = pops.compile(resolved)
for block in artifact.blocks:
    evidence = block.model.source_provenance(require_complete=True)
    block.model.dump_cpp(output_directory / (block.name + ".cpp"))
```

`allow_missing` is the compatibility default. A legacy cache without the TU
and record reports `unavailable-legacy`, `complete=False`; `dump_cpp` refuses
without generating source. `require` refuses missing evidence before model
publication. `recompile` explicitly re-enters the authenticated authoring
compiler route under the publication lock and requires complete fresh
evidence; it never reconstructs a purported historical TU. A complete cache
hit returns verified retained bytes without invoking the compiler.
Recompilation refuses a destination previously published in the current
interpreter's native path registry: a dynamic loader may retain its old
handle. Use a fresh process or cache path for that explicit migration.

Whole-artifact retained-source evidence is versioned @2, with explicit
availability and model compile records. Its completeness concerns retained
source coverage, not execution/science or a compiler graph proof. Existing
Program source inspection is not newly qualified as an observed compiler TU.
The earlier diagnostic helper actual-compile-capture@1 remains separate.

Validation is Source/real tiny C++ only. Cache publication and cross-process
tests exercise real OS locks; metadata-only compiler seams do not mint
simulation results. The unrelated old `failed_program_compile_leaves...`
node requires a selected Native dimension and is not runnable under the
pure Source profile. Two older artifact-protocol fixtures likewise require
Native platform authority; their missing-authority failures are preserved,
not bypassed. ROOT must independently review, integrate, rebuild SDK and
execute authentic public Native fresh/cache-hit inspection before reception.

Both model cache specifications include `pops.codegen-source@1`: a length-framed
SHA-256 of sorted logical relative Python codegen filenames and their bytes.
An emitter-only correction therefore invalidates the cache even with unchanged
physical declarations, Native binary and SDK headers. This broad implementation
identity is not a digest of the emitted TU or a compiler dependency graph proof.
The actual compiled TU has its separate authenticated retained evidence.

Same-path recompile after process publication is currently refused explicitly.
Use a fresh process or a fresh cache destination. This conservative migration
restriction preserves previously loaded dynamic-library handles; immutable
binary-address publication remains a separate implementation improvement.

Frozen Source check (2026-10-02):

```sh
env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 /Users/romaindespoulain/miniforge3/envs/pops-api040-ir17/bin/python -m pytest --noconftest -p no:cacheprovider -o 'pythonpath=python .' tests/review/test_sol61_model_source_evidence.py tests/python/unit/codegen/test_compile_provenance.py tests/python/unit/codegen/test_facade_compile_cache.py tests/python/unit/codegen/test_compile_cache_lock.py -k 'not failed_program_compile_leaves' -q
```

Result: 32 passed, one explicitly deselected Native-authority node, 13.55 s.
The emitter-only return mutation changes the source-authority digest while
relocation and bytecode changes preserve it. Real tiny compiler input/output,
poisoned companions, legacy refusal, all-model public policy, foreign semantic
identity, nested publication lock and commit ordering are covered.

Independent findings follow-up: both explicit destination routes now apply
`recompile` refusal before compiler entry. Retention checks that the declared
TU occurs uniquely in the compiler arguments and the command output matches
the declared binary. Inspection compares all attached handle identities with
the committed sidecar. Public inspection/export acquires the publication lock;
export additionally hashes the exact bytes it returns, refusing a change after
authentication instead of exporting a second unverified read.

Replay of the six unchanged non-author probes plus the coherent 32 checks:
38 passed, one Native-authority node deselected, 19.89 s. The external test file
was selected from Hooke's WT; `pythonpath` must use absolute paths to this
checkout. A preceding relative-path run loaded the sibling's old 06bd source
and reproduced its five failures; that run is not evidence against this delta.
No Native runtime or MPI result is inferred.
