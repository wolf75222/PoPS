# Shared SDK, cohorts and qualification receipts

A qualification is worth something only if it names what it ran on. ROM-17 gives every PoPS
qualification a **receipt** that names the source commit, the installed SDK, every DSO mapped in
the process, the parameters, the backend and the ranks, and from which each K6 level (E, G, C,
R, S) can be read. A receipt counts for a **cohort** (one clean source and the one installed SDK
built from it) only if it passes the admission rules below. Two scripts own the protocol:

- [`scripts/sdk_cohort.py`](../../scripts/sdk_cohort.py): source and SDK identities, coherence,
  native build attestations, qualification prefixes, the immutable cohort record and the
  installation lease;
- [`scripts/qualification_receipt.py`](../../scripts/qualification_receipt.py): the receipt, its
  levels, the overlay findings and the admission decision.

Neither imports `pops`: identities are computed from files, not taken from the code under
qualification. A few facts can only come from that code, and the receipt names them as such: the
program hash a System installed (`installed_program_hash()`), the program hash a mapped program
image exports (`pops_program_hash()`), and the ABI key and runtime environment report of the
native leaf. The build attestation also loads the leaf it attests, in a child interpreter
(`verify_installed_native.py`).

## Identities

| Object | Identity | Refused when |
| --- | --- | --- |
| Source | commit, tree, digest of the SDK inputs (`SDK_INPUT_PATHS`) | a tracked file differs from HEAD (a renamed file counts at both paths), or an SDK input path holds an untracked file |
| SDK | digest of the exact installed bytes of the `pops` package | pip `RECORD` does not authenticate a file (native leaves may instead be authenticated by `variants.json`), or the installation is not from a retained wheel |
| Coherence | installed Python sources and headers equal the blobs of the commit; each leaf's `abi_key` names the installed header signature | any file missing, different or unexpected |
| Native provenance | `.pops-sdk/native-builds/<leaf sha256>/<SDK inputs sha256>.json` in the build environment and `<wheel>.pops-build.json` beside the retained wheel, written by `build_python.sh` | no build of the same clean SDK inputs attests the leaf, or the source changed during that build |
| Cohort | digest of commit, tree and SDK identity; record `<site-packages>/.pops-sdk/cohort.json`, mode 0444 | publishing another cohort, or building or installing anything (`mutate`), into an installation that already holds one |

Native bytes cannot be compared with C++ sources, so the build that installed a leaf states which
SDK inputs it built. `build_python.sh` identifies the source before its first `pip` call; with
`--wheel-dir`, once the retained wheel is installed, restored and verified, it runs
`sdk_cohort.py attest-build`, which:

- runs only inside `build_python.sh`, under the exclusive lease the build inherited; a leaf is
  never attested by hand;
- identifies the source again and writes `dirty.json`, which attests nothing, if the SDK inputs
  were modified or the source changed between the start and the end of the build;
- loads the leaf in a child interpreter and requires the dimension, version and ABI key it
  declares to be those of `variants.json`;
- records the build environment: the interpreter, `CONDA_PREFIX`, the conda executable and its
  sha256, a digest of `conda list --explicit` when the environment is a real conda environment,
  the versions of scikit-build-core, pybind11, NumPy, CMake and Ninja, and from the CMake cache of
  the build tree the C++ compiler (path, sha256, id, version), Kokkos (directory, version),
  pybind11, MPI and HDF5.

A leaf restored from an earlier build (`preserve_native_variants`) keeps the attestation of the
inputs it was built from, so a sibling dimension built from other sources never joins a new
cohort. The refusal binds the PoPS tools, not a forger: an attestation is a file, and whoever can
write the installation can write one.

## The installation lease

A qualification takes a shared lease on the installation for one source commit; a build, an
install or a cohort publication takes it exclusively (`flock` on `.pops-sdk/lease.lock`, never
waiting). While one source qualifies, a second source and any build are refused at once
(`InstallationBusy`, exit code 3); while a build runs, every qualification is refused.
`build_python.sh` re-executes itself under `sdk_cohort.py mutate`, so the whole build, from the
first `pip` call to the attestation, holds the installation. A whole test session can hold it
with `sdk_cohort.py hold --installation SITE --source . -- python -m pytest ...`. Each `Receipt`
on a retained wheel installation also takes the lease itself, right after it identifies the
source and before it identifies the SDK; on a checkout or another unmanaged tree it takes none,
writes nothing there and is never admitted.

The lease is advisory: it binds the PoPS tools, not an arbitrary `pip install` typed by hand. Its
domain is one host and a file system with POSIX `flock`. The lease directory takes the mode of
the installation, and lock files are opened read-only, so in an environment shared by a group
every user who can read the installation can take the lease; a user who cannot create or record
it gets a refusal (exit code 2), not a crash. Without `flock`, a build runs unleased with a
warning and a receipt records that no lease was held, which refuses its admission.

## Receipts and levels

`Receipt` is used as a context manager around the real lifecycle. Each call records one stage
with its evidence; an exception is recorded as the failure of the stage in progress and the
receipt is still written. A class counts as the SDK's only if an SDK file with its installed
bytes defines it.

| Stage | Call | Evidence |
| --- | --- | --- |
| E | `equations(module, ...)` | each case source hashed and committed unchanged at HEAD |
| G | `plan(pops.resolve(...))` | the SDK's `ResolvedSimulationPlan`, verified, with its plan and semantic identities |
| C | `compiled(pops.compile(plan))` | the SDK's `CompiledSimulationArtifact` of the G plan; every program, model and component DSO hashed with its identity sidecar and generated C++ |
| loaded | `loaded(pops.bind(...))` | `pops._pops` is the SDK's native leaf, mapped with its recorded bytes; the System is a class of that leaf and the runtime and its executor are SDK objects; every artifact DSO is found mapped in this process by content (private copies included); the System's `installed_program_hash()` equals `pops_program_hash()` of the mapped program, read with `RTLD_NOLOAD` |
| advanced | `advanced(pops.run(...), runtime)` | the SDK's `RunReport` of the same runtime, with accepted steps of the artifact the C stage compiled |
| S | `property(name, check, runtime_or_report, ..., domain=)` | the receipt calls the oracle `check` itself on the runtime or report of the run and records its verdict, its file and the bounds it closes over; the oracle must be a committed file of the source, not of the SDK |

The level is the highest consecutive rung of `E < G < C < R < S`; R needs both `loaded` and
`advanced`. A receipt gives the level of its requirement in its domain, nothing more: the
reference M01 receipt qualifies two steps of M01 on 16x8 cells with the case's committed oracle,
not M01 at the tolerances of the full case.

The code that drives the qualification is judged too. Every frame that leads to the creation of
the `Receipt` (its driver) and every module the process imported must come from the source, the
SDK or the interpreter; a frame or module of the source must be committed unchanged, and the
innermost driver frame must be a committed file of the source. Code compiled from a string is
allowed only as the outermost frame (the `python -c` bootstrap of CI). A receipt is therefore
written by a committed receiver, never by an ad hoc script.

Closing the receipt also records the process (interpreter, environment variables that select
code, the hash of the harness itself), the backend reported by the loaded native leaf, all mapped
DSOs and the ranks. In a multi-rank run, pass `gather=` (an allgather of one string): the ranks
agree only if every rank observed the same source, SDK, artifact, DSOs, overlay findings and level.
[`test_qualification_receipt_native.py`](../../tests/python/integration/runtime/test_qualification_receipt_native.py)
holds the reference receivers: M01 in `System` and the finite AMR case in `AmrSystem`.

## Admission

`qualification_receipt.py check RECEIPT --cohort COHORT [--require R] [--rehash]` exits 0 only
when every rule holds:

- the receipt is self-consistent (its id is the digest of its content) and its levels follow from
  its stages; this is no signature, so a receipt is evidence only as far as the files and the
  commands that wrote it are trusted;
- it is a PoPS runtime receipt that reaches the required level, closed without failure;
- no overlay was seen: clean source, wheel installation coherent with it, lease held, the
  committed harness, a committed driver and committed source modules, no module from outside the
  source, the SDK and the interpreter, `pops` and every `pops.*` module loaded from the SDK with
  its exact bytes, every mapped SDK file (the native leaf included) with its SDK bytes and not
  deleted, no `POPS_NATIVE_VARIANTS_ROOT` outside the SDK, no `POPS_INCLUDE` with other headers,
  no preloaded library, and every artifact DSO the file that was hashed;
- the ranks agreed;
- its source commit and tree and its SDK identity are the cohort's.

A standalone C++ program is recorded with `record-standalone`; its receipt is C-finite or
R-finite and never admitted (K6). A syntax or compile-only run reaches at most C and is refused
at R. An overlay is refused at every level, E included.

## Requalification after a lot is integrated

A cohort never changes. When a lot lands on the integration branch, its commit gets a new
installation and a new cohort; receipts of the previous cohort do not carry over, since the
cohort identity changes with the commit. The build environment of the worktree stays incremental:
the cohort lives in a fresh qualification prefix made from the retained wheel.

1. Check out the integrated commit; `git status` must be clean.
2. Once per worktree, create the environment (`bash scripts/setup_env.sh`).
3. Build every dimension the lot needs from that checkout, each into a new empty wheel
   directory: `bash scripts/build_python.sh --dim 2 --wheel-dir "$WHEELS/dim2"`. Each build
   attests its leaf in the environment and beside its wheel.
4. Install the wheel into a new qualification prefix, then publish and verify the cohort there:

   ```bash
   WHEEL="$(ls "$WHEELS"/dim2/pops-*.whl)"
   python -I scripts/sdk_cohort.py prefix --wheel "$WHEEL" --prefix "$QUALIFY/dim2"
   PY="$QUALIFY/dim2/bin/python"
   SITE="$("$PY" -I -c 'import sysconfig; print(sysconfig.get_path("platlib"))')"
   "$PY" -I scripts/sdk_cohort.py publish --source . --package "$SITE/pops" --seal
   "$PY" -I scripts/sdk_cohort.py verify --package "$SITE/pops" --source .
   ```

   `prefix` makes a venv of the build environment's interpreter (`--system-site-packages`, so a
   conda environment's NumPy and pytest stay visible), installs the wheel alone, proves the
   installation byte-identical to the wheel and verifies its native leaf.
5. Run the receivers of the lot with that interpreter and `env -u PYTHONPATH`, writing their
   receipts; for the reference receivers:
   `env -u PYTHONPATH PYTHONNOUSERSITE=1 POPS_NATIVE_DIM=2 POPS_CI_PYTEST_TIMINGS_DIR="$OUT" "$PY"
   -m pytest tests/python/integration/runtime/test_qualification_receipt_native.py`.
6. Admit each receipt: `"$PY" -I scripts/qualification_receipt.py check RECEIPT --cohort
   "$SITE/.pops-sdk/cohort.json" --require R --rehash`. Archive the cohort record, the receipts
   and the exact commands with the lot's handoff.

## Limits

- Only the reference receivers write receipts so far. A lot whose receivers write none yet runs
  them in its cohort's prefix under `sdk_cohort.py hold --installation "$SITE" --source . --` and
  archives their output with the cohort record. That shows which installation they ran on, but
  it is no receipt and the lot claims no K6 level from it. To claim one, the lot gives a receiver
  a receipt: wrap its lifecycle in `Receipt`, record each stage and pass its committed oracle to
  `property`, as the reference receivers do.
- CI installs one retained wheel per dimension into a fresh prefix and does not publish a cohort
  file. The reference receivers build the cohort of that installation in memory and re-prove the
  installation byte-identical to the retained wheel (`prove_installed_wheel.py`); the wheel's own
  provenance is the CI run that built it from the same checkout. Publishing CI's own cohorts with
  `prefix` is a follow-up.
- The lease protects only the installation it names; the compilation cache (`POPS_CACHE_DIR`) is
  identified by content in each receipt, not leased.
- Overlay detection covers the import system, the driver, the native selector, the header
  override and `LD_PRELOAD`/`DYLD_INSERT_LIBRARIES`; it does not prove the absence of code
  injected by other means.
