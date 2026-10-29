# Shared SDK, cohorts and qualification receipts

A qualification is worth something only if it names what it ran on. ROM-17 gives every PoPS
qualification a **receipt** that names the source commit, the installed SDK, every DSO mapped in
the process, the parameters, the backend and the ranks, and from which each K6 level (E, G, C,
R, S) can be read. A receipt counts for a **cohort** (one clean source and the one installed SDK
built from it) only if it passes the admission rules below. Two scripts own the protocol; neither
imports `pops`, so the code under qualification never vouches for itself:

- [`scripts/sdk_cohort.py`](../../scripts/sdk_cohort.py): source and SDK identities, coherence,
  native build attestations, the immutable cohort record and the installation lease;
- [`scripts/qualification_receipt.py`](../../scripts/qualification_receipt.py): the receipt, its
  levels, the overlay findings and the admission decision.

## Identities

| Object | Identity | Refused when |
| --- | --- | --- |
| Source | commit, tree, digest of the SDK inputs (`SDK_INPUT_PATHS`) | a tracked file differs from HEAD, or an SDK input path holds an untracked file |
| SDK | digest of the exact installed bytes of the `pops` package | pip `RECORD` does not authenticate a file (native leaves may instead be authenticated by `variants.json`), or the installation is not from a retained wheel |
| Coherence | installed Python sources and headers equal the blobs of the commit; each leaf's `abi_key` names the installed header signature | any file missing, different or unexpected |
| Native provenance | `.pops-sdk/native-builds/<leaf sha256>/<SDK inputs sha256>.json`, written by the official build | no clean build of the same SDK inputs attests the leaf |
| Cohort | digest of commit, tree and SDK identity; record `<site-packages>/.pops-sdk/cohort.json`, mode 0444 | publishing another cohort into an installation that already holds one |

Native bytes cannot be compared with C++ sources, so the official build states which SDK inputs it
built: `build_python.sh --wheel-dir` runs `sdk_cohort.py attest-build` after the retained wheel is
installed, restored and verified. A leaf restored from an earlier build (`preserve_native_variants`)
keeps the attestation of the inputs it was built from, so a sibling dimension built from other
sources never joins a new cohort. A build from modified SDK inputs writes `dirty.json`, which
attests nothing.

## The installation lease

A qualification takes a shared lease on the installation for one source commit; a build, an
install or a cohort publication takes it exclusively (`flock` on `.pops-sdk/lease.lock`, never
waiting). While one source qualifies, a second source and any build are refused at once
(`InstallationBusy`, exit code 3); while a build runs, every qualification is refused.
`build_python.sh` re-executes itself under `sdk_cohort.py mutate`, so the whole build, from the
first `pip` call to the attestation, holds the installation. A whole test session can hold it
with `sdk_cohort.py hold --installation SITE --source . -- python -m pytest ...`; each `Receipt`
also takes the lease itself. The lease is advisory: it binds the PoPS tools, not an arbitrary
`pip install` typed by hand. It needs POSIX `flock`; elsewhere a build runs unleased with a
warning and a receipt records that no lease was held, which refuses its admission.

## Receipts and levels

`Receipt` is used as a context manager around the real lifecycle. Each call records one stage
with its evidence; an exception is recorded as the failure of the stage in progress and the
receipt is still written.

| Stage | Call | Evidence |
| --- | --- | --- |
| E | `equations(module, ...)` | each case source hashed and committed unchanged at HEAD |
| G | `plan(pops.resolve(...))` | the verified plan, its plan and semantic identities |
| C | `compiled(pops.compile(plan))` | every program, model and component DSO hashed with its identity sidecar and generated C++, tied to the G plan |
| loaded | `loaded(pops.bind(...))` | every artifact DSO found mapped in this process by content (private authenticated copies included); the System's `installed_program_hash()` equals `pops_program_hash()` of the mapped program, read with `RTLD_NOLOAD` |
| advanced | `advanced(pops.run(...), runtime)` | accepted steps of the artifact the C stage compiled |
| S | `property(name, value=, bound=, passed=, oracle=, domain=)` | one scientific property, its independent oracle and its domain |

The level is the highest consecutive rung of `E < G < C < R < S`; R needs both `loaded` and
`advanced`. Closing the receipt also records the process (interpreter, environment variables that
select code, the hash of the harness itself), the backend reported by the loaded native leaf, all
mapped DSOs and the ranks. In a multi-rank run, pass `gather=` (an allgather of one string): the
ranks agree only if every rank observed the same source, SDK, artifact and DSOs.
[`test_qualification_receipt_native.py`](../../tests/python/integration/runtime/test_qualification_receipt_native.py)
is the reference receiver.

## Admission

`qualification_receipt.py check RECEIPT --cohort COHORT [--require R] [--rehash]` exits 0 only
when every rule holds:

- the receipt authenticates its own identity and its levels follow from its stages;
- it is a PoPS runtime receipt that reaches the required level, closed without failure;
- no overlay was seen: clean source, wheel installation coherent with it, lease held, the
  committed harness, `pops` and every `pops.*` module loaded from the SDK with its exact bytes, no
  `POPS_NATIVE_VARIANTS_ROOT` outside the SDK, no `POPS_INCLUDE` with other headers, no preloaded
  library, the native leaf and every artifact DSO the files that were hashed;
- the ranks agreed;
- its source commit and tree and its SDK identity are the cohort's.

A standalone C++ program is recorded with `record-standalone`; its receipt is C-finite or
R-finite and never admitted (K6). A syntax or compile-only run reaches at most C and is refused
at R. An overlay is refused at every level, E included.

## Requalification after a lot is integrated

A cohort never changes. When a lot lands on the integration branch, its commit gets a new
installation and a new cohort; receipts of the previous cohort do not carry over, since the
cohort identity changes with the commit.

1. Check out the integrated commit; `git status` must be clean.
2. Create a fresh environment once per worktree (`bash scripts/setup_env.sh`, with a new
   `POPS_ENV_NAME` if the previous one holds a cohort).
3. Build every dimension the lot needs from that checkout, retaining each wheel:
   `bash scripts/build_python.sh --dim 2 --wheel-dir "$WHEELS/dim2"`. Each build attests its leaf.
4. Publish and verify the cohort:

   ```bash
   SITE="$(python -I -c 'import sysconfig; print(sysconfig.get_path("platlib"))')"
   python -I scripts/sdk_cohort.py publish --source . --package "$SITE/pops" --seal
   python -I scripts/sdk_cohort.py verify --package "$SITE/pops" --source .
   ```

5. Run the receivers of the lot under the lease, with `env -u PYTHONPATH`, writing their receipts
   (`POPS_CI_PYTEST_TIMINGS_DIR` selects the directory of the reference receiver).
6. Admit each receipt: `python -I scripts/qualification_receipt.py check RECEIPT --cohort
   "$SITE/.pops-sdk/cohort.json" --require R --rehash`. Archive the cohort record, the receipts
   and the exact commands with the lot's handoff.

## Limits

- CI installs one retained wheel per dimension into a fresh prefix and does not publish a cohort
  file. The reference receiver builds the cohort of that installation in memory and re-proves the
  installation byte-identical to the retained wheel (`prove_installed_wheel.py`); the wheel's own
  provenance is the CI run that built it from the same checkout.
- The lease protects only the installation it names; the compilation cache (`POPS_CACHE_DIR`) is
  identified by content in each receipt, not leased.
- Overlay detection covers the import system, the native selector, the header override and
  `LD_PRELOAD`/`DYLD_INSERT_LIBRARIES`; it does not prove the absence of code injected by other
  means.
