# Contributing to PoPS

Work on a branch or isolated worktree from `master`. Keep each change coherent, preserve
numerical and runtime contracts, and describe the problem, resulting behavior and validation
in the pull request. Use a linked issue for substantial design changes.

## Environment and build

Follow the [README](README.md) setup once per worktree. Activate `pops`, and rebuild with
`bash scripts/build_python.sh --dim 2` after C++ or generated native-code changes. Use
`env -u PYTHONPATH` for installed-package checks so another checkout cannot supply Python code.
`POPS_ENV_NAME` selects a separate conda environment when worktrees need incompatible artifacts.

CMake presets are in [CMakePresets.json](CMakePresets.json): `serial`, `parallel` (OpenMP),
`mpi` (MPI and parallel HDF5), and `python` / `python-parallel` (bindings). Each has its own
build directory. The checked-in presets use Dim=2; other dimensions require a separate build
configuration. A backend must be installed before selecting its preset.

## Focused validation

Finish a coherent change before testing. Select tests by affected contract using
[tests/test_manifest.toml](tests/test_manifest.toml), then build the required C++ targets and
run their CTest entries, or run the corresponding Python files with `python -m pytest`.
Reinstall the Python package before tests that exercise its runtime. Source-only tests must
explicitly load the checkout when an older installed package is present.

For example, from the repository root:

```bash
cmake --preset serial
cmake --build --preset serial --target test_prepared_cartesian_nd
ctest --preset serial -L '^cpp-target:test_prepared_cartesian_nd$' --output-on-failure
python scripts/check_packaging_manifest.py
bash scripts/build_docs.sh
```

Use the repository's gate scripts for release or benchmark qualification rather than inventing
a replacement. [Verification scope](docs/development/migration_verification_scope.md) separates
source checks, native execution and scientific acceptance. Preserve failing-case inputs and
numerical tolerances; do not weaken guards or count skips as passing evidence.

## CI

[Change-aware CI routing](docs/development/change_aware_ci.md) documents the planner and local
reproduction commands. The required `ci.yml` gate selects affected tests and conservative
fallbacks. `ci-kokkos` forces the Serial Kokkos gates; `ci-full` adds the wider MPI/OpenMP lanes.
The `quality` label requests static analysis and deeper checks. Scheduled/manual runs provide
broader coverage, and release tags invoke the wheel/release workflow.

`docs.yml` checks documentation on pull requests, pushes and manual dispatch. Its deterministic
entry point is `bash scripts/build_docs.sh`. Documentation-only changes do not need a native build.
Report actual GitHub status separately from local validation.

## Code and documentation conventions

- Use `.clang-format` (clang-format 19) and `.clang-tidy` for C++; Ruff settings are in
  `pyproject.toml`. Avoid unrelated formatting sweeps.
- Keep public headers self-contained, use `#pragma once`, and preserve existing naming within
  a subsystem. Use explicit ownership and typed handles instead of name-based runtime dispatch.
- Check authored input at host boundaries. Exceptions must not escape device kernels; carry
  failures through the numerical/runtime status protocol. Assertions do not replace public
  validation. Document collective participation and synchronization for MPI operations.
- Comment invariants, mathematical sign/normalization conventions, layout/ghost requirements,
  ownership, and failure behavior. Prefer concise `///` API comments and Python docstrings over
  restating code. State the conditions under which a method preserves conservation or stability.
- Keep physical model authoring visible in tutorials. Use linear scripts followed by simulation
  setup, with helper functions only where the subject itself requires one.
- Update the canonical page with the code. Keep one explanation per topic and link to executable
  examples. Source behavior takes precedence over design intent. Documentation rules and the
  source-dependency map are in [docs/DOC_QUALITY.md](docs/DOC_QUALITY.md).

## Review and release

Review the complete diff and directory structure after moves or deletions, including links,
imports, manifests, CI selectors and packaging. Add regression tests for changed contracts,
and keep numerical evidence separate from structural checks. Explain any unrun configuration.

Use the [PR template](.github/PULL_REQUEST_TEMPLATE.md), add notable changes to
[CHANGELOG.md](CHANGELOG.md), and follow [versioning](docs/VERSIONING.md). The CMake project
version is the version source. Merge only after reviewing the changes and the required GitHub
checks; do not infer CI success from a local test run.
