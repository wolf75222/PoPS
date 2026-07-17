# Test Architecture Spec

Date: 2026-07-02.
Status: target architecture.
Scope: all tests owned by `adc_cpp`: C++ core tests, Python package tests,
architecture tests, backend tests, type checks, coverage, and CI test selection.

This spec assumes no compatibility constraint with the current test layout. The
current autonomous C++ `main()` tests, Python script-style tests, duplicated
assertion helpers, and name-based CI selection may be replaced wholesale.

## Decision

PoPS uses one real test framework per language and one orchestration layer:

- C++ tests are written in **GoogleTest**.
- Python tests are written in **pytest**.
- Python type checking is done by **Pyright**.
- Test execution, backend variants, MPI launches, labels, timeouts, coverage,
  and sanitizer configurations are orchestrated by **CTest/CMake presets** and CI.

GoogleTest replaces the C++ mini-harness. It does not replace CTest. CTest is
the runner that understands configured build trees, test labels, MPI commands,
resource limits, and preset-specific execution. GoogleTest is the assertion and
fixture framework inside C++ test binaries.

## Goals

1. Make tests readable as product code, not as one-off scripts.
2. Make every test discoverable by framework discovery, labels, and manifest data.
3. Make backend coverage explicit: Serial Kokkos, OpenMP Kokkos, MPI, Python,
   generated native loaders, sanitizers, coverage, and GPU/manual runs.
4. Remove duplicated helper code from test files.
5. Make failures actionable: test name, parameter value, expected value, actual
   value, tolerance, backend, and seed are visible in the failure report.
6. Keep the common PR path fast while preserving full coverage in full/nightly
   lanes.
7. Make coverage measurable by domain and tier, not only by a global number.
8. Make Python static correctness a first-class gate, not an optional lint.

## Non-Goals

- A single universal tool for all checks. C++ tests, Python tests, type checks,
  and coverage are different activities and keep different tools.
- Preserving existing test file names, output formats, or helper APIs.
- Keeping every historical test as a one-to-one rewritten test. The rewrite may
  merge, split, or delete tests when the new coverage map remains complete.
- Running GPU validation on generic GitHub runners. GPU remains a dedicated
  backend lane with the same metadata model.

## Practices To Adopt Immediately

These practices apply from the first rewrite patch. They are compatible with
the current Kokkos-only CMake/CTest topology and should not wait for the full
manifest generator.

### C++ and GoogleTest

1. Use a project-owned `tests/cpp/support/gtest_main.cpp`.
   Do not use `GTest::gtest_main` as the final entry point. PoPS needs one
   explicit place to initialize GoogleTest, install any global test
   environment, and later centralize Kokkos/MPI lifecycle policy if a test lane
   needs it.

2. Register tests through a PoPS CMake wrapper from day one.
   The wrapper may be thin at first, but every GoogleTest target must go
   through it so labels, timeouts, discovery mode, XML output, resource locks,
   and heavy-translation-unit pools are not reimplemented per suite.

3. Prefer `gtest_discover_tests()` for normal host builds.
   It discovers the real tests from the compiled executable, handles
   value-parameterized tests correctly, and avoids requiring CMake reruns when
   test cases change. The wrapper can switch to `PRE_TEST` discovery or another
   mode for cross/device environments.

4. Give every parameterized test a stable readable parameter name.
   Parameter suffixes must be deterministic and CTest-safe. Use names such as
   `Rusanov`, `HLLC`, `Weno5`, `Dirichlet`, `Neumann`, `HostSerial`, not raw
   tuple indices.

5. Use GoogleTest matchers and `testing::AssertionResult` for numerical checks.
   Avoid `EXPECT_TRUE(complex_expression)`. PoPS numerical checks should report
   the norm, tolerance, field name, backend, and worst cell/index.

6. Test public contracts first.
   Private implementation tests are allowed only for extracted implementation
   classes or intentional `detail` contracts. The default is black-box testing
   through public headers and public Python/C++ facades.

7. Keep device assertions host-side.
   Kokkos/device code should produce observable host data; GoogleTest assertions
   evaluate that data on the host. Do not put GoogleTest macros in device
   lambdas.

### CTest and Backend Execution

1. Use `RESOURCE_LOCK` for global mutable resources.
   Native-loader build caches, shared generated source directories, global HDF5
   files, and process-wide runtime caches must be locked instead of forcing the
   whole suite through `RUN_SERIAL`.

2. Use `RESOURCE_GROUPS` for scarce counted resources.
   GPU lanes and future multi-device runners should declare resource slots so
   CTest can avoid oversubscription even when `ctest -j` is high.

3. Keep MPI rank counts as separate CTest tests.
   A rank matrix is not a loop inside one GoogleTest case. Each rank count gets
   its own CTest test name, labels, timeout, and failure.

4. Emit GoogleTest XML through the CMake wrapper.
   XML output must use a per-target/per-test output directory so parallel CTest
   runs do not race on the same report file.

### Python and pytest

1. Turn on pytest strictness immediately.
   Use registered markers and strict config so marker typos become failures.
   If the installed pytest version makes global `strict = true` too broad,
   enable `strict_config`, `strict_markers`, `strict_parametrization_ids`, and
   `strict_xfail` individually.

2. Move shared behavior into `tests/python/conftest.py`.
   Kokkos root discovery, `_pops` import, compiler discovery, temporary native
   build directories, deterministic RNG, and skip/fail backend policy belong in
   fixtures, not in individual files.

3. Use `tmp_path` and `monkeypatch` for filesystem and environment changes.
   No Python test should write into the source tree, depend on the caller's
   current environment, or mutate `os.environ` without automatic restoration.

4. Use pytest parametrization instead of loops with manual failure counters.
   Each limiter, flux, backend, solver, or layout case should become an
   independently reported pytest node.

5. Use `pytest-xdist` only for tests that are fixture-isolated.
   Tests marked `native_loader`, `mpi`, `hdf5`, or any future `serial_resource`
   marker are excluded from parallel xdist runs unless their fixture declares a
   safe per-worker directory and no shared process/global state.

### Pyright

1. Add `pyrightconfig.json` early, before full strictness is achievable.
   Configure `include`, `exclude`, `pythonVersion`, and `extraPaths` first so
   the checked file set is deterministic.

2. Use Pyright's `strict` path list for production Python.
   `python/pops` should move to strict first; tests can start less strict while
   shared fixtures and public helper APIs are typed.

3. Create `_pops.pyi` before trying to type-check bindings-heavy tests.
   The compiled extension is part of the public Python contract; without a stub,
   strict type checking will either lose signal or produce avoidable noise.

### Coverage

1. Filter coverage to production source.
   C++ coverage is filtered to `include/pops/` and relevant binding/runtime
   source. Python coverage is filtered to `python/pops`, not tests.

2. Enable branch coverage where the tool supports it.
   Line coverage alone is not enough for codegen decisions, solver routes, and
   validation branches.

3. Store coverage exceptions explicitly.
   Any excluded file, generated region, backend-only branch, or intentionally
   uncovered defensive path needs a short entry in
   `tests/coverage_exceptions.toml`.

## Tool Roles

### GoogleTest

GoogleTest owns C++ test expression:

- `TEST`, `TEST_F`, `TEST_P`, and typed tests define cases.
- Fixtures own shared setup for `Mesh`, `MultiFab`, `System`, `AmrSystem`,
  solver problems, native loader temp directories, and deterministic seeds.
- Matchers and helper assertions own numerical comparisons.
- Parameterized tests replace repeated copies over limiter, Riemann solver,
  geometry, time integrator, backend mode, or MPI rank count.

GoogleTest binaries must not parse project-specific environment policy directly.
They receive backend state from the configured build and from CTest properties.

### CTest

CTest owns execution:

- Selecting test subsets by labels.
- Applying timeouts.
- Launching MPI tests with `mpiexec`.
- Running a configured tree for one backend/preset.
- Exposing test metadata to CI and coverage dashboards.
- Enforcing resource locks or serial execution when a test compiles loaders or
  uses a global cache.

Every GoogleTest case is visible to CTest using `gtest_discover_tests` or an
equivalent project wrapper.

### pytest

pytest owns Python runtime tests:

- All Python tests are normal pytest tests with `assert`, fixtures, `tmp_path`,
  `monkeypatch`, `capsys`, `pytest.raises`, `pytest.mark.parametrize`, and
  registered markers.
- No Python test file is executed as a standalone script in CI.
- No global `fails` counters or `chk` helpers are allowed.
- Tests that need `_pops`, Kokkos, native MPI/parallel HDF5, or a native compiler declare
  that requirement through markers and fixtures.

### Pyright

Pyright owns Python static type checks:

- `python/pops` is type checked.
- Public Python APIs expose enough annotations and stubs for meaningful checks.
- The compiled `_pops` module has a checked stub surface.
- Type checks are a separate CI job and do not run through pytest.

### Coverage Tools

Coverage tools own measurement:

- C++ coverage uses `gcovr` with GCC or `llvm-cov` with Clang, selected by the
  configured coverage preset.
- Python coverage uses `coverage.py`/`pytest-cov`.
- Coverage reports are filtered by source domain and compared against tiered
  thresholds.

## Repository Layout

The target layout is:

```text
tests/
  cpp/
    CMakeLists.txt
    support/
      gtest_main.cpp
      numeric_matchers.hpp
      mesh_fixtures.hpp
      solver_fixtures.hpp
      system_fixtures.hpp
      amr_fixtures.hpp
      native_loader_fixtures.hpp
      mpi_test_utils.hpp
      test_data.hpp
    unit/
      mesh/
      numerics/
      elliptic/
      runtime/
      codegen/
      descriptors/
    integration/
      system/
      amr/
      mpi/
      kokkos/
      native_loader/
    regression/
      bugs/
      parity/
      numerical_oracles/
  python/
    conftest.py
    unit/
      descriptors/
      codegen/
      time/
      mesh/
      solvers/
    integration/
      bindings/
      runtime/
      amr/
      native_loader/
      io/
    architecture/
    regression/
  gpu/
    romeo/
      CMakeLists.txt
      README.md
      *.cpp
      *.py
      *.sh
```

No new top-level `tests/cpp/**/test_*.cpp` or `tests/python/**/test_*.py` files are
accepted after the rewrite.

## Test Tiers

Each test belongs to exactly one tier.

| Tier | Meaning | Expected runtime | PR default |
| --- | --- | ---: | --- |
| `unit` | Small contract around one API or algorithm | milliseconds to seconds | yes |
| `integration` | Multiple subsystems or real runtime object | seconds | selected |
| `regression` | Specific historical failure or numerical oracle | seconds to minutes | selected |
| `backend` | Backend-specific behavior: MPI, OpenMP, GPU, native loader | variable | no, unless touched |
| `quality` | Sanitizers, coverage, static analysis, type checks | variable | pyright yes, others opt-in |

Tier is not a subsystem. Subsystems are separate labels.

## Labels

CTest labels and pytest markers share the same vocabulary. Required labels:

- Tier: `unit`, `integration`, `regression`, `backend`, `quality`.
- Domain: `mesh`, `amr`, `elliptic`, `runtime`, `numerics`, `codegen`,
  `python`, `bindings`, `io`, `diagnostics`, `time`, `physics`.
- Backend/runtime requirement: `kokkos_serial`, `kokkos_openmp`, `mpi`,
  `native_loader`, `gpu`, `hdf5`, `compiler`.
- Speed: `fast`, `medium`, `slow`.
- Safety/instrumentation: `asan`, `ubsan`, `tsan`, `coverage`.

Every test has at least:

```text
<tier>, <domain>, <speed>
```

Backend tests additionally declare all backend requirements.

## Manifest

Test metadata is centralized in `tests/test_manifest.toml`. CMake and pytest
configuration are generated from it or validated against it.

Example:

```toml
[[cpp.suite]]
name = "pops_mesh_unit"
sources = [
  "unit/mesh/box_test.cpp",
  "unit/mesh/multifab_test.cpp",
]
labels = ["unit", "mesh", "fast", "kokkos_serial"]

[[cpp.suite]]
name = "pops_amr_mpi_parity"
sources = ["integration/mpi/amr_parity_test.cpp"]
labels = ["backend", "amr", "mpi", "medium"]
mpi_nproc = [1, 2, 4]

[[python.suite]]
path = "tests/python/unit/codegen"
labels = ["unit", "codegen", "python", "fast"]

[[python.suite]]
path = "tests/python/integration/native_loader"
labels = ["integration", "codegen", "native_loader", "compiler", "medium"]
```

The manifest is the source of truth for CI selection. Name heuristics are not
accepted as the long-term selection mechanism.

## CMake API

The project provides wrappers instead of raw `add_executable` and raw
`add_test` calls.

```cmake
pops_add_gtest_suite(
  NAME pops_mesh_unit
  SOURCES
    unit/mesh/box_test.cpp
    unit/mesh/multifab_test.cpp
  LABELS unit mesh fast kokkos_serial
)

pops_add_mpi_gtest_suite(
  NAME pops_amr_mpi_parity
  SOURCES integration/mpi/amr_parity_test.cpp
  NPROCS 1 2 4
  LABELS backend amr mpi medium
)
```

Wrapper responsibilities:

- Link `pops::pops`, `GTest::gtest`, `pops_gtest_main`, and
  `pops::dev_options`.
- Attach labels, timeout, processors, resource locks, and environment.
- Register discovered GoogleTest cases with CTest.
- Add MPI test entries for each configured rank count.
- Apply Kokkos-specific compile flags and heavy translation unit pools when
  required.
- Fail configuration if required metadata is missing.

Raw `add_test` is forbidden outside wrapper implementation files.

## C++ Test Style

### Assertions

Use GoogleTest assertions directly:

```cpp
EXPECT_EQ(box.numPts(), 16);
EXPECT_THAT(error, NearRelative(0.0, 1e-12, 1e-14));
ASSERT_TRUE(result.converged) << result.summary();
```

Project numerical helpers live in `tests/cpp/support/numeric_matchers.hpp`.
Required helpers:

- absolute and relative floating comparison
- vector and field comparison with max norm
- bit identity comparison
- conservation check
- monotonicity/positivity check
- finite field check
- expected exception message check

### Fixtures

Shared setup is fixture-based:

- `MeshFixture`
- `MultiFabFixture`
- `EllipticMmsFixture`
- `SystemFixture`
- `AmrFixture`
- `NativeLoaderFixture`
- `MpiFixture`

Fixtures own setup and cleanup. Tests do not manually create temporary
directories, global caches, or compiler command fragments unless they are inside
a fixture.

### Parameterization

Repeated matrices are parameterized:

```cpp
class RiemannParityTest
    : public ::testing::TestWithParam<RiemannCase> {};

INSTANTIATE_TEST_SUITE_P(
    AllSupportedRiemannSolvers,
    RiemannParityTest,
    ::testing::Values(
        RiemannCase{"rusanov"},
        RiemannCase{"hll"},
        RiemannCase{"hllc"},
        RiemannCase{"roe"}));
```

The parameter name must appear in the failure output.

### Device and MPI Rules

GoogleTest assertions stay on the host. Device kernels return observable data
to host-side assertions.

MPI tests aggregate failures on rank 0 only when required, but each rank may
emit scoped diagnostics through a shared utility. Rank counts are CTest test
instances, not loops inside one test binary.

## Python Test Style

`pytest.ini` or `[tool.pytest.ini_options]` defines strict markers:

```toml
[tool.pytest.ini_options]
testpaths = ["tests/python"]
addopts = "-ra --strict-markers --strict-config"
markers = [
  "unit: small Python-only or pure API test",
  "integration: requires compiled _pops or multiple subsystems",
  "regression: historical bug or numerical oracle",
  "kokkos: requires a visible Kokkos install",
  "native_loader: compiles or loads a native shared object",
  "mpi: requires MPI",
  "slow: too slow for the default PR lane",
]
```

`tests/python/conftest.py` owns:

- import path setup for the built package
- `_pops` availability fixture
- Kokkos root fixture
- native compiler fixture
- MPI availability fixture
- deterministic RNG fixture
- temp build/cache directories
- skip behavior for unavailable optional backends

Skip behavior must be explicit and counted. A missing backend may skip a test
only when the test is not selected by a backend-required lane. In a backend lane,
missing backend setup is a failure.

## Pyright

The target type-checking setup is:

```text
pyrightconfig.json
python/pops/py.typed
python/pops/_pops.pyi
```

Policy:

- `python/pops` is checked in strict mode.
- `tests/python` starts in basic mode and moves to strict for shared fixtures
  and helpers.
- Generated or build-tree files are excluded.
- The `_pops` stub is treated as part of the public Python contract.

Pyright runs before Python runtime tests in CI. A Pyright failure blocks the
normal PR gate.

## Coverage Policy

Coverage is reported by tier and domain.

Minimum target thresholds after the rewrite:

| Area | Line coverage target | Branch coverage target |
| --- | ---: | ---: |
| C++ unit-covered headers | 85% | 70% |
| C++ numerics/mesh core | 90% | 75% |
| C++ runtime/AMR integration paths | 70% | 55% |
| Python package | 85% | 70% |
| Python codegen/control logic | 90% | 75% |

Coverage exceptions require an entry in `tests/coverage_exceptions.toml` with a
reason and owner.

Coverage reports include:

- HTML artifact
- XML artifact for CI integrations
- textual summary in CI
- top uncovered files by domain

## CI Matrix

### Required PR Gate

Runs on normal PRs:

1. `pyright`
2. `pytest -m "unit or integration and not slow and not native_loader and not mpi"`
3. `ctest -L "unit|fast"` on Kokkos Serial
4. affected integration/regression suites selected by manifest

### Full Gate

Runs on master, nightly, manual dispatch, or `ci-full`:

1. Required PR gate
2. full C++ GoogleTest suite on Kokkos Serial
3. full Python pytest suite
4. MPI suites with rank counts from the manifest
5. Kokkos OpenMP suites with bounded thread count
6. native loader suites

### Quality Gate

Runs weekly, manually, or on `quality` label:

1. clang-format
2. ruff
3. Pyright strict report
4. clang-tidy
5. ASan/UBSan
6. TSan on Kokkos OpenMP
7. C++ coverage
8. Python coverage
9. CodeQL

The target end state should make selected quality jobs blocking once their
finding count is intentionally reduced to zero or to an accepted baseline.

### GPU/ROMEO Gate

Runs outside generic GitHub runners:

1. Kokkos CUDA build
2. GPU C++ GoogleTest backend suites
3. GPU Python native loader smoke tests
4. MPI+GPU validation where hardware is available

GPU tests use the same manifest schema and labels. They are not a separate
unstructured script pile.

## Test Selection

The long-term selection algorithm reads `tests/test_manifest.toml`.

Inputs:

- changed files
- changed public headers
- changed Python modules
- changed bindings
- changed CMake/presets/toolchain files
- requested labels from PR labels or manual dispatch

Outputs:

- CTest labels or explicit test list
- pytest marker expression or explicit node ids
- required build presets
- required artifacts
- JSON explanation artifact containing changed files, inferred areas, expanded
  labels, selected tests, and selection reasons

Rules:

- Unknown source paths select the full relevant suite.
- Build system changes select all compiled tests.
- Public API changes select unit, integration, and regression tests for the
  affected domain.
- Test support changes select every suite that consumes the support component.
- Selection failure is a CI failure, never a silent zero-test pass.

Implementation:

- `scripts/ci_select_tests.py cpp` reads `tests/test_manifest.toml`, excludes
  MPI-only suites from the Serial gate, maps changed source paths to domain
  labels, and emits either selected CTest targets/regex or the full C++ suite.
- `scripts/ci_select_tests.py python` reads the same manifest, excludes
  architecture suites owned by the source-only quality job, maps changed Python
  package paths to suite labels, and emits the pytest files for a shard.
  `scripts/ci_shard_binpack.py` packs the selected files onto shards by measured
  duration (`tests/python/test_durations.json`, greedy longest-processing-time) so
  the slowest shard is minimized; the multi-compile DSL compile-cache test runs in a
  dedicated cached job and is excluded from the shard partition. `ci_select_tests.py
  verify` reconstructs every shard and fails CI unless they exactly cover the
  selection (no test dropped or duplicated).
- CI stores each selector decision as an uploaded JSON artifact so selection
  errors can be reviewed like build or test failures.

## Commands

User-facing commands are simple wrappers over CMake/CTest/pytest/Pyright:

```bash
pops test fast
pops test full
pops test cpp --label mesh
pops test py --label codegen
pops test mpi
pops test openmp
pops test pyright
pops test coverage
```

The wrappers print the underlying command before executing it. Direct
`cmake`, `ctest`, `pytest`, and `pyright` commands remain supported for CI and
debugging.

## Acceptance Criteria

The rewrite is complete when all conditions below are true:

1. No C++ test file contains a standalone `main()` except shared GoogleTest
   entry points.
2. No C++ test file contains local `fails` counters, `chk` lambdas, or copied
   tolerance helpers.
3. No Python test is launched as a raw script in CI.
4. pytest runs all Python tests through collection.
5. Pyright is configured and runs in the required PR gate.
6. Every test has tier, domain, and speed metadata.
7. MPI rank variants are represented as CTest tests.
8. Kokkos Serial and Kokkos OpenMP are explicit backend lanes.
9. GPU/ROMEO tests live under the same metadata model.
10. CI selection is manifest-driven, not filename-heuristic-driven.
11. Coverage reports are generated for C++ and Python.
12. Test documentation explains how to add a new test, label it, and decide
    which backend lanes it must run in.

## Rejected Alternatives

### GoogleTest as the only tool

Rejected. GoogleTest is not a build orchestrator, MPI launcher, Python test
runner, type checker, or coverage reporter. Using it for those roles would
create custom infrastructure worse than the standard tools.

### CTest-only tests

Rejected. CTest can run binaries, but it does not provide fixtures,
parameterized assertions, typed tests, or rich numerical failure diagnostics.
That is the current failure mode of the mini-harness.

### Catch2

Rejected as the primary C++ framework. Catch2 is pleasant and has good
section/tag ergonomics, but GoogleTest is the more conventional choice for a
large CMake/HPC codebase with typed and parameterized tests.

### doctest

Rejected as the primary C++ framework. Compile-time cost is attractive, but the
target architecture values large-suite organization, fixtures, CI conventions,
and standard contributor familiarity more.

### Python unittest

Rejected. pytest has better fixtures, markers, parametrization, skip handling,
and ecosystem support for this test matrix.

## References

- GoogleTest CMake quickstart: https://google.github.io/googletest/quickstart-cmake.html
- CTest command and labels: https://cmake.org/cmake/help/latest/manual/ctest.1.html
- pytest markers: https://docs.pytest.org/en/stable/how-to/mark.html
- pytest fixtures: https://docs.pytest.org/en/stable/explanation/fixtures.html
- Pyright configuration: https://github.com/microsoft/pyright/blob/main/docs/configuration.md
