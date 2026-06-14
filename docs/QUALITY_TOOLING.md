# Quality tooling and static analysis

`adc_cpp` has a static-analysis / code-quality chain that is **deliberately kept off the
critical path of PRs**. The `ci.yml` gate remains the authority on compilability and correctness
(tests) ; quality runs separately, without slowing down the development cycle.

Linear tracking : epic **ADC-105** (milestone *Code quality and hardened CI*).

## Triggers (`.github/workflows/quality.yml`)

The `Quality` workflow **never runs on a push or on an ordinary PR**. It is triggered on :

| Trigger | When |
| --- | --- |
| `schedule` (cron `0 4 * * 0`) | every **Sunday** ~04:00 UTC |
| `workflow_dispatch` | manually (*Actions* tab -> *Quality* -> *Run workflow*) |
| `quality` label on a PR | full **opt-in** pass on that PR (risky PR) |

> The `quality` label must exist in the repository :
> `gh label create quality --description "Declenche quality.yml sur cette PR" --color FBCA04`.
> The workflow only takes effect (cron, dispatch, label) once it is present on the default branch
> (`master`).

## Policy : informative first

At startup, **nothing is blocking** : findings appear as GitHub annotations, in the job summary
and as artifacts, but they do not make the run fail (no `-Werror`, `clang-format` in
`--dry-run`, `clang-tidy` without `WarningsAsErrors`). All CMake options are **OFF by default**
-> `ci.yml`, local builds and `adc_cases` are unchanged. Once the base is cleaned up, we will be able
to switch this or that job to blocking.

## The five jobs

| Job | Tool | Config | Preset / option |
| --- | --- | --- | --- |
| `format` | clang-format | `.clang-format` | -- (no build) |
| `warnings` | gcc `-Wall -Wextra …` | `cmake/AdcDevTooling.cmake` | preset `ci-warnings` (`ADC_ENABLE_WARNINGS`) |
| `tidy` | clang-tidy | `.clang-tidy` | preset `ci-kokkos` (compile DB) |
| `sanitizers` | ASan + UBSan | `cmake/AdcDevTooling.cmake` | preset `ci-asan` (`ADC_ENABLE_SANITIZERS`) |
| `codeql` | CodeQL C++ | suite `security-and-quality` | preset `ci-kokkos` (traced build) |

Warnings and sanitizers are carried by an **`INTERFACE adc_dev_options`** target that **only** the
internal targets link in `PRIVATE` (the ~140 tests via `adc_add_test`, the `_adc` module). The public
core `adc::adc` is never touched -> no flag leaks to consumers.

CodeQL is free here because the repository is **public** ; the results show up in
**Security > Code scanning**.

## Reproduce locally

```bash
# Style
clang-format --dry-run --Werror include/adc/**/*.hpp     # signale ; -i pour appliquer

# Warnings stricts (Kokkos requis : env conda 'adc' actif, ou KOKKOS_PREFIX pointant une install)
cmake --preset parallel -DADC_ENABLE_WARNINGS=ON
cmake --build --preset parallel

# Sanitizers ASan+UBSan
cmake --preset parallel -DADC_ENABLE_SANITIZERS=ON -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build --preset parallel
ASAN_OPTIONS=detect_leaks=0:detect_container_overflow=0 ctest --preset parallel --output-on-failure

# clang-tidy (après un configure qui exporte compile_commands.json)
run-clang-tidy -p build 'tests/.*\.cpp'
```

> In CI, the jobs reuse the Kokkos Serial install **cached** by the gate (`ci.yml`), via
> the composite action `.github/actions/setup-kokkos` (same cache key).

## Out of scope (future extensions)

- `clang-format` sweep of the whole base (massive reformat) -- separate PR.
- Switch to `-Werror` / blocking PRs -- after cleanup.
- TSan, coverage, include-what-you-use, cppcheck.
