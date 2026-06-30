# Contributing to adc_cpp

`adc_cpp` is the header-only C++23 core of the PoPS solver, with its Python bindings
(pybind11), its DSL path and its CMake packaging. This guide summarizes the workflow; the
technical detail lives in the [README](README.md) and in
[docs/DOC_QUALITY.md](docs/DOC_QUALITY.md).

## Build and tests (CMake presets)

The build is driven by the presets in `CMakePresets.json`, not by ad-hoc `-D` flags. The
`pops` conda env (Python 3.12) must be active for the `python`, `parallel` and `mpi` presets.

```bash
bash scripts/setup_env.sh && conda activate pops   # env + pinned toolchain

cmake --preset serial   && cmake --build --preset serial   && ctest --preset serial
cmake --preset python   && cmake --build --preset python    # _pops module (bindings)
cmake --preset mpi      && cmake --build --preset mpi      && ctest --preset mpi
cmake --preset parallel && cmake --build --preset parallel && ctest --preset parallel
```

The CI presets (`ci-serial`, `ci-python`, `ci-mpi`, `ci-kokkos`, `ci-kokkos-python`)
mirror `.github/workflows/ci.yml`: match your flags to a CI job rather than
inventing new ones. The GPU / GH200 paths cannot be validated outside ROMEO: say so
explicitly in the PR.

## CI modes

Use PR labels and commit tokens for validation modes; reserve Git tags for releases only.

| Situation | What runs |
| --- | --- |
| PR docs-only | `Docs PR` reset lint only; no `_pops`, Sphinx, Doxygen or Kokkos build. |
| PR CI / workflow / metadata only | Required `gate (agregation requise)` routing verdict only; no Kokkos build by default. |
| Push to `master` touching docs | Transitional `Docs` reset lint only; no Kokkos build. |
| PR touching C++ / Python | Required `gate (agregation requise)` through `ci.yml`. |
| PR with label `ci-kokkos` | Forces the Serial Kokkos C++ + Python gates even if only CI files changed. |
| PR with label `ci-full` | Adds MPI + Kokkos OpenMP + bench compile. |
| PR with label `quality` | Runs `quality.yml` static/deep checks. |
| Weekly cron / manual dispatch | Backstop CI/docs/quality lanes. |
| Git tag `vX.Y.Z` | Release creation and wheel packaging. |

If a full Sphinx/Doxygen site is reintroduced, keep it opt-in: use a PR label such as
`docs-full` for pre-merge validation and a master commit token such as `[docs]` for publish.
Do not make normal docs-only pushes compile Kokkos.

## Documentation

`bash scripts/build_docs.sh` runs the transitional documentation lint. The Sphinx/Doxygen
site was removed during the reset; the retained corpus and rebuild rules are described in
[docs/DOC_QUALITY.md](docs/DOC_QUALITY.md). The docs CI lanes now validate this reduced
corpus until the new documentation structure is introduced; they do not compile `_pops` or
Kokkos.

## Standards

The conventions are written down; follow the project's decision first, then the upstream guide.

- **C++ style**: the [Google C++ Style Guide](https://google.github.io/styleguide/cppguide.html),
  adapted. The ratified decisions (what we follow, adapt or drop, D1-D15) are in
  [docs/CODING_STANDARDS_DECISIONS.md](docs/CODING_STANDARDS_DECISIONS.md); `.clang-format` and
  `.clang-tidy` enforce the mechanical part. The `format` gate is blocking, so run the pinned
  clang-format 19 locally (`pipx install clang-format==19.1.7`) to match it; output drifts between
  major versions.
- **Python style**: the [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)
  for `python/pops/**`, the bindings, the DSL and the tests; `ruff` (see `quality.yml`) enforces the
  mechanical part.
- **Documentation style**: the Google documentation guide is vendored verbatim under
  [docs/docguide/](docs/docguide/) (philosophy, best practices, Markdown style, README files) and is
  the canonical reference. Follow it, in particular *Update Docs with Code* (change the docs in the
  same commit as the code), *Minimum viable documentation*, and *Duplication is evil* (link, do not
  re-document). Its lean companion, [docs/docguide/agile_documentation.md](docs/docguide/agile_documentation.md)
  (adapted from Scott Ambler), covers *whether*, *what*, and *when* to document at all: document late
  and stable, prefer executable specs and generated reference, and update only when it hurts.
  The local reset policy is in [docs/DOC_QUALITY.md](docs/DOC_QUALITY.md). Write in English
  (en-US), keep new docs ASCII-clean, and treat documentation as code.
- **Templates**: open a PR with [the PR template](.github/PULL_REQUEST_TEMPLATE.md) (its five
  questions) and file issues with the forms under `.github/ISSUE_TEMPLATE/`.
- **Automated checks** run before any review: the `ci.yml` gate (build and tests), `quality.yml`
  (format, warnings, clang-tidy, sanitizers, coverage, CodeQL), `no-ai-authors.yml`, and
  `check_docs.py`. See Code review below for how their results are handled.

## Workflow

- **Linear** is the source of truth for tasks: one `PoPS-NN` issue = one PR.
- Branch: `adc-<n>-short-description`. PR title: `PoPS-<n> Description`. PR body:
  `Fixes PoPS-<n>`.
- `master` is the default branch; never commit directly to it. Deliver through a branch or
  an isolated `git worktree` off `master`.
- Minimal diffs, scoped to the issue; no incidental reformatting.

## Branches

Trunk-based, a single trunk. `master` is the only long-lived branch and is protected (see
[Code review](#code-review)). Work happens on short-lived `adc-<n>-description` branches cut
from a Linear issue, merged by PR, then deleted automatically (`delete_branch_on_merge` is
on). There is no Git Flow (`develop`, `release/*`): with continuous delivery and no release
train, that structure buys nothing. The only long-lived exceptions are explicit, documented
work areas (for example the frozen `feat/perf-campaign-*` branches); a branch that outlives
its merged PR without such a reason is dead and gets pruned.

## Versioning

`adc_cpp` follows Semantic Versioning. The public API under guarantee, the bump rules
(PATCH / MINOR / MAJOR) and the release steps are in [docs/VERSIONING.md](docs/VERSIONING.md).
Two things on every notable PR: keep the version single-sourced (`project(VERSION)` in
`CMakeLists.txt`, never duplicated), and add a line to the `## [Unreleased]` section of
[CHANGELOG.md](CHANGELOG.md) under Added / Changed / Fixed (Keep a Changelog, ISO dates).

## Pull Request Guidelines

Keep the PR focused on one logical change. Open a Linear issue first for large work: new
model architecture, DSL change, AMR refactor, change of numerical defaults, new GPU/MPI
backend, boundary conditions, physical normalization. Do not mix refactoring, formatting,
documentation and numerical changes in one PR; formatting goes in its own PR (`style:
clang-format the solver module`).

Before requesting review: read your own diff, build and test locally, add or update tests
for behavior changes, update documentation for user-facing changes. For a change to a
solver, model, flux, Poisson, AMR, backend or the DSL, include numerical validation
(reference case, observed quantity, expected value, tolerance, reason) so a reviewer can
tell a normal difference from a silent model change.

The PR template asks five questions: what changed, why, how, how it was tested (with the
commands run), and what to focus on. For a multi-file PR, give a suggested review order.

Commit messages follow the [seven rules of a great commit message](https://cbea.ms/git-commit/):
separate subject from body with a blank line, keep the subject a short (~50 characters) imperative
with no trailing period, and wrap the body at 72 columns to explain what and why (not how). One
project adaptation: keep a lowercase `scope:` prefix (e.g. `flux: add HLL state reconstruction
helper`), so we do not capitalize the subject (cbeams rule 3 relaxed). A single-line subject is fine
for a focused change, since the PR description carries the full context. Avoid `fix`, `wip`,
`update`.

## Code review

`master` is protected: every change lands through a pull request, never a direct push, and
the required status check (`gate (agregation requise)`, the aggregating job of `ci.yml`) must
be green before a merge. Force-pushes and deletion of `master` are blocked.

Review is routed by zone through [`.github/CODEOWNERS`](.github/CODEOWNERS): the owner of a
touched directory is auto-requested as reviewer. Today @wolf75222 owns every zone; the per-zone
split documents the responsible owner per area and prepares reviewer routing as the team grows. Making
that approval a merge gate (the "require review from Code Owners" branch-protection rule) is a
governance decision, separate from the file itself.

Before merging a substantial PR, run an independent review pass (the `/code-review` skill, or
`pr-review-toolkit`) and act on its result: a High-severity finding is either fixed or
explicitly dismissed in the PR, with the reason. In a solo plus agents setting this pass is
the one independent check, so it stands in for a second pair of eyes; keep it to one reviewer
(human or agent) and one pass rather than chasing consensus. Treat every review comment as a
TODO: resolve it, or reply why not, before closing it.

### By change type

Different changes need a different focus (SWE-at-Google ch.10):

- **New code**: confirm it needs to exist (code is a liability; prefer reuse or deletion). Check
  the design was agreed, the public surface is tested, an owner exists, and CI runs it.
- **Behavioral change or optimization**: tests updated for the new behavior; for a numerical
  change include the validation (reference case, observed quantity, expected value, tolerance,
  reason); add a benchmark for a performance claim.
- **Bug fix or rollback**: scoped to the bug only (no drive-by changes), with a regression test
  that would have caught it, and small enough to roll back cleanly.
- **Refactor or large-scale change**: behavior-preserving. Review correctness and applicability,
  and do not expand the scope of an automated change.

## Guardrails

- **No AI author, committer or co-author** (Claude, Copilot, Anthropic, ...) anywhere in
  the history: the `no-ai-authors.yml` workflow rejects such commits at the source (the
  GitHub squash hoists `Co-authored-by` trailers). Use your default git identity.
- Documentation style: ASCII strict for user-facing docs, no em-dash anywhere; these
  rules are checked by `docs/check_docs.py` (run by `build_docs.sh` and the PR lane).

## License

By contributing, you agree that your contributions are published under the BSD-3-Clause
license (see [LICENSE](LICENSE)).
