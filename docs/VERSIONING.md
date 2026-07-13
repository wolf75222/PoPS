# Versioning

`PoPS` follows [Semantic Versioning 2.0.0](https://semver.org). Package SemVer and the independently
evolving API, semantic IR, normalization, component registry, native ABI, and checkpoint schema
revisions are recorded by `schemas/release_contract.v1.json` and generated for Python/C++.

## Single source of the version number

The version lives in one place: `project(VERSION x.y.z)` in `CMakeLists.txt`. Everything
derives from it: `pops.__version__` (available before native loading and authenticated against the
`POPS_VERSION` baked into `_pops`), the pip wheel
(scikit-build-core regex on `pyproject.toml`), and `adcConfigVersion.cmake`. Do not
duplicate the number elsewhere. When the generated documentation site is rebuilt, it must keep
deriving the published version from this same `project(VERSION)` value.

## Public API (under SemVer guarantee)

What a version bump is allowed to break is exactly this surface:

- C++ runtime facade: `pops::System`, `pops::AmrSystem` and their public methods (block
  composition, `set_poisson`, `set_refinement`, stepping).
- The concepts a model composes against: `PhysicalModel`, `PhysicalFlux`, `NumericalFlux`,
  `SpatialOperator`, `EllipticSolver`,
  and the named generic bricks in `include/pops/physics/`.
- Python bindings: the documented `pops.*` surface (`pops.Model`, `pops.Case`, `pops.compile`,
  `pops.bind`, `pops.physics.facade.Model`, the brick classes, `pops.doctor`, `pops.set_threads`,
  `pops.parallel_info`, `pops.has_kokkos`, `pops.__version__`). The runtime engines `System` /
  `AmrSystem` are internal seams behind `pops.compile` / `pops.bind` (reachable as
  `pops.runtime.system.*`); they carry no SemVer guarantee and may change.
- DSL surface: the fixed aux names (`phi`, `grad_x`, `grad_y`, `B_z`, `T_e`) and the
  documented builders.
- Component interchange: the current `ComponentManifest` schema, canonical identity domains, and
  generated builtin component catalog. New optional component capabilities are additive; changing
  the meaning or required shape of an existing semantic field is breaking.
- Consumable CMake: the `pops::pops` target, `find_package(pops)`, and the documented options
  (`POPS_USE_MPI`, `POPS_USE_HDF5`, `POPS_USE_KOKKOS`, ...) and presets.

## Internal (no guarantee, may change in any release)

Private helpers, memory layouts and `Fab` / `MultiFab` internals, the DSL code-generation
internals and the production `.so` ABI key, test harnesses, benchmarks, and anything not in
the list above. The ABI key intentionally invalidates the DSL cache across toolchains; that
is not a SemVer-relevant break.

## Bump rules

- PATCH (`x.y.Z`): bug fixes with no change to the public API.
- MINOR (`x.Y.0`): backward-compatible additions to the public API (new bricks/catalog rows, new
  options, new Python surface, or a new versioned extension schema).
- MAJOR (`X.0.0`, post-1.0): a break of the public API or of the production DSL ABI.

While in `0.y.z` initial development, `0.y` is the compatibility boundary: `0.3.z` may satisfy a
`0.3` consumer, while `0.4.0` must not. The generated CMake package therefore uses
`SameMinorVersion` before 1.0 and `SameMajorVersion` from 1.0 onward.

Schema and ABI compatibility is exact unless the owning protocol explicitly defines a migration.
Runtime loaders never infer compatibility from package SemVer. Historical artifacts may be handled
only by an offline migration tool that emits a complete current artifact.

## Supported release matrix

The normative matrix is the generated `SUPPORTED_MATRIX` projection of
`schemas/release_contract.v1.json`. It currently promises Python 3.12, C++20, Kokkos 4.4.01 Serial
and OpenMP source builds, a Serial OpenMPI source lane, and a macOS arm64 CPython 3.12 Serial wheel.
CUDA/HIP, MPI and Windows wheels are explicitly not promised. A release may narrow or extend this
matrix only by changing the versioned contract and proving every declared lane.

## Releasing

1. Bump `project(VERSION x.y.z)` in `CMakeLists.txt`. The Python `__version__` and the pip
   wheel derive from it automatically; nothing else is edited by hand.
2. Move the `## [Unreleased]` entries of [CHANGELOG.md](../CHANGELOG.md) into a
   `## [x.y.z] - YYYY-MM-DD` section.
3. Run `python scripts/generate_release_contract.py --check` and the release preflight; a missing
   build/codesign/example/conformance evidence record blocks tagging.
4. Merge, then `git tag vx.y.z` on master and `git push --tags`. The `release.yml` workflow
   turns the tag into a GitHub Release built from that CHANGELOG section.
