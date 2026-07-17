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

- Python lifecycle: exactly `Model`, `Program`, `Case`, `validate`, `inspect`, `explain`, `resolve`,
  `compile`, `bind`, `run`, and `__version__` at package root. `pops.run` is the sole execution
  transition; native engines and imperative stepping methods are implementation details.
- Documented typed authoring protocols: qualified handles, immutable expressions, component
  interfaces, numerical/layout descriptors, `Program.solve(problem, solver=...)`, and the
  consumer/checkpoint declarations used by the final examples. A catalog row is public only when
  it has an executable lowering; unavailable or planned placeholders are not API.
- C++ component interchange: the current `ComponentManifest` schema, generated component-interface
  vocabulary, canonical identity domains, external package contract, and generated builtin
  component catalog. New optional component capabilities are additive; changing the meaning or
  required shape of an existing semantic field is breaking.
- Native component tables have two independent axes: the envelope protocol ABI and each interface
  version. Adding a new table or a new version is additive; changing a published table layout,
  operation semantics or required POD field without a new interface version is breaking. Loaders
  match exact `(interface_id, interface_version, table_size)` tuples and never infer compatibility
  from package SemVer. Shared request/value structs have their own generated common-ABI version;
  that version participates in the catalog digest, so a binary built against another common layout
  is rejected before any table is prepared or invoked.
- Consumable generic C++ concepts and component interfaces documented for external implementations.
  Concrete runtime engines (`System`, `AmrSystem`, AMR couplers), their builders, and their stepping
  or block-registration methods are internal seams behind the Python lifecycle and carry no SemVer
  guarantee.
- Consumable CMake: the `pops::pops` target, `find_package(pops)`, and the documented options
  (`POPS_USE_MPI`, `POPS_USE_HDF5`, `POPS_USE_KOKKOS`, ...) and presets.

## Internal (no guarantee, may change in any release)

Private helpers, native runtime facades, memory layouts and `Fab` / `MultiFab` internals, fixed
model-specific auxiliary names, code-generation internals and the production `.so` ABI key, test
harnesses, benchmarks, and anything not in the list above. The ABI key intentionally invalidates
the compiled cache across toolchains; that is not a SemVer-relevant break.

## Bump rules

- PATCH (`x.y.Z`): bug fixes with no change to the public API.
- MINOR (`x.Y.0`): backward-compatible additions to the public API (new bricks/catalog rows, new
  options, new Python surface, or a new versioned extension schema).
- MAJOR (`X.0.0`): a break of the public API or of the production DSL ABI.

The released `1.x` line uses `SameMajorVersion` in the generated CMake package. Historical `0.y.z`
artifacts used `0.y` as their compatibility boundary and `SameMinorVersion`; that pre-1.0 rule is
retained only so old package metadata can be interpreted, not as the policy of the current release.

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
