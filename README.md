# PoPS - Plasma-Oriented PDE Solver

PoPS is the reusable core of `adc_cpp`: a C++/Kokkos/MPI solver engine with
Python bindings and a typed authoring layer for coupled hyperbolic-elliptic
systems on uniform or AMR meshes.

The documentation corpus is being rebuilt from a clean base. The previous
Sphinx site, archived design notes, validation figures, and historical audit
pages have been removed so the next documentation pass can start from the
current code instead of patching stale prose.

## Kept documentation

- [Architecture](docs/ARCHITECTURE.md): current technical map of the core.
- [Algorithms](docs/ALGORITHMS.md): numerical methods and implementation notes.
- [Versioning](docs/VERSIONING.md): public API scope and release process.
- [Documentation quality](docs/DOC_QUALITY.md): rules for rebuilding the docs.
- [Bibliography](docs/BIBLIOGRAPHY.md): external references.
- [Contributing](CONTRIBUTING.md): build, test, review, and PR workflow.
- [Security](SECURITY.md): vulnerability reporting policy.
- [Changelog](CHANGELOG.md): notable changes.

The Google documentation guide remains vendored under [docs/docguide](docs/docguide/).

## Build

The project is driven by CMake presets. The standard local checks remain:

```bash
cmake --preset serial
cmake --build --preset serial
ctest --preset serial
```

The documentation reset check is intentionally lightweight until the new site
is rebuilt:

```bash
bash scripts/build_docs.sh
```

## License

BSD-3-Clause. See [LICENSE](LICENSE).
