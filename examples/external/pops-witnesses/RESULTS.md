# ROMEO execution results

## Accepted campaign

Executed on 2026-10-09 against frozen #681 revision
**620ac6b6d2ad532a7616652d1bdaac9ca507287a**. The six native captures in
campaign-7 and their separate integrity/math reader passed on ROMEO.
The core source, installed Python/SDK payload, native binary, compiler and wheel
identities stayed equal before/after each execution and across all six cases.
The contribution consists of the external package and a changelog entry.

WSL Ubuntu holds and publishes the isolated contribution checkout. Every native
build, witness execution and test reported below ran through ROMEO MCP SLURM
jobs on x64cpu, node romeo-c040. No Windows build/test was used.
Earlier macOS builds and results from other commits remain historical evidence;
none is counted as validation of this revision.

| Completed job | Work and observed result |
| --- | --- |
| 738859 | Frozen core build, wheel installation, installed-native Dim2/Serial verification passed; 6 CPUs, 32 GiB, elapsed 3m23s |
| 738890 | Complete baseline capture and independent baseline reader passed; 1 CPU, 4 GiB, elapsed 58s |
| 738891 | Five further native captures and complete six-case independent reader passed; 1 CPU, 4 GiB, elapsed 4m38s |
| 738899 | Ruff check/format, 34 tests, packaging manifest and docs checks passed; elapsed 22s, tests 18.28s |
| 738902 | Independent campaign reader and final unchanged-core/library-manifest check passed; summary collected in 4s |

### Numerical observations

Cell means are stored as binary64 arrays of shape (1,12,16). Every one of the
12 arrays per capture was compared with the independent full-grid reference.
All field residual, Q=3 and RHS=4 premises passed. The largest absolute q at
any initial/intermediate/accepted state was **2.1163218149826304**, below 3.

| Native variant | Max accepted-q error against oracle | Fixed accepted-q budget | Max saved-field residual |
| --- | --- | --- | --- |
| baseline | 6.661338147750939e-16 | 2.458583115919689e-13 | 5.089706434091568e-12 |
| linear, linear-renamed, beta-zero | 4.440892098500626e-16 | 2.46885141980481e-13 | 6.091127602303459e-12 |
| cubic, cubic-renamed | 4.440892098500626e-16 | 2.469045197548255e-13 | 5.217160037318536e-12 |

| Independent comparison | Observed | Fixed comparison budget |
| --- | --- | --- |
| Max accepted-q gap, baseline -> linear | 2.818743741528351e-6 | 4.927434535724498e-13 |
| Max accepted-q gap, linear -> cubic | 4.738371638923766e-5 | 4.937896617353065e-13 |
| First-stage cubic-gap cosine coefficient (2,0) | -1.1307020207973311e-7 | 6.061295170246945e-13 |

The independently predicted cosine coefficient is -1.1307020207527317e-7:
absolute observation/prediction difference about 4.46e-18. The coefficient uses
the projection and cell-centered phase specified in [README.md](README.md).
Both whole-array first-stage relations passed:
h epsilon (phi0-psi0) and -h beta q0^3.
Final gaps and the harmonic exceeded 100 times their frozen budgets.
Renaming every exercised owner/state/field/publication left all canonical
arrays byte identical. beta=0 reproduced linear arrays byte exactly.
The actual bound donor remained byte identical throughout.

The cube is (stored cell mean)^3. These observations establish the selected
discrete CPU Dim2, serial, frozen-donor SSPRK3 composition. They do not establish
continuous-PDE accuracy, convergence order, another temporal composition,
nonlinear Fields, evolving joint owners, GPU/MPI/AMR execution or owner acceptance.

### Formula, compiler and loaded-image observations

The real compiler-input receiver TU contains these statements at lines 567,
752 and 938, once for each stage. These are retained input bytes, not a later
dump_cpp regeneration:

~~~cpp
// linear
outA(index, 0) = (((0.25 * screened_first) + (0.35 * screened_second)) - (0.8 * q));
// cubic
outA(index, 0) = ((((0.25 * screened_first) + (0.35 * screened_second)) - (0.8 * q)) - (((0.05 * q) * q) * q));
~~~

The complete original paths, compiler invocations, input-file hashes, raw loader
maps and loaded binary identities are in [receipt.json](receipt.json) and the
original retained capsule. The receiver input TU SHA256 changed from
c3e895f5c1e28b14264b1ce3e14902804710285b9768ba4139aedb78ae644528
(linear) to 4689e54bfddf54fa63a910c61b33fc744df65c6dce92e15cc5a0430bf0a84317
(cubic). The aggregate compiler-input identities also differ baseline/linear/cubic.

| Loaded receiver model | SHA256 of image actually mapped |
| --- | --- |
| baseline | 29e32682828537acef590a35c1206e3e03355883e5343a717e4603db51655e0c |
| linear | d819120e3aef0ffa374b0d78145853c979abb020a9d3f501ca57a79702b2bded |
| cubic | e685e92f744d781a1322429c360f4ca143bbe9a18998caa46d4929b1fe501b30 |

PoPS loads model DSOs through sealed /tmp/pops-native-* images. Their actual
mapped paths were recorded and bytes matched to the catalogue artifact and
successful compiler output. Program DSOs were likewise observed and matched.
The core DSO was mapped from the installed venv package.

## Native environment and unchanged core

Private ROMEO venv: Spack Python 3.12.5, GCC 14.2.0, CMake 3.31.10,
Ninja 1.13.2, Kokkos 4.4.01 Serial, NumPy 2.5.3, pybind11 2.13.6,
pytest 9.1.1, Ruff 0.16.10 and scikit-build-core 1.1.1.
CMake was privately installed through pip after the available Spack CMake was
unusable. Kokkos was built from the official release asset whose SHA256 matches
the frozen CMakeLists.txt. Neither adjustment edited PoPS.

The existing scikit-build/CMake entry point built the wheel with
POPS_NATIVE_DIM=2, POPS_USE_MPI=OFF, POPS_USE_HDF5=OFF,
POPS_HEAVY_MODULE_TU_POOL=3, POPS_USE_CCACHE=OFF and exported compile commands.
GCC's own libstdc++ directory was prepended to LD_LIBRARY_PATH before NumPy/PoPS
imports. Public serial bind used empty resources.

| Retained identity | SHA256 |
| --- | --- |
| Loaded native core | a96d95355709d4d75e7f16a9c486028a5a552df27c09572ed508eab2164b0b62 |
| Installed wheel | 4ae8e2f139ff6e5b1faa1f43cd771d0fb74bd01da3a0ad1f573d73180b847ca3 |
| Selected 1,274 source files | 7b30f78aebc10559dd1d88120cef37455e9c8173d0271d0ae863df7f8faf47df |
| Installed 1,177 payload files | ba1e1a5eb04b0a8404a4b90ba1fecae8958687b74edbc243d2060f956f517765 |

The two aggregate fingerprints hash UTF-8 json.dumps(file_hash_mapping,
sort_keys=True) with Python's default separators. Each original capture retains
the per-file mappings. Selected source files include python, include, src,
cmake, scripts and the core build definitions; bytes equal Git objects at the
frozen revision. Extra core source files are rejected.
Installed Python/SDK bytes were checked against source, and the JIT include
path/header signature against the actually installed native ABI.

Native ABI:
~~~text
compiler=14.2.0;std=202002L;headers=087fead68988afda3c730e232f15b58ded9a07a1e4387b212c33248ac7661626;kokkos=1;stdlib=libstdc++_20240801;mpi=0;mpi_abi=off;dim=2
~~~

The independent reader establishes integrity relative to the trusted execution
journal and compiler/loader evidence. Hashes and file modes do not establish
origin against a producer able to replace outputs, journals and hashes together.
No formal compiler-to-binary graph proof or third-party computational attestation
is claimed.

## Checks executed

In the recorded ROMEO venv, from the frozen source root:

~~~bash
env -u PYTHONPATH python scripts/verify_installed_native.py --expect-dim 2 --expect-serial --json
python -m ruff check --config pyproject.toml examples/external/pops-witnesses
python -m ruff format --check --config pyproject.toml examples/external/pops-witnesses
env -u PYTHONPATH OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python -m pytest examples/external/pops-witnesses/tests -q --junitxml="$ROOT/external-unit-final-reviewed.xml"
python scripts/check_packaging_manifest.py
python docs/check_docs.py
~~~

34 tests passed, including dense/FFT and affine-modal cross-checks, nonlinear
harmonics, rejection of q^2/linearized/frozen-field substitutes and integrity
failure when replacement oracle arrays still pass the mathematical comparison.
Packaging: OK (172 api, 7 abi, 19 sdk-root, 186 sdk-support, 9 test-only headers).
Docs: exit 0; four existing freshness warnings affect root README.md,
docs/ARCHITECTURE.md, docs/ALGORITHMS.md and
docs/design/SPECIFICATION_TECHNIQUE_FINALE_POPS_ARCHITECTURE.md.
The complete root test suites and GitHub CI were not run in this campaign.

See README for the exact six capture commands and the separate reader command.
Each case used a fresh process/cache and a new empty output directory.

## Retention and unsuccessful attempts

Original campaign:
 /scratch_p/nimarano/pops-witnesses-620ac6b6/campaign-7

Longer-lived private capsule:
 /home/nimarano/pops-witnesses-620ac6b6-evidence.tar.gz

[retention.json](retention.json) records the capsule SHA256 and archive job.
The capsule contains the original arrays/bind inputs/checkpoint, actual compiler
input trees/invocations/outputs, loader maps and copied loaded DSOs, core Git
archive, wheel, CMakeCache.txt/compile_commands.json, external source, JUnit
reports, bootstrap logs/environment and available SLURM scripts/logs.
Large native artifacts are retained on ROMEO and are not committed to Git.
The archive is private to the ROMEO account; readers without access must request
the capsule or rerun the published commands.

These failed attempts remain distinct from accepted campaign-7:

| Jobs | Preserved failure / correction |
| --- | --- |
| 738852, 738853 | Spack CMake resolution/dependency failures; private pip CMake used |
| 738855 | Tag tarball hash differed from pinned release-asset bytes; official release asset used |
| 738867, 738870 | External file transport quoting / lint failures; no successful witness claimed |
| 738871 | Earlier 33-test pass, preceding the report-bytes preservation test |
| 738873 | Older libstdc++ loaded before GCC14 core; job library path corrected |
| 738875 | GPFS rename-no-replace EINVAL while checkpointing; public checkpoint moved to node-local /tmp and all original checkpoint bytes copied |
| 738878, 738883 | Catalogue paths were insufficient to identify mapped model images; raw maps exposed sealed copies |
| 738884 | Native Program report contained bytes; external evidence serializer now preserves hex bytes |
| 738886 | 34 unit tests passed but incomplete loaded-path receipt rejected by the independent reader; this job is not accepted native proof |
| 738900 | Collection shell quote error; summary collection corrected in 738902 |
| 738907 | Final docs check rejected the not-yet-created retention pointer; the pointer is now created before validation and excluded from its own capsule |

campaign-1 through campaign-6 and all available failed-job logs are retained.
No failed attempt was reused as an accepted receipt. No core patch was required.
