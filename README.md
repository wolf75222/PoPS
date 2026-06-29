<div align="center">

# PoPS - Plasma-Oriented PDE Solver

**A model-free C++23 core for coupled hyperbolic-elliptic systems on adaptive (AMR) meshes.**

![C++20](https://img.shields.io/badge/C%2B%2B-20-blue?logo=cplusplus)
![CMake](https://img.shields.io/badge/CMake-3.21%2B-064F8C?logo=cmake)
![Backends](https://img.shields.io/badge/backends-MPI%20%7C%20Kokkos-orange)
![Python](https://img.shields.io/badge/python-3.12-3776AB?logo=python)
![License](https://img.shields.io/badge/license-BSD--3-green)

</div>

<p align="center">
  <img src="docs/banner_pops.png" alt="PoPS - Plasma-Oriented PDE Solver" width="100%">
</p>

---

PoPS is a compiled solver engine, not a Python numerical library and not a scenario repository.
Python authors inert typed objects: a physics/model module, a time `Program`, mesh layout
descriptors, finite-volume descriptors, field problems, outputs, and runtime parameters.
`pops.compile_problem(...)` lowers that assembly to one compiled problem artifact; a `pops.System`
or `pops.AmrSystem` installs that artifact with `sim.install(compiled, ...)`; `sim.step_cfl(...)`
advances with C++/Kokkos/MPI kernels. Python never runs a per-cell loop.

Named applications such as diocotron, Euler-Poisson, two-fluid, and benchmark setups live in
[`adc_cases`](https://github.com/wolf75222/adc_cases). This repository owns the reusable solver core,
the Python DSL that builds compiled artifacts, and the C++ runtime that executes them.

At the mathematical level, a case usually couples conservative states `U` to one or more elliptic
fields:

```
dU/dt + div F(U, fields, aux) = S(U, fields, aux)
D phi                         = f(U)
```

Field outputs are exposed through named auxiliary channels. The standard Poisson contract provides
`phi`, `grad_x`, and `grad_y`; a model may also declare named aux fields such as `B_z` or `T_e`.
All these names are metadata for the generated C++ path, not Python callbacks.

## Table of contents

- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Documentation](#documentation)
- [Versioning](#versioning)
- [Contributing](#contributing)
- [License](#license)

## Prerequisites

- **C++20** compiler: AppleClang 16+, GCC 13+, Clang 17+ (`nvcc_wrapper` for the CUDA target).
- **CMake >= 3.21**: the build is driven by presets ([CMakePresets.json](CMakePresets.json)).
- **[Kokkos](https://kokkos.org) 4.2+**: the only on-node backend, required. No need to
  pre-install it; if it is not found, CMake fetches and builds it (FetchContent).
- **MPI** *(optional, `-DPOPS_USE_MPI=ON`: halos and distributed FFT)*.
- **HDF5** parallel *(optional, `-DPOPS_USE_HDF5=ON`: DataWriter)*.
- **Python 3.12 + numpy** *(optional, the `pops` bindings; conda env via `scripts/setup_env.sh`)*.

Per-platform backend coverage and known pitfalls (macOS, CUDA, conda, CI runners):
[docs/BACKEND_COVERAGE.md](docs/BACKEND_COVERAGE.md).

## Installation

Three ways. Build-from-source details live in the
[installation guide](docs/sphinx/getting-started/installation.md) rather than inline here.

C++ core, via CMake presets:

```bash
git clone https://github.com/wolf75222/adc_cpp.git && cd adc_cpp
cmake --preset serial && cmake --build --preset serial && ctest --preset serial
```

The Ninja build already uses all available cores; pin it to fewer jobs on a constrained machine with
`cmake --build --preset serial -j<N>`. The serial test preset runs tests one at a time;
parallelize with `ctest --preset serial -j<N>` (`-j$(nproc)` on Linux, `-j$(sysctl -n
hw.ncpu)` on macOS), and add `--output-on-failure` for logs. Two other presets build a
parallel backend instead of the serial one (both read `$CONDA_PREFIX`, so the conda env must
be active):

```bash
cmake --preset parallel && cmake --build --preset parallel && ctest --preset parallel  # threaded, Kokkos OpenMP
cmake --preset mpi      && cmake --build --preset mpi      && ctest --preset mpi        # distributed, MPI
```

Each preset writes into its own folder (`build`, `build-kokkos`, `build-mpi`). Backends and
runtime thread control (`pops.set_threads()`) are covered in the
[installation guide](docs/sphinx/getting-started/installation.md).

Python module (`pops`): `scripts/setup_env.sh` creates the conda env and pins the platform
toolchain, then `scripts/build_python.sh` builds and installs the module in one command. It sizes
the heavy translation-unit pool, exports the discovery variables, and ends on `pops.doctor()`.
`pip install .` (scikit-build-core) drives the build directly if you prefer. Build-time backends are
selected by environment variables (`POPS_USE_MPI`, `Kokkos_ROOT`, ...); user-facing compile choices
inside Python should be typed objects such as `Production()`, not backend strings.
`scripts/uninstall_pops.sh` reverses the setup scripts when you want a clean teardown.

```bash
bash scripts/setup_env.sh      # conda env + toolchain
bash scripts/build_python.sh   # build + install, then pops.doctor()
bash scripts/uninstall_pops.sh # full teardown (env + caches); --keep-env drops only the module
# or, by hand:  pip install .  # see the installation guide for backends
```

Released versions and binaries: the
[Releases page](https://github.com/wolf75222/adc_cpp/releases).

## Usage

<p align="center">
  <img src="docs/anim_romeo_diocotron_amr3.gif" alt="Diocotron instability, 3-level AMR, on ROMEO" width="480">
</p>

<div align="center">
<sub>
Validation scenario: diocotron instability (E x B drift) on a 3-level nested AMR hierarchy, ROMEO (96 cores).
The scenario itself lives outside this core repository:
<a href="https://github.com/wolf75222/adc_cases/tree/master/diocotron_amr"><code>adc_cases/diocotron_amr</code></a>.
</sub>
</div>

### From a C++ project

The C++ core is header-only for consumers and is consumed via `find_package(pops)` or FetchContent:

```cmake
include(FetchContent)
FetchContent_Declare(adc_cpp GIT_REPOSITORY https://github.com/wolf75222/adc_cpp.git)
FetchContent_MakeAvailable(adc_cpp)   # adc_cpp's own tests are not built for the consumer
target_link_libraries(my_app PRIVATE pops::pops)
```

Define a type that satisfies the `PhysicalModel` concept and compose it with the C++ coupling and
time machinery. This is the low-level engine path. Most users should author typed Python
model/program objects and let PoPS generate the corresponding C++ compiled problem artifact.

### From Python

The public Python path is typed and compiled. Strings name user objects such as blocks, fields, and
program nodes; typed objects choose algorithms and routes. The reduced example below couples a
scalar density to a Poisson field and advances it through a generated C++ program:

```python
import numpy as np
import pops
from pops.time import Program
from pops.lib.time import ssprk3
from pops.physics import Model
from pops.math import laplacian, grad, div, ddt
from pops.mesh.cartesian import CartesianMesh
from pops.mesh.layouts import Uniform
from pops.solvers.elliptic import GeometricMG
from pops.numerics.riemann import Rusanov
from pops.numerics.reconstruction.limiters import Minmod
from pops.numerics.spatial import spatial
from pops.codegen import Production

m = Model("diocotron")
U = m.state("U", components=["ne"], roles={"ne": "density"})
(ne,) = U
phi = m.field("phi")
m.solve_field("fields_from_state",
              equation=(-laplacian(phi) == ne),
              outputs={"phi": phi, "grad_x": grad(phi).x, "grad_y": grad(phi).y},
              solver=GeometricMG())
E = m.vector_field("E", x=-grad(phi).x, y=-grad(phi).y)
flux = m.flux("F", on=U, x=[ne * E.y], y=[ne * (-E.x)], waves={"x": [E.y], "y": [-E.x]})
m.rate("explicit_rate", ddt(U) == -div(flux))
m.check()

time = Program("advance")
ssprk3(time, "ne")

mesh = CartesianMesh(n=96, L=1.0, periodic=True)
layout = Uniform(mesh)
module = m.to_module()

compiled = pops.compile_problem(model=module, time=time, backend=Production(), layout=layout)
sim = pops.System(n=96, L=1.0, periodic=True)
sim.install(
    compiled,
    instances={
        "ne": {
            "model": module,
            "initial": ne0,  # ne0: initial density (2D array)
            "spatial": spatial.FiniteVolume(reconstruction=Minmod(), riemann=Rusanov()),
        },
    },
    solvers={"phi": GeometricMG()},
)
while sim.time() < 0.1:
    sim.step_cfl(0.4)
sim.write("ne.npz", format="npz")                   # save the block states (npz; "vtk" also available)
```

For an adaptive run, swap the layout to `pops.mesh.layouts.AMR(mesh, max_levels=2, ratio=2)` and
author the refinement with typed `pops.mesh.amr` policies. The compiled artifact is installed on
the matching runtime with `sim.install(compiled, ...)`. Users do not pass a public target string.
Step-by-step tutorial: [getting-started/tutorial](docs/sphinx/getting-started/tutorial.md).
Reference: [native-bricks](docs/sphinx/reference/native-bricks.md),
[symbolic-dsl](docs/sphinx/reference/symbolic-dsl.md),
[public API contract](docs/sphinx/reference/public-api-contract.md).

## Documentation

- User guide (Sphinx): <https://wolf75222.github.io/adc_cpp/>
- C++ reference (Doxygen): <https://wolf75222.github.io/adc_cpp/cpp/>
- Canonical guides: [ARCHITECTURE](docs/ARCHITECTURE.md) (layers, modules, AMR),
  [ALGORITHMS](docs/ALGORITHMS.md) (methods, formulas), [CHOICES](docs/CHOICES.md) (design),
  [BACKEND_COVERAGE](docs/BACKEND_COVERAGE.md) (backend / test matrix),
  [VALIDATION](docs/VALIDATION.md).
- Documentation policy (taxonomy, tooling, update guide): [DOC_QUALITY](docs/DOC_QUALITY.md).

### Core layers

| Layer | Role | Entry point |
|---|---|---|
| `python/pops/physics`, `python/pops/model`, `python/pops/time` | typed Python authoring: physics facade, operator-first model IR, and compiled time programs | [python/pops/physics](python/pops/physics) |
| `python/pops/mesh`, `python/pops/fields`, `python/pops/solvers`, `python/pops/numerics` | descriptors for layouts, AMR policies, field problems, solvers, Riemann fluxes, reconstruction, and finite-volume spatial choices | [python/pops/mesh](python/pops/mesh) |
| `python/pops/codegen` | validation, inspection, generated C++ emission, cache keys, and `.so` loading | [python/pops/codegen](python/pops/codegen) |
| `include/pops/core` | C++ concepts, state layout, model contracts, and equation blocks | [physical_model.hpp](include/pops/core/model/physical_model.hpp) |
| `include/pops/numerics` | C++ finite-volume, elliptic, time, Krylov, reconstruction, and Riemann kernels | [include/pops/numerics](include/pops/numerics) |
| `include/pops/amr`, `include/pops/mesh`, `include/pops/parallel` | C++ mesh hierarchy, AMR clustering/regrid, MultiFab storage, halos, MPI seams, and reflux support | [include/pops/amr](include/pops/amr) |
| `include/pops/runtime`, `python/pops/runtime` | C++ runtime facade used by `pops.System(...).install(compiled, ...)` / `pops.AmrSystem(...).install(compiled, ...)` | [system.hpp](include/pops/runtime/system.hpp) |

### Ecosystem

| Repo | Role |
|---|---|
| `adc_cpp` (this repo) | reusable PoPS core, Python DSL, codegen, C++/Kokkos/MPI runtime, AMR infrastructure |
| [`adc_cases`](https://github.com/wolf75222/adc_cases) | named applications, validation cases, run scripts, scenario-specific facades |
| [`poisson_cpp`](https://github.com/wolf75222/poisson_cpp) | Poisson solvers (Thomas, SOR, CG, DST, multigrid) |
| [`advection_cpp`](https://github.com/wolf75222/advection_cpp) | advection, Burgers, Chorin Navier-Stokes |
| [`euler_cpp`](https://github.com/wolf75222/euler_cpp) | 2D Euler, viscous Navier-Stokes, plasma sources |

## Versioning

PoPS follows [Semantic Versioning](https://semver.org). The public API under guarantee and
the bump rules are declared in [docs/VERSIONING.md](docs/VERSIONING.md). Available versions and
their change logs: the [Releases page](https://github.com/wolf75222/adc_cpp/releases) and
[CHANGELOG.md](CHANGELOG.md). The project is in `0.y.z` initial development: the public API may
still change until `1.0.0`.

## Contributing

Build, test and workflow conventions: [CONTRIBUTING.md](CONTRIBUTING.md).

## License

BSD-3-Clause. See [LICENSE](LICENSE).
