"""Keep the reproducible local MPI stack aligned with ParaView's bundled ABI."""

from __future__ import annotations

import json
import pathlib
import tomllib


ROOT = pathlib.Path(__file__).resolve().parents[3]


def test_conda_and_pixi_pin_the_paraview_611_mpich_stack():
    environment = (ROOT / "environment.yml").read_text(encoding="utf-8")
    pixi = tomllib.loads((ROOT / "pixi.toml").read_text(encoding="utf-8"))
    presets = json.loads((ROOT / "CMakePresets.json").read_text(encoding="utf-8"))

    assert "- mpich=4.1.2" in environment
    assert "- mpi=1.0=mpich" in environment
    assert 'hdf5=1.14.3=mpi_mpich_*' in environment
    assert "- openmpi" not in environment
    assert "mpi_openmpi_" not in environment

    dependencies = pixi["dependencies"]
    assert dependencies["mpich"] == "4.1.2.*"
    assert dependencies["mpi"] == {"version": "1.0.*", "build": "mpich"}
    assert dependencies["hdf5"] == {"version": "1.14.3.*", "build": "mpi_mpich_*"}
    assert "openmpi" not in dependencies

    mpi_preset = next(
        preset for preset in presets["configurePresets"] if preset["name"] == "mpi"
    )
    assert "MPICH 4.1.2" in mpi_preset["description"]


def test_paraview_launcher_uses_one_neutral_python_and_the_active_mpich_stack():
    launcher = (ROOT / "scripts/paraview_python.sh").read_text(encoding="utf-8")
    live_probe = (ROOT / "scripts/check_catalyst_live_mpi.sh").read_text(encoding="utf-8")
    host = (ROOT / "scripts/paraview_python_host.c").read_text(encoding="utf-8")
    build = (ROOT / "scripts/build_python.sh").read_text(encoding="utf-8")
    setup = (ROOT / "scripts/setup_env.sh").read_text(encoding="utf-8")
    bootstrap = (ROOT / "python/pops/_paraview_python_bootstrap.py").read_text(
        encoding="utf-8")

    assert '"$CONDA_PREFIX/bin/mpiexec" -n "$MPI_RANKS"' in launcher
    assert "DYLD_FALLBACK_LIBRARY_PATH" in launcher
    assert "POPS_ACTIVE_MPI_LIBRARY" in launcher
    assert "POPS_ACTIVE_PMPI_LIBRARY" in launcher
    assert "POPS_ACTIVE_HDF5_LIBRARY" in launcher
    assert '"$PARAVIEW_LIB/python$POPS_MINOR/site-packages"' in launcher
    assert 'LD_LIBRARY_PATH="$MPI_OVERLAY:$PARAVIEW_LIB"' in launcher
    assert 'MPI_OVERLAY="$CACHE_DIR/mpi-overlay"' in launcher
    assert 'LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$PARAVIEW_LIB' not in launcher
    assert "HOST_SOURCE_HASH" in launcher
    assert "env PYTHONPATH= PYTHONNOUSERSITE=1" in launcher
    assert "pops._paraview_python_bootstrap" in launcher
    assert '"$PARAVIEW_ROOT/bin/pvpython"' not in launcher
    assert '"$PARAVIEW_ROOT/bin/pvbatch"' not in launcher
    assert 'dlopen(library, RTLD_NOW | RTLD_GLOBAL)' in host
    assert 'dlsym(handle, "Py_BytesMain")' in host
    assert 'load_optional_global("POPS_ACTIVE_MPI_LIBRARY"' in host
    assert 'load_optional_global("POPS_ACTIVE_HDF5_LIBRARY"' in host
    assert "POPS_PARAVIEW_PMPI_LIBRARY" not in host
    assert "MPICH_CC=/usr/bin/clang" in setup
    assert 'export MPICH_CC="${MPICH_CC:-${CC:-/usr/bin/clang}}"' in build
    assert bootstrap.index("from pops import _pops") < bootstrap.index(
        '__import__("catalyst")')
    assert "sys.path[0] = str(script.parent)" in bootstrap
    assert "sys.path[:] = previous_path" in bootstrap
    assert "probe_catalyst_live_client.py" in live_probe
    assert "probe_catalyst_live_mpi.py" in live_probe
    assert '"$PVPYTHON" --no-mpi' in live_probe
    assert '"$HERE/scripts/paraview_python.sh"' in live_probe
    assert "SERVER_STATUS=124" in live_probe
    assert "mpi_images=active-exact" in (
        ROOT / "tests/python/integration/mpi/probe_catalyst_live_mpi.py"
    ).read_text(encoding="utf-8")
