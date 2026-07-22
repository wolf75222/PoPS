from pathlib import Path


REPOSITORY = Path(__file__).resolve().parents[3]


def test_environment_selects_one_exact_conda_mpi_implementation() -> None:
    environment = (REPOSITORY / "environment.yml").read_text(encoding="utf-8")

    assert '"mpi=1.0=openmpi"' in environment
    assert "- openmpi" in environment
    assert '"hdf5=*=mpi_openmpi_*"' in environment
    assert "mpi=1.0=mpich" not in environment


def test_collective_hdf5_discovery_cannot_select_a_serial_config_package() -> None:
    cmake = (REPOSITORY / "CMakeLists.txt").read_text(encoding="utf-8")

    config_guard = "set(HDF5_NO_FIND_PACKAGE_CONFIG_FILE TRUE)"
    prefer_parallel = "set(HDF5_PREFER_PARALLEL TRUE)"
    discovery = "find_package(HDF5 MODULE REQUIRED COMPONENTS C)"
    parallel_guard = "if(NOT HDF5_IS_PARALLEL)"
    assert cmake.index(config_guard) < cmake.index(prefer_parallel) \
        < cmake.index(discovery) < cmake.index(parallel_guard)


def test_mpi_python_build_bounds_hdf5_to_the_active_env_without_clobbering_override() -> None:
    build = (REPOSITORY / "scripts" / "build_python.sh").read_text(encoding="utf-8")

    preserve_override = 'if [[ -z "${HDF5_ROOT+x}" ]]; then'
    default_root = 'export HDF5_ROOT="$CONDA_PREFIX"'
    build_command = 'python -m pip "${pip_args[@]}"'
    assert build.index(preserve_override) < build.index(default_root) < build.index(build_command)
    assert "export HDF5_ROOT=\"$CONDA_PREFIX\"" in build


def test_setup_authenticates_the_solved_openmpi_parallel_hdf5_stack() -> None:
    setup = (REPOSITORY / "scripts" / "setup_env.sh").read_text(encoding="utf-8")

    update = '"$PKG" env update -n "$ENV_NAME"'
    mpi_provider = 'mpi_provider="$(conda list -n "$ENV_NAME" mpi'
    hdf5_variant = '"$hdf5_build" != mpi_openmpi_*'
    wrapper = 'h5pcc="$POPS_PREFIX/bin/h5pcc"'
    parallel_probe = "grep -Eqi 'Parallel HDF5:[[:space:]]*(yes|on)'"
    persist = 'conda env config vars set -n "$ENV_NAME"'
    assert setup.index(update) < setup.index(mpi_provider)
    assert setup.index(mpi_provider) < setup.index(wrapper)
    assert setup.index(wrapper) < setup.index(hdf5_variant)
    assert setup.index(hdf5_variant) < setup.index(parallel_probe) < setup.index(persist)
    assert 'openmpi_version="$(conda list -n "$ENV_NAME" openmpi' in setup
    assert "OMPI_CC=/usr/bin/clang OMPI_CXX=/usr/bin/clang++" in setup
    assert "H5Pset_fapl_mpio(access, MPI_COMM_WORLD, MPI_INFO_NULL)" in setup
    assert 'conda run -n "$ENV_NAME" "$h5pcc" "$probe_source"' in setup
    assert 'conda run -n "$ENV_NAME" "$probe_binary"' in setup


def test_macos_build_keeps_openmpi_wrappers_on_the_pinned_compiler() -> None:
    build = (REPOSITORY / "scripts" / "build_python.sh").read_text(encoding="utf-8")

    activate = 'conda activate "$ENV_NAME"'
    platform_guard = 'if [[ "$(uname)" == "Darwin" ]]; then'
    mpi_cc = 'export OMPI_CC="${OMPI_CC:-${CC:-/usr/bin/clang}}"'
    mpi_cxx = 'export OMPI_CXX="${OMPI_CXX:-${CXX:-/usr/bin/clang++}}"'
    build_command = 'python -m pip "${pip_args[@]}"'
    assert build.index(activate) < build.index(platform_guard) < build.index(mpi_cc)
    assert build.index(mpi_cc) < build.index(mpi_cxx) < build.index(build_command)
