"""HDF5 output and checkpoint validation through the final public route."""

import numpy as np
import pytest

from examples.spec_final import manual_board_predictor_corrector_poisson_lorentz as manual
from pops import System, compile_problem
from pops.codegen import KokkosOpenMP, Production
from pops.mesh import CartesianMesh
from pops.mesh.layouts import Uniform
from pops.numerics.riemann import Rusanov
from pops.numerics.reconstruction import MUSCL
from pops.numerics.reconstruction.limiters import Minmod
from pops.numerics.spatial import FiniteVolume
from pops.output import HDF5, NPZ
from pops.solvers.elliptic import GeometricMG


@pytest.fixture(scope="module")
def compiled_case():
    include = manual._configure_source_tree_include()
    mesh = CartesianMesh(n=8, L=1.0, periodic=True)
    layout = Uniform(mesh)
    module = manual.build_model()
    program = manual.build_program(module)
    compiled = compile_problem(
        model=module,
        program=program,
        layout=layout,
        backend=Production(platform=KokkosOpenMP()),
        include=include,
    )
    return compiled, mesh, layout


def _install(compiled_case):
    compiled, mesh, layout = compiled_case
    sim = System(layout=layout)
    sim.install(
        compiled,
        instances={
            "plasma": {
                "initial": manual.initial_state(mesh),
                "spatial": FiniteVolume(
                    riemann=Rusanov(),
                    reconstruction=MUSCL(limiter=Minmod()),
                ),
            }
        },
        aux={"B_z": np.full((mesh.n, mesh.n), 0.2)},
        solvers={"phi": GeometricMG()},
    )
    return sim


def _step(sim, steps=3):
    for _ in range(steps):
        sim.step_cfl(0.2)
    return sim


def _read_h5(path):
    h5py = pytest.importorskip("h5py")
    out = {"attrs": {}, "blocks": {}}
    with h5py.File(path, "r") as f:
        for key in ("t", "macro_step", "nx", "ny", "abi_key"):
            out["attrs"][key] = f.attrs[key]
        out["phi"] = np.asarray(f["phi"][...])
        for block in f:
            if block == "phi":
                continue
            group = f[block]
            out["blocks"][block] = {
                "state": np.asarray(group["state"][...]),
                "names": [bytes(s) for s in group.attrs["names"]],
                "roles": [bytes(s) for s in group.attrs["roles"]],
            }
    return out


def _assert_dumps_equal(a, b):
    assert set(a["attrs"]) == set(b["attrs"])
    assert a["attrs"]["t"] == b["attrs"]["t"]
    assert int(a["attrs"]["macro_step"]) == int(b["attrs"]["macro_step"])
    assert int(a["attrs"]["nx"]) == int(b["attrs"]["nx"])
    assert int(a["attrs"]["ny"]) == int(b["attrs"]["ny"])
    assert str(a["attrs"]["abi_key"]) == str(b["attrs"]["abi_key"])
    np.testing.assert_array_equal(a["phi"], b["phi"])
    assert set(a["blocks"]) == set(b["blocks"])
    for block in a["blocks"]:
        np.testing.assert_array_equal(a["blocks"][block]["state"], b["blocks"][block]["state"])
        assert a["blocks"][block]["names"] == b["blocks"][block]["names"]
        assert a["blocks"][block]["roles"] == b["blocks"][block]["roles"]


@pytest.mark.requires_toolchain
def test_parallel_equals_serial_mono_rank(tmp_path, compiled_case):
    h5py = pytest.importorskip("h5py")
    if not h5py.get_config().mpi:
        pytest.skip("h5py is present without MPI support")
    pytest.importorskip("mpi4py", reason="mpi4py absent: mpio open not testable")

    sim = _step(_install(compiled_case))
    serial = sim.write(str(tmp_path / "ser"), format=HDF5(), parallel=False)
    parallel = sim.write(str(tmp_path / "par"), format=HDF5(), parallel=True)
    _assert_dumps_equal(_read_h5(serial), _read_h5(parallel))


@pytest.mark.requires_toolchain
def test_parallel_clear_error_when_h5py_without_mpi(tmp_path, compiled_case):
    h5py = pytest.importorskip("h5py")
    if h5py.get_config().mpi:
        pytest.skip("h5py is built with MPI; the non-MPI error path is not reproducible")

    sim = _step(_install(compiled_case))
    with pytest.raises(RuntimeError) as exc:
        sim.write(str(tmp_path / "x"), format=HDF5(), parallel=True)
    message = str(exc.value)
    assert "MPI" in message
    assert "parallel=False" in message


@pytest.mark.requires_toolchain
def test_serial_hdf5_default_matches_public_state(tmp_path, compiled_case):
    pytest.importorskip("h5py")
    sim = _step(_install(compiled_case))
    path = sim.write(str(tmp_path / "ser"), format=HDF5())
    dump = _read_h5(path)

    state = np.asarray(sim.get_state("plasma"))
    fields = sim.get_current_fields("plasma")
    np.testing.assert_array_equal(dump["blocks"]["plasma"]["state"], state)
    np.testing.assert_array_equal(dump["phi"], np.asarray(fields["phi"]))


@pytest.mark.requires_toolchain
def test_checkpoint_parallel_true_rejected(compiled_case):
    sim = _install(compiled_case)
    with pytest.raises(ValueError) as exc:
        sim.checkpoint("ignored_path", parallel=True)
    assert "parallel=False" in str(exc.value)


@pytest.mark.requires_toolchain
def test_parallel_rejected_for_non_hdf5(tmp_path, compiled_case):
    sim = _install(compiled_case)
    with pytest.raises(ValueError) as exc:
        sim.write(str(tmp_path / "x"), format=NPZ(), parallel=True)
    assert "hdf5" in str(exc.value)


def test_hdf5_parallel_file_does_not_use_legacy_runtime_route():
    text = __import__("pathlib").Path(__file__).read_text(encoding="utf-8")
    forbidden = (
        "pops." + "Model",
        "_add" + "_block",
        "_set" + "_poisson",
        "_get" + "_state",
    )
    offenders = [token for token in forbidden if token in text]
    assert not offenders
