"""TASK-056: typed OutputPolicy / CheckpointPolicy through the final public route."""

from pathlib import Path

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
from pops.output import AllLevels, CheckpointPolicy, CoarseOnly, HDF5, OutputPolicy
from pops.runtime._output_driver import _format_token, fire_output_policies, policy_due
from pops.solvers.elliptic import GeometricMG
from pops.time.schedule import always, every, on_end, on_start


def _build_compiled(n=8):
    include = manual._configure_source_tree_include()
    mesh = CartesianMesh(n=n, L=1.0, periodic=True)
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


def _install(compiled, mesh, layout, outputs=()):
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
        outputs=list(outputs),
    )
    return sim


def test_policy_due_and_format_token_are_host_testable():
    assert policy_due(every(3), 3)
    assert policy_due(every(3), 6)
    assert not policy_due(every(3), 4)
    assert policy_due(5, 10)
    assert not policy_due(5, 7)
    assert policy_due(always(), 1)
    assert policy_due(None, 1)
    assert policy_due(on_start(), 0, phase="start")
    assert not policy_due(on_start(), 1, phase="step")
    assert policy_due(on_end(), 4, phase="end")
    assert not policy_due(on_end(), 4, phase="step")
    assert not policy_due(every(2), 0)
    assert _format_token(HDF5()) == "hdf5"
    assert _format_token(None) == "npz"


def test_fire_output_policies_rejects_non_policy():
    class Sim:
        pass

    with pytest.raises(TypeError, match="OutputPolicy"):
        fire_output_policies(Sim(), [object()], 1, "/tmp")


@pytest.mark.requires_toolchain
def test_output_policy_npz_cadence_and_contents(tmp_path):
    compiled, mesh, layout = _build_compiled(n=8)
    sim = _install(
        compiled,
        mesh,
        layout,
        outputs=[OutputPolicy(format=None, cadence=every(2), prefix="out")],
    )

    taken = sim.run(t_end=1.0, cfl=0.2, max_steps=4, output_dir=str(tmp_path))
    assert taken == 4

    present = sorted(p.name for p in tmp_path.iterdir() if p.name.startswith("out"))
    assert present == ["out_000002.npz", "out_000004.npz"]
    data = np.load(tmp_path / "out_000004.npz")
    assert "state_plasma" in data
    assert "phi" in data
    assert int(data["macro_step"]) == 4


@pytest.mark.requires_toolchain
def test_output_policy_start_end_and_field_selection(tmp_path):
    compiled, mesh, layout = _build_compiled(n=8)
    sim = _install(
        compiled,
        mesh,
        layout,
        outputs=[
            OutputPolicy(format=None, cadence=on_start(), prefix="start"),
            OutputPolicy(format=None, cadence=on_end(), prefix="end"),
            OutputPolicy(format=None, cadence=every(1), fields=["plasma"], prefix="sel"),
        ],
    )

    taken = sim.run(t_end=1.0, cfl=0.2, max_steps=2, output_dir=str(tmp_path))
    assert taken == 2
    present = sorted(p.name for p in tmp_path.iterdir() if p.suffix == ".npz")
    assert present == [
        "end_000002.npz",
        "sel_000001.npz",
        "sel_000002.npz",
        "start_000000.npz",
    ]
    selected = np.load(tmp_path / "sel_000001.npz")
    assert "state_plasma" in selected


@pytest.mark.requires_toolchain
def test_level_selection_is_noop_on_uniform_system(tmp_path):
    compiled, mesh, layout = _build_compiled(n=8)
    for level_policy, name in ((AllLevels(), "all"), (CoarseOnly(), "coarse")):
        out_dir = tmp_path / name
        out_dir.mkdir()
        sim = _install(
            compiled,
            mesh,
            layout,
            outputs=[OutputPolicy(format=None, cadence=every(1), levels=level_policy, prefix="lv")],
        )
        sim.run(t_end=1.0, cfl=0.2, max_steps=1, output_dir=str(out_dir))
        assert (out_dir / "lv_000001.npz").exists()


@pytest.mark.requires_toolchain
def test_checkpoint_policy_round_trips_public_state(tmp_path):
    compiled, mesh, layout = _build_compiled(n=8)
    sim = _install(
        compiled,
        mesh,
        layout,
        outputs=[CheckpointPolicy(cadence=every(2), restartable=True, prefix="ck")],
    )

    sim.run(t_end=1.0, cfl=0.2, max_steps=2, output_dir=str(tmp_path))
    checkpoints = sorted(p.name for p in tmp_path.iterdir() if p.name.startswith("ck"))
    assert checkpoints == ["ck_000002.npz"]
    reference = np.asarray(sim.get_state("plasma"))

    restored = _install(compiled, mesh, layout)
    restored.restart(str(tmp_path / "ck_000002"))
    assert restored.macro_step() == 2
    np.testing.assert_array_equal(np.asarray(restored.get_state("plasma")), reference)


def test_output_policy_file_does_not_use_legacy_runtime_route():
    text = Path(__file__).read_text(encoding="utf-8")
    forbidden = (
        "pops." + "Model",
        "_add" + "_block",
        "_set" + "_poisson",
        "_get" + "_state",
        "._output" + "_policies",
        "_output" + "_policies =",
    )
    offenders = [token for token in forbidden if token in text]
    assert not offenders
