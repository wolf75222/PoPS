"""Final AMR public route: layout=AMR -> compile_problem -> System(layout)."""

import os

import numpy as np
import pytest

import pops
from examples.spec_final import amr_poisson_lorentz
from pops._capabilities import inspect_amr
from pops.codegen import Production
from pops.mesh import CartesianMesh
from pops.mesh.amr import CheckpointPolicy, FrozenRegrid, Refine, RegridEvery
from pops.mesh.layouts import AMR


def test_amr_layout_inspection_carries_all_policies():
    layout = amr_poisson_lorentz.build_layout(n=16)
    info = layout.inspect()
    opts = info["options"]
    for key in ("base", "max_levels", "ratio", "regrid", "patches",
                "refine", "nesting", "checkpoint", "output"):
        assert key in opts
    report = inspect_amr(layout).to_dict()
    slots = {row["slot"] for row in report["policies"]}
    assert {"regrid", "patches", "refine", "nesting", "checkpoint", "output"} <= slots


def test_system_layout_amr_constructs_amr_runtime():
    layout = AMR(
        CartesianMesh(n=16, L=1.0, periodic=True),
        max_levels=2,
        ratio=2,
        regrid=FrozenRegrid(),
    )
    sim = pops.System(layout=layout)
    assert isinstance(sim, pops.AmrSystem)
    assert getattr(sim, "_layout", None) is layout


def test_compile_problem_layout_amr_carries_inspection(monkeypatch, tmp_path):
    captured = {"target": None, "compiled": False}

    def fake_emit(self, model=None, target="system", problem_hash=None):
        captured["target"] = target
        return "extern \"C\" int pops_test_amr_final_route() { return 0; }\n"

    def fake_run_compile(cmd, label):
        captured["compiled"] = True

    def fake_loader_build_flags(cxx=None):
        return "c++", [], []

    monkeypatch.setattr(
        pops.time.Program,
        "_emit_cpp_program_for_target",
        fake_emit,
    )
    import pops.codegen.compile_drivers as drivers

    monkeypatch.setattr(drivers, "_run_compile", fake_run_compile)
    monkeypatch.setattr(drivers, "pops_loader_build_flags", fake_loader_build_flags)

    layout = amr_poisson_lorentz.build_layout(n=16)
    module = amr_poisson_lorentz.build_model()
    program = amr_poisson_lorentz.build_program(module)
    compiled = pops.compile_problem(
        os.path.join(str(tmp_path), "amr_problem.so"),
        model=module,
        program=program,
        layout=layout,
        backend=Production(),
        include=str(tmp_path),
        force=True,
    )

    # The public API is layout=AMR(...); the native target token is only the
    # internal codegen route selected from that descriptor.
    assert captured == {"target": "amr_system", "compiled": True}
    report = compiled.inspect_amr().to_dict()
    assert report["layout"] == "amr"
    assert report["max_levels"] == 2
    assert report["ratio"] == 2
    assert any(row["slot"] == "refine.criterion" for row in report["policies"])


def test_amr_refine_role_absent_rejected_before_codegen():
    layout = AMR(
        CartesianMesh(n=16),
        max_levels=2,
        ratio=2,
        refine=Refine.on("NotADeclaredRole").above(0.1),
    )
    module = amr_poisson_lorentz.build_model()
    program = amr_poisson_lorentz.build_program(module)
    with pytest.raises(ValueError, match="NotADeclaredRole"):
        pops.compile_problem(
            "/tmp/should_not_compile_missing_role.so",
            model=module,
            program=program,
            layout=layout,
            backend=Production(),
            force=True,
        )


def test_amr_checkpoint_policy_rejects_bit_identical_dynamic_regrid():
    layout = AMR(
        CartesianMesh(n=16),
        max_levels=2,
        ratio=2,
        regrid=RegridEvery(2),
        checkpoint=CheckpointPolicy(require_bit_identical=True),
    )
    with pytest.raises(ValueError, match="frozen AMR hierarchy"):
        layout.validate()


def test_amr_validate_does_not_mutate_module_state_space():
    layout = amr_poisson_lorentz.build_layout(n=16)
    module = amr_poisson_lorentz.build_model()
    before = module.state_spaces()["U"].components
    layout.validate(module)
    after = module.state_spaces()["U"].components
    assert after == before == ("rho", "mx", "my")


def test_amr_initial_state_routes_full_conservative_state():
    class Harness:
        def __init__(self):
            self.calls = []

        def set_density(self, name, value):
            self.calls.append(("density", name, np.asarray(value).shape))

        def set_conservative_state(self, name, value):
            self.calls.append(("state", name, np.asarray(value).shape))

    h = Harness()
    pops.AmrSystem._install_initial_state(h, "plasma", np.zeros((3, 8, 8)))
    pops.AmrSystem._install_initial_state(h, "density", np.zeros((8, 8)))
    assert h.calls == [
        ("state", "plasma", (3, 8, 8)),
        ("density", "density", (8, 8)),
    ]


def test_amr_public_get_state_reads_full_block_state():
    class Native:
        def nx(self):
            return 4

        def block_n_vars(self, name):
            assert name == "plasma"
            return 3

        def block_level_state(self, name, level):
            assert (name, level) == ("plasma", 0)
            return np.arange(3 * 4 * 4, dtype=np.float64)

    sim = object.__new__(pops.AmrSystem)
    sim._s = Native()

    state = sim.get_state("plasma")
    assert state.shape == (3, 4, 4)
    np.testing.assert_array_equal(state.ravel(), np.arange(3 * 4 * 4, dtype=np.float64))
