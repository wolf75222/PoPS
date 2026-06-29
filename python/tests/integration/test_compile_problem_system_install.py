"""Integration: compile_problem -> System.install -> step_cfl public route."""

from pathlib import Path

import pytest

from examples.spec_final import manual_board_predictor_corrector_poisson_lorentz as manual
from pops import System, compile_problem
from pops.codegen import KokkosOpenMP, Production
from pops.mesh import CartesianMesh
from pops.mesh.layouts import Uniform


def test_generated_problem_source_is_combined_problem_route():
    module = manual.build_model()
    program = manual.build_program(module)
    cpp = program._emit_cpp_program_for_target(model=module)

    assert "pops_problem_install" in cpp
    assert "GeneratedModule::Operators::explicit_rate" in cpp
    assert ("pops_install_" + "program") not in cpp


@pytest.mark.requires_toolchain
def test_compile_problem_artifact_is_combined_problem_source(tmp_path):
    module = manual.build_model()
    program = manual.build_program(module)
    compiled = compile_problem(
        model=module,
        program=program,
        layout=Uniform(CartesianMesh(n=8, L=1.0, periodic=True)),
        backend=Production(platform=KokkosOpenMP()),
        include=manual._configure_source_tree_include(),
    )

    assert compiled.problem_hash
    assert compiled.module_hash
    assert compiled.program_hash

    cpp_path = compiled.dump_cpp(tmp_path)
    cpp = Path(cpp_path).read_text(encoding="utf-8")
    assert "pops_problem_install" in cpp
    assert "GeneratedModule::Operators::explicit_rate" in cpp
    assert ("pops_install_" + "program") not in cpp

    args = compiled.arguments()
    assert set(args.instances) == {"plasma"}
    assert set(args.aux) == {"B_z"}
    assert {"phi", "grad_x", "grad_y"} <= set(args.outputs)


@pytest.mark.requires_toolchain
def test_compile_problem_system_install_step_cfl_public_route_runs():
    result = manual.run_once(n=8, cfl=0.2)
    compiled = result["compiled"]

    assert isinstance(compiled.problem_hash, str) and compiled.problem_hash
    assert result["state"].shape == (3, 8, 8)
    assert {"mass", "rho_min", "rho_max"} <= set(result["diagnostics"])

    sim = System(layout=Uniform(CartesianMesh(n=8, L=1.0, periodic=True)))
    assert hasattr(sim, "install")
    assert not hasattr(sim, "install_" + "program")
    assert not hasattr(sim, "add_" + "equation")
