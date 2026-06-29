"""Final moments route: generic builders and provided models lower to Module + Program."""

import importlib
import os
from pathlib import Path

import pytest

import pops
from pops.lib.models import Euler, Isothermal, IdealMHD
from pops import compile_problem
from pops.codegen import KokkosOpenMP, Production
from examples.spec_final import custom_moment_model
from examples.spec_final import provided_hyqmom15_model
from pops.mesh import CartesianMesh
from pops.mesh.layouts import Uniform
from pops import model as model_api
from pops import moments
from pops.time import Program


REPO_ROOT = Path(__file__).resolve().parents[3]


def _configure_source_tree_include():
    include = REPO_ROOT / "include"
    os.environ["POPS_INCLUDE"] = str(include)
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        os.environ.setdefault("POPS_KOKKOS_ROOT", conda_prefix)
        os.environ.setdefault("Kokkos_ROOT", conda_prefix)
    return str(include)


def test_custom_moments_builds_module_without_legacy_facade():
    module = custom_moment_model.build_model()
    assert isinstance(module, model_api.Module)
    assert set(module.list_operators()) >= {
        "flux",
        "source_default",
        "fields_from_state",
        "explicit_rate",
    }
    assert not hasattr(module, "_m")


def test_provided_hyqmom15_builds_module_without_custom_case_file():
    module = provided_hyqmom15_model.build_model()
    assert isinstance(module, model_api.Module)
    assert module.capabilities()["moment_model"] is True
    assert module.capabilities()["moment_order"] == 4
    assert set(module.list_operators()) >= {
        "flux",
        "source_default",
        "fields_from_state",
        "explicit_rate",
    }
    assert list((REPO_ROOT / "python" / "pops" / "lib" / "models").rglob("custom.py")) == []


def test_provided_fluid_models_build_modules():
    for builder, expected in (
        (Euler.model, "euler"),
        (Isothermal.model, "isothermal_euler"),
    ):
        module = builder().to_module()
        assert isinstance(module, model_api.Module)
        assert module.capabilities()["fluid_model"] is True
        assert module.capabilities()["equation"] == expected
        assert set(module.list_operators()) >= {"flux", "explicit_rate"}


def test_provided_mhd_model_builds_module():
    module = IdealMHD.model().to_module()
    assert isinstance(module, model_api.Module)
    assert module.capabilities()["mhd_model"] is True
    assert module.capabilities()["equation"] == "ideal_mhd"
    assert set(module.list_operators()) >= {"flux", "explicit_rate"}


@pytest.mark.requires_toolchain
def test_provided_mhd_model_compiles(tmp_path):
    include = _configure_source_tree_include()
    module = IdealMHD.model().to_module()
    T = Program("ideal_mhd_compile")
    T.bind_operators(module)
    U = T.state("U", block="mhd", space=module.state_spaces()["U"])
    R_n = T.call(module.operator_registry().get("explicit_rate"), U.n, name="R_n")
    T.define(U.next, U.n + T.dt * R_n)
    T.commit(U.next)
    T.validate()
    mesh = CartesianMesh(n=4, L=1.0, periodic=True)
    compiled = compile_problem(
        model=module,
        program=T,
        layout=Uniform(mesh),
        backend=Production(platform=KokkosOpenMP()),
        include=include,
    )
    cpp = Path(compiled.dump_cpp(tmp_path)).read_text(encoding="utf-8")
    assert "GeneratedModule::Operators::explicit_rate" in cpp


def test_lib_models_catalog_has_no_deferred_or_case_specific_placeholders():
    root = REPO_ROOT / "python" / "pops" / "lib" / "models"
    forbidden = ("DEFER", "later phase", "diocotron", "column reference")
    for path in root.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        for token in forbidden:
            assert token not in text, "%s contains forbidden token %r" % (path, token)


def test_moment_programs_are_operator_first_call_nodes():
    for example in (custom_moment_model, provided_hyqmom15_model):
        module = example.build_model()
        program = example.build_program(module)
        ops = [value.op for value in program._values]
        assert "call" in ops
        assert "rhs" not in ops
        assert "solve_fields" not in ops
        assert "linear_source" not in ops


def test_generic_moment_model_forbidden_shortcuts_are_absent():
    forbidden = (
        "add_vlasov_electric_source",
        "add_magnetic_source",
        "add_bgk_collision",
    )
    for name in forbidden:
        assert not hasattr(moments.MomentModel, name)


def test_lib_moments_package_is_not_public():
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("pops.lib.moments")


def test_high_order_exact_speeds_are_rejected_before_codegen():
    with pytest.raises(ValueError, match="exact_speeds=True"):
        moments.CartesianVelocityMoments(
            order=4,
            closure=moments.HyQMOM15Closure(),
            exact_speeds=True,
        )


@pytest.mark.requires_toolchain
def test_custom_moment_example_compiles(tmp_path):
    compiled = custom_moment_model.compile_example(n=4)
    assert compiled.problem_hash
    cpp = Path(compiled.dump_cpp(tmp_path)).read_text(encoding="utf-8")
    assert "GeneratedModule::Operators::explicit_rate" in cpp


@pytest.mark.requires_toolchain
def test_provided_hyqmom15_example_compiles(tmp_path):
    compiled = provided_hyqmom15_model.compile_example(n=4)
    assert compiled.problem_hash
    cpp = Path(compiled.dump_cpp(tmp_path)).read_text(encoding="utf-8")
    assert "GeneratedModule::Operators::explicit_rate" in cpp
