"""Final moments route: generic builders and provided models lower to Module + Program."""

import importlib
from pathlib import Path

import pytest

import pops
from examples.spec_final import custom_moment_model
from examples.spec_final import provided_hyqmom15_model
from pops import model as model_api
from pops import moments


REPO_ROOT = Path(__file__).resolve().parents[2]


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
