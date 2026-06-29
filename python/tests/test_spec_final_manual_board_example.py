"""Final Spec corrective example: public route only, real toolchain execution."""

from pathlib import Path

import pytest

from pops.codegen.inspect_compiled import build_arguments

from examples.spec_final import manual_board_predictor_corrector_poisson_lorentz as example


FORBIDDEN_EXAMPLE_TOKENS = (
    "try:",
    "except ",
    "skip",
    "pops.compile(",
    "pops.bind",
    "pops.Problem",
    "pops.Case",
    "add_equation",
    "install_program",
    "_get_state",
    "_eval_rhs",
    "P.rhs",
    "P.solve_fields",
    "P.linear_source",
)


def test_manual_board_example_uses_only_final_public_route():
    """The public example must not document or rely on transitional routes."""
    path = Path(example.__file__)
    text = path.read_text(encoding="utf-8")
    offenders = [token for token in FORBIDDEN_EXAMPLE_TOKENS if token in text]
    assert offenders == []


def test_manual_board_arguments_treat_field_outputs_as_outputs_not_aux_inputs():
    """Poisson outputs are produced by the field operator; only B_z is a runtime aux input."""
    module = example.build_model()
    program = example.build_program(module)

    fake = type("FakeCompiled", (), {})()
    fake.model = module
    fake.program_model = module
    fake.program = program

    args = build_arguments(fake)
    assert set(args.aux) == {"B_z"}
    assert {"phi", "grad_x", "grad_y"} <= set(args.outputs)
    assert "phi" in args.solvers


@pytest.mark.requires_toolchain
def test_manual_board_example_compiles_installs_and_steps_under_toolchain():
    """TASK-065 acceptance: compile the final public example and execute one CFL step."""
    result = example.run_once(n=8, cfl=0.2)
    assert result["state"].shape == (3, 8, 8)
    assert {"phi", "grad_x", "grad_y"} <= set(result["fields"])
    assert {"mass", "rho_min", "rho_max"} <= set(result["diagnostics"])
    assert result["profile"] is not None
